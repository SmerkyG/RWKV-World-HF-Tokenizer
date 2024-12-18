# coding=utf-8
# Copyright 2024 The RWKV team and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RWKV7 World model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from pathlib import Path

import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_utils import PreTrainedModel, GenerationMixin, _init_weights
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_ninja_available,
    is_torch_cuda_available,
    logging,
)

from .configuration_rwkv7 import Rwkv7Config

# MIT License

# Copyright (c) 2024 Songlin Yang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) 2024, Johan Sokrates Wind

import torch as th
import triton
import triton.language as tl

@triton.jit
def IND4(a,b,c,d,nb,nc,nd):
    return ((a*nb+b)*nc+c)*nd+d
@triton.jit
def IND5(a,b,c,d,e,nb,nc,nd,ne):
    return (((a*nb+b)*nc+c)*nd+d)*ne+e

@triton.jit
def _prod(a,b): return a*b

# inv(I-A) where A is a strictly lower triangular nxn matrix
@triton.jit
def tri_minv(A, n:tl.constexpr, prec:tl.constexpr):
    i = tl.arange(0,n)
    prod = (i[None,:]==i[:,None]).to(tl.float32)
    for j in range(n-1):
        prod += tl_dot(prec, prod, (A*((i[None,:]==j)*(i[:,None]>i[None,:]))).trans())
    return prod.trans()

@triton.jit
def fw_attn_triton(w_,q_,k_,v_,a_,b_, s0_,y_,s_,sT_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr):
    bi = tl.program_id(1)
    hi = tl.program_id(0)

    i = tl.arange(0,C)[None,:]
    state = tl.load(s0_+IND4(bi,hi,i.trans(),i, H,C,C)).to(tl.float32)
    for t0 in range(T//dT):
        t = t0*dT+tl.arange(0,dT)[:,None]
        sw = tl.load(w_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sq = tl.load(q_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sk = tl.load(k_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sa = tl.load(a_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sb = tl.load(b_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

        w = (-sw.exp()).exp()
        fw = tl.reduce(w, 0, _prod, keep_dims=True)
        incl_pref = tl.cumprod(w,axis=0)
        non_incl_pref = incl_pref / w
        inv_incl_pref = 1 / incl_pref

        wq = sq * incl_pref
        wa = sa * non_incl_pref
        kwi = sk * inv_incl_pref
        bwi = sb * inv_incl_pref

        mask1 = (t > t.trans())
        ab = tl_dot(prec, wa, bwi.trans()) * mask1
        ak = tl_dot(prec, wa, kwi.trans()) * mask1

        ab_inv = tri_minv(ab, dT, prec)

        ab_u = tl_dot(prec, ak, sv) + tl_dot(prec, wa, state.trans())
        u = tl_dot(prec, ab_inv, ab_u)
        mask2 = (t >= t.trans())
        qk = tl_dot(prec, wq, kwi.trans()) * mask2
        qb = tl_dot(prec, wq, bwi.trans()) * mask2
        yy = tl_dot(prec, qk, sv) + tl_dot(prec, qb, u) + tl_dot(prec, wq, state.trans())
        tl.store(y_+IND4(bi,t,hi,i, T,H,C), yy.to(tl.bfloat16))

        tl.store(s_+IND5(bi,hi,t0,i.trans(),i, H,T//dT,C,C), state.to(tl.float32))
        state = state * fw + tl_dot(prec, sv.trans(), kwi*fw) + tl_dot(prec, u.trans(), bwi*fw)
    tl.store(sT_+IND4(bi,hi,i.trans(),i, H,C,C), state.to(tl.bfloat16))

@triton.jit
def bw_attn_triton(w_,q_,k_,v_,a_,b_, dy_,s_,dsT_, dw_,dq_,dk_,dv_,da_,db_,ds0_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr):
    bi = tl.program_id(1)
    hi = tl.program_id(0)

    i = tl.arange(0,C)[None,:]
    dstate = tl.load(dsT_+IND4(bi,hi,i.trans(),i, H,C,C)).to(tl.float32)

    for t0 in range(T//dT-1,-1,-1):
        t = t0*dT+tl.arange(0,dT)[:,None]

        state = tl.load(s_+IND5(bi,hi,t0,i.trans(),i, H,T//dT,C,C)).to(tl.float32)

        sw = tl.load(w_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sq = tl.load(q_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sk = tl.load(k_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sa = tl.load(a_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sb = tl.load(b_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sdy = tl.load(dy_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

        dw_fac = -sw.exp()
        w = dw_fac.exp()
        fw = tl.reduce(w, 0, _prod, keep_dims=True)
        incl_pref = tl.cumprod(w,axis=0)
        non_incl_pref = incl_pref / w
        inv_incl_pref = 1 / incl_pref

        wq = sq * incl_pref
        wa = sa * non_incl_pref
        kwi = sk * inv_incl_pref
        bwi = sb * inv_incl_pref

        mask1 = (t > t.trans())
        ab = tl_dot(prec, wa, bwi.trans()) * mask1
        ak = tl_dot(prec, wa, kwi.trans()) * mask1

        ab_inv = tri_minv(ab, dT, prec)

        ab_u = tl_dot(prec, ak, sv) + tl_dot(prec, wa, state.trans())
        u = tl_dot(prec, ab_inv, ab_u)
        mask2 = (t >= t.trans())
        qk = tl_dot(prec, wq, kwi.trans()) * mask2
        qb = tl_dot(prec, wq, bwi.trans()) * mask2

        du = tl_dot(prec, qb.trans(), sdy) + tl_dot(prec, bwi*fw, dstate.trans())
        dab_u = tl_dot(prec, ab_inv.trans(), du)

        dv = tl_dot(prec, qk.trans(), sdy) + tl_dot(prec, kwi*fw, dstate.trans()) + tl_dot(prec, ak.trans(), dab_u)
        tl.store(dv_+IND4(bi,t,hi,i, T,H,C), dv.to(tl.bfloat16))

        dab = tl_dot(prec, tl_dot(prec, ab_inv.trans(), du), u.trans()) * mask1
        dak = tl_dot(prec, dab_u, sv.trans()) * mask1
        dab_u_state = tl_dot(prec, dab_u, state)
        da = non_incl_pref * (tl_dot(prec, dab, bwi) + tl_dot(prec, dak, kwi) + dab_u_state)
        tl.store(da_+IND4(bi,t,hi,i, T,H,C), da.to(tl.bfloat16))

        dqb = tl_dot(prec, sdy, u.trans()) * mask2
        dqk = tl_dot(prec, sdy, sv.trans()) * mask2
        dy_state = tl_dot(prec, sdy, state)
        dq = incl_pref * (tl_dot(prec, dqb, bwi) + tl_dot(prec, dqk, kwi) + dy_state)
        tl.store(dq_+IND4(bi,t,hi,i, T,H,C), dq.to(tl.bfloat16))

        fw_u_dstate = fw * tl_dot(prec, u, dstate)
        db = inv_incl_pref * (tl_dot(prec, dab.trans(), wa) + tl_dot(prec, dqb.trans(), wq) + fw_u_dstate)
        tl.store(db_+IND4(bi,t,hi,i, T,H,C), db.to(tl.bfloat16))

        fw_v_dstate = fw * tl_dot(prec, sv, dstate)
        dk = inv_incl_pref * (tl_dot(prec, dak.trans(), wa) + tl_dot(prec, dqk.trans(), wq) + fw_v_dstate)
        tl.store(dk_+IND4(bi,t,hi,i, T,H,C), dk.to(tl.bfloat16))

        dw0 = fw * tl.sum(state*dstate, axis=0,keep_dims=True)
        for k in range(t0*dT,t0*dT+dT):
            lmask = (t<k).trans()
            A = (tl_dot(prec, dab*lmask, bwi) + tl_dot(prec, dak*lmask, kwi)) * wa * (t>k)
            A += (tl_dot(prec, dqb*lmask, bwi) + tl_dot(prec, dqk*lmask, kwi)) * wq * (t>=k)
            A += (fw_v_dstate*kwi + fw_u_dstate*bwi) * (t<k)
            A += dab_u_state*wa * (t>k) + dy_state*wq * (t>=k)
            dw = tl.sum(A, axis=0,keep_dims=True) + dw0

            wk = tl.load(w_+IND4(bi,k,hi,i, T,H,C)).to(tl.float32)
            dw *= -wk.exp()
            tl.store(dw_+IND4(bi,k,hi,i, T,H,C), dw.to(tl.bfloat16))

        dstate = dstate * fw + tl_dot(prec, sdy.trans(), wq) + tl_dot(prec, dab_u.trans(), wa)
    tl.store(ds0_+IND4(bi,hi,i.trans(),i, H,C,C), dstate.to(tl.bfloat16))


class TritonRWKV7(th.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b,s0, dot_prec):
        K = 16
        B,T,H,C = w.shape
        s0 = th.zeros(B,H,C,C, dtype=w.dtype,device=w.device) if s0 is None else s0
        y = th.empty_like(v)
        sT = th.empty_like(s0)
        s = th.zeros(B,H,T//K,C,C, dtype=th.float32,device=w.device)
        fw_attn_triton[(H,B)](w,q,k,v,z,b, s0,y,s,sT, B,T,H,C,K, dot_prec)
        ctx.dot_prec = dot_prec
        ctx.save_for_backward(w,q,k,v,z,b,s)
        return y, sT
    @staticmethod
    def backward(ctx, dy, dsT):
        K = 16
        w,q,k,v,z,b,s = ctx.saved_tensors
        B,T,H,C = w.shape
        dw,dq,dk,dv,dz,db,ds0 = [th.empty_like(x) for x in [w,q,k,v,z,b,dsT]]
        bw_attn_triton[(H,B)](w,q,k,v,z,b, dy,s,dsT, dw,dq,dk,dv,dz,db,ds0, B,T,H,C,K, ctx.dot_prec)
        return dw,dq,dk,dv,dz,db,ds0,None

@triton.jit
def tl_dot(prec:tl.constexpr, a, b) -> torch.Tensor:
    if prec == 'fp32':
        return tl.dot(a.to(tl.float32),b.trans().to(tl.float32).trans(), allow_tf32=False)
    elif prec == 'tf32':
        return tl.dot(a.to(tl.float32),b.trans().to(tl.float32).trans(), allow_tf32=True)
    elif prec == 'bf16':
        return tl.dot(a.to(tl.bfloat16),b.trans().to(tl.bfloat16).trans(), allow_tf32=True)
    else:
        tl.static_assert(False)

def rwkv7_attn_triton(r,w,k,v,a,b, HEAD_SIZE, dot_prec = 'fp32'):
    B,T,HC = w.shape
    C = HEAD_SIZE
    H = HC//C
    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
    s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
    return TritonRWKV7.apply(w,r,k,v,a,b,s0,dot_prec)[0].view(B,T,HC)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "RWKV/v7-Goose-1.6B-Pile-HF"
_CONFIG_FOR_DOC = "Rwkv7Config"

class Rwkv7SelfAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        C = hidden_size = config.hidden_size
        attention_hidden_size = config.attention_hidden_size
        self.attention_hidden_size = attention_hidden_size
        H = self.num_heads = attention_hidden_size // config.head_size
        N = self.head_size = config.head_size

        calc_lora_rank = lambda exponent, multiplier: max(1, round(hidden_size ** exponent * multiplier / 32)) * 32
        lora_rank_decay = config.lora_rank_decay or calc_lora_rank(0.5, 1.8)
        lora_rank_iclr = config.lora_rank_iclr or calc_lora_rank(0.5, 1.8)
        lora_rank_value_residual_mix = config.lora_rank_value_residual_mix or calc_lora_rank(0.5, 1.3)
        lora_rank_gate = config.lora_rank_gate or calc_lora_rank(0.8, 0.6)

        self.x_r = nn.Parameter(torch.empty(1,1,C))
        self.x_w = nn.Parameter(torch.empty(1,1,C))
        self.x_k = nn.Parameter(torch.empty(1,1,C))
        self.x_v = nn.Parameter(torch.empty(1,1,C))
        self.x_a = nn.Parameter(torch.empty(1,1,C))
        self.x_g = nn.Parameter(torch.empty(1,1,C))

        self.w0 = nn.Parameter(torch.empty(1,1,C))
        self.w1 = nn.Parameter(torch.empty(C, lora_rank_decay))
        self.w2 = nn.Parameter(torch.empty(lora_rank_decay, C))

        self.a0 = nn.Parameter(torch.empty(1,1,C))
        self.a1 = nn.Parameter(torch.empty(C, lora_rank_iclr))
        self.a2 = nn.Parameter(torch.empty(lora_rank_iclr, C))

        if layer_id > 0:
            self.v0 = nn.Parameter(torch.empty(1,1,C))
            self.v1 = nn.Parameter(torch.empty(C, lora_rank_value_residual_mix))
            self.v2 = nn.Parameter(torch.empty(lora_rank_value_residual_mix, C))

        self.g1 = nn.Parameter(torch.empty(C, lora_rank_gate))
        self.g2 = nn.Parameter(torch.empty(lora_rank_gate, C))

        self.k_k = nn.Parameter(torch.empty(1,1,C))
        self.k_a = nn.Parameter(torch.empty(1,1,C))
        self.r_k = nn.Parameter(torch.empty(H,N))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=self.head_size * 1e-5)


    def forward(self, hidden, state=None, v_first=None, use_cache=False, seq_mode=True):        
        # Mix hidden with the previous timestep to produce key, value, receptance
        if hidden.size(1) == 1 and state is not None:
            shifted = state[0][self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[0][self.layer_id]
        if len(shifted.size()) == 2:
            shifted = shifted.unsqueeze(1)

        x = hidden

        B, T, C = hidden.shape
        H = self.num_heads
        N = self.head_size

        xx = shifted - x

        xr = x+xx*self.x_r
        xw = x+xx*self.x_w
        xk = x+xx*self.x_k
        xv = x+xx*self.x_v
        xa = x+xx*self.x_a
        xg = x+xx*self.x_g

        r = self.receptance(xr)
        w = torch.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = torch.nn.functional.normalize((k * self.k_k).view(B,T,H,-1), dim=-1, p=2.0).view(B,T,-1)
        k = k * (1 + (a-1) * self.k_a)
        if self.layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)        

        if T == 1 or not self.training:
            w = torch.exp(-0.606531 * torch.sigmoid((self.w0 + w).float())) # 0.606531 = exp(-0.5)
            vk_state = state[1][self.layer_id]
            for t in range(T):
                r_, w_, k_, v_, kk_, a_ = r[:,t], w[:,t], k[:,t], v[:,t], kk[:,t], a[:,t]
                vk = v_.view(B,H,N,1) @ k_.view(B,H,1,N)
                ab = (-kk_).view(B,H,N,1) @ (kk_*a_).view(B,H,1,N)
                vk_state = vk_state * w_.view(B,H,1,N) + vk_state @ ab.float() + vk.float()
                xx[:,t] = (vk_state.to(dtype=x.dtype) @ r_.view(B,H,N,1)).view(B,H*N)
            state[1][self.layer_id] = vk_state
            # FIXME - support fast triton kernel for non-training pre-fill with state in and out
        else:
            w = -torch.nn.functional.softplus(-(self.w0 + w)) - 0.5
            rwkv7_attn_triton(r, w, k, v, -kk, kk*a, self.head_size)
            
        xx = torch.nn.functional.group_norm(xx.view(B*T,H*N), num_groups=H, weight=self.ln_x.weight, bias=self.ln_x.bias, eps = self.ln_x.eps).view(B,T,H*N)
        xx = xx + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        xx = self.output(xx * g)

        if state is not None:
            state[0][self.layer_id] = hidden[:, -1]
        
        return xx, state, v_first


class Rwkv7FeedForward(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        intermediate_size = (
            config.intermediate_size
            if config.intermediate_size is not None
            else int(config.hidden_size * 4)
        )


        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.x_k = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden, state=None):
        if hidden.size(1) == 1 and state is not None:
            shifted = state[2][self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[2][self.layer_id]
        if len(shifted.size()) == 2:
            shifted = shifted.unsqueeze(1)

        delta_hidden_to_shifted = shifted - hidden
        key = hidden + delta_hidden_to_shifted * self.x_k

        key = torch.square(torch.relu(self.key(key)))
        value = self.value(key)

        if state is not None:
            state[2][self.layer_id] = hidden[:, -1]

        return value, state


class Rwkv7Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.attention = Rwkv7SelfAttention(config, layer_id)
        self.feed_forward = Rwkv7FeedForward(config, layer_id)

    def forward(self, hidden, state=None, v_first=None, use_cache=False, output_attentions=False, seq_mode=True):
        attention, state, v_first = self.attention(self.ln1(hidden), state=state, v_first=v_first, use_cache=use_cache, seq_mode=seq_mode)
        hidden = hidden + attention

        feed_forward, state = self.feed_forward(self.ln2(hidden), state=state)
        hidden = hidden + feed_forward

        outputs = (hidden, state, v_first)
        if output_attentions:
            outputs += (attention,)
        else:
            outputs += (None,)

        return outputs


class Rwkv7PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Rwkv7Config
    base_model_prefix = "rwkv7"
    _no_split_modules = ["Rwkv7Block"]
    _keep_in_fp32_modules = []
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        return
        
        """Initialize the weights."""
        if isinstance(module, Rwkv7SelfAttention):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size
            attention_hidden_size = module.attention_hidden_size
            head_size = module.config.head_size
            num_heads = attention_hidden_size // head_size

            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.x_k.dtype,
                device=module.x_k.device,
            )
            time_weight = time_weight[None, None, :]

            decay_speed = [
                -7.0 + 5.0 * (n / (attention_hidden_size - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
                for n in range(attention_hidden_size)
            ]
            decay_speed = torch.tensor(decay_speed, dtype=module.w0.dtype, device=module.w0.device)

            with torch.no_grad():
                module.x_r.copy_( 1.0 - torch.pow(time_weight, 0.2 * ratio_1_to_almost0) )
                module.x_w.copy_( 1.0 - torch.pow(time_weight, 0.9 * ratio_1_to_almost0) )
                module.x_k.copy_( 1.0 - (torch.pow(time_weight, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1) )
                module.x_v.copy_( 1.0 - (torch.pow(time_weight, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1) )
                module.x_a.copy_( 1.0 - torch.pow(time_weight, 0.9 * ratio_1_to_almost0) )
                module.x_g.copy_( 1.0 - torch.pow(time_weight, 0.2 * ratio_1_to_almost0) )

                def ortho_init(x, scale):
                    with torch.no_grad():
                        shape = x.shape
                        if len(shape) == 2:
                            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                            nn.init.orthogonal_(x, gain=gain * scale)
                        elif len(shape) == 3:
                            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                            for i in range(shape[0]):
                                nn.init.orthogonal_(x[i], gain=gain * scale)
                        else:
                            assert False
                        return x

                module.w0.copy_(decay_speed.reshape(1,1,attention_hidden_size) + 0.5) # !!! 0.5 comes from F.softplus !!!
                module.w1.zero_()
                ortho_init(module.w2, 0.1)

                module.a0.zero_()
                module.a1.zero_()
                ortho_init(module.a2, 0.1)

                module.v0.copy_(1.0)
                module.v1.zero_()
                ortho_init(module.v2, 0.1)

                module.g1.zero_()
                ortho_init(module.g2, 0.1)

                self.k_k.copy_(0.85)
                self.k_a.copy_(1.0)
                self.r_k.zero_()

                module.receptance.weight.data.uniform_(-0.5/(hidden_size**0.5), 0.5/(attention_hidden_size**0.5))
                module.key.weight.data.uniform_(-0.05/(hidden_size**0.5), 0.05/(attention_hidden_size**0.5))
                module.value.weight.data.uniform_(-0.5/(hidden_size**0.5), 0.5/(attention_hidden_size**0.5))
                module.output.weight.data.zero_()

        elif isinstance(module, Rwkv7FeedForward):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size

            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.x_k.dtype,
                device=module.x_k.device,
            )
            time_weight = time_weight[None, None, :]

            with torch.no_grad():                
                module.x_k.copy_( 1.0 - torch.pow(time_weight, ratio_1_to_almost0**4) )

                self.key.weight.data.uniform_(-0.5/(hidden_size**0.5), 0.5/(hidden_size**0.5))
                self.value.weight.data.zero_()

@dataclass
class Rwkv7Output(ModelOutput):
    """
    Class for the RWKV model outputs.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    state: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Rwkv7CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    state: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


RWKV7_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.) This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Parameters:
        config ([`Rwkv7Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

RWKV7_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary. If `past_key_values` is used, only `input_ids` that do not have their
            past calculated should be passed as `input_ids`. Indices can be obtained using [`AutoTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        state (tuple of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the last state is returned and can be used to quickly generate the next logits.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare RWKV7 Model transformer outputting raw hidden-states without any specific head on top.",
    RWKV7_START_DOCSTRING,
)
class Rwkv7Model(Rwkv7PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pre_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.blocks = nn.ModuleList([Rwkv7Block(config, layer_id=idx) for idx in range(config.num_hidden_layers)])
        self.ln_out = nn.LayerNorm(config.hidden_size)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(RWKV7_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Rwkv7Output,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Rwkv7Output]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if state is None:
            state = []
            head_size = self.config.head_size
            num_heads = self.config.attention_hidden_size // head_size
            state_attn_x = torch.zeros(
                    (self.config.num_hidden_layers, inputs_embeds.size(0), self.config.hidden_size),
                    dtype=inputs_embeds.dtype,
                    requires_grad=False,
                    device=inputs_embeds.device,
                ).contiguous()
            state_attn_vk = torch.zeros(
                    (
                        self.config.num_hidden_layers,
                        inputs_embeds.size(0),
                        num_heads,
                        head_size,
                        head_size,
                    ),
                    dtype=torch.float32,
                    requires_grad=False,
                    device=inputs_embeds.device,
                ).contiguous()
            state_ffn_x = torch.zeros(
                    (self.config.num_hidden_layers, inputs_embeds.size(0), self.config.hidden_size),
                    dtype=inputs_embeds.dtype,
                    requires_grad=False,
                    device=inputs_embeds.device,
                ).contiguous()
            state.append(state_attn_x)
            state.append(state_attn_vk)
            state.append(state_ffn_x)

        seq_mode = inputs_embeds.shape[1] > 1
        hidden_states = self.pre_ln(inputs_embeds)
        v_first = None

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for idx, block in enumerate(self.blocks):
            hidden_states, state, v_first, attentions = block(
                hidden_states, state=state, v_first=v_first, use_cache=use_cache, output_attentions=output_attentions, seq_mode=seq_mode
            )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (attentions,)

        hidden_states = self.ln_out(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return (hidden_states, state, all_hidden_states, all_self_attentions)

        return Rwkv7Output(
            last_hidden_state=hidden_states,
            state=state,
            hidden_states=all_hidden_states,  # None
            attentions=all_self_attentions,  # None
        )

# copied from HuggingFace https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py
@add_start_docstrings(
    """
    The RWKV7 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    RWKV7_START_DOCSTRING,
)
class Rwkv7ForCausalLM(Rwkv7PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Rwkv7Model(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, state=None, inputs_embeds=None, **kwargs):
        # only last token for inputs_ids if the state is passed along.
        if state is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and state is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["state"] = state
        return model_inputs

    @add_start_docstrings_to_model_forward(RWKV7_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Rwkv7CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Rwkv7CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        logits = self.head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Rwkv7CausalLMOutput(
            loss=loss,
            logits=logits,
            state=outputs.state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
