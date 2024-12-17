#!/bin/bash
set -x

cd scripts
python convert_rwkv7_checkpoint_to_hf.py --repo_id BlinkDL/rwkv-7-world \
 --checkpoint_file RWKV-x070-World-1B6-v2-20240208-ctx4096.pth \
 --output_dir ../../rwkv_model/rwkv7-world-1b6/ \
 --tokenizer_file ../rwkv5_world_tokenizer \
 --size 1B6 \
 --is_world_tokenizer True

cp ../rwkv6_world_tokenizer/added_tokens.json ../../rwkv_model/rwkv7-world-1b6/
cp ../rwkv6_world_tokenizer/hf_rwkv_tokenizer.py ../../rwkv_model/rwkv7-world-1b6/
cp ../rwkv6_world_tokenizer/special_tokens_map.json ../../rwkv_model/rwkv7-world-1b6/
cp ../rwkv6_world_tokenizer/rwkv_vocab_v20230424.txt ../../rwkv_model/rwkv7-world-1b6/
cp ../rwkv6_world_tokenizer/tokenizer_config.json ../../rwkv_model/rwkv7-world-1b6/
cp ../rwkv7_model/configuration_rwkv7.py ../../rwkv_model/rwkv7-world-1b6/
cp ../rwkv7_model/modeling_rwkv7.py ../../rwkv_model/rwkv7-world-1b6/
cp ../rwkv7_model/generation_config.json ../../rwkv_model/rwkv7-world-1b6/
