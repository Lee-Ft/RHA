#!/usr/bin/env bash

release_path="/home/data/tvqa_plus_stage_features"
test_path=${release_path}/tvqa_plus_test_preprocessed_no_anno.json
valid_path=${release_path}/tvqa_plus_valid_preprocessed_no_anno.json
model_dir=$1
mode=$2
python inference.py \
--model_dir=${model_dir} \
--mode=${mode} \
--test_path=${test_path} \
--inference_mode
