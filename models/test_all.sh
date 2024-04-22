#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

for model_config in config/*
do
    echo "Testing model $model_config"
    model_file="trained/$(basename ${model_config/.json/.torch})"

    # Run all tests at once for speed (high memory usage)
    python3 -u ../src/model_runner.py --data-path ../data/raw/ --dataset topstocks --model-config "$model_config" --model-file "$model_file" test &
done