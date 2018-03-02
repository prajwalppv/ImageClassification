#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/main.py \
  --model_name="conv_net" \
  --reset_output_dir \
  --data_path="/Users/prajwalvasisht/Desktop/CMU/Spring 2018/11695 - Competitive Engineering/HW1/cifar-10-batches-py" \
  --output_dir="outputs" \
  --batch_size=32 \
  --num_epochs=10 \
  --log_every=100 \
  --eval_every_epochs=1 \
  "$@"