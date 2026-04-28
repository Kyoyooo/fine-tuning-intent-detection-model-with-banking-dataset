#!/bin/bash

echo "Starting the training process for Banking Intent Detection..."

# Chạy script huấn luyện
python scripts/train.py

echo "Training completed. Check the 'outputs/' directory for model checkpoints."
