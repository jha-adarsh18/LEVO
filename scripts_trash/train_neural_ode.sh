#!/bin/bash
# Neural ODE Training Launcher Script
# Usage: ./train_neural_ode.sh

set -e  # Exit on any error

echo "🚀 Neural ODE Event Pose Prediction Training"
echo "=============================================="

# Configuration - MODIFY THESE PATHS FOR YOUR SETUP
DATASET_ROOT="/kaggle/input/mvsec/dataset/train"  # Your dataset path
EXPERIMENT_DIR="./experiments"                           # Where to save results
CONDA_ENV="neural_ode_env"                              # Your conda environment (optional)

# Training Parameters - ADJUST AS NEEDED
BATCH_SIZE=8
NUM_EPOCHS=100
LEARNING_RATE=0.001
LATENT_DIM=256
NUM_EVENTS=1024
NUM_TIME_STEPS=10

# GPU Settings
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0, set to "" for CPU only

echo "📊 Training Configuration:"
echo "   Dataset: $DATASET_ROOT"
echo "   Batch Size: $BATCH_SIZE"
echo "   Epochs: $NUM_EPOCHS"
echo "   Learning Rate: $LEARNING_RATE"
echo "   Latent Dim: $LATENT_DIM"
echo "   Events per Sample: $NUM_EVENTS"
echo ""

# Check if dataset exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo "❌ Error: Dataset directory not found: $DATASET_ROOT"
    echo "Please update DATASET_ROOT in this script with your actual path"
    exit 1
fi

# Create experiment directory
mkdir -p $EXPERIMENT_DIR
echo "📁 Results will be saved to: $EXPERIMENT_DIR"

# Activate conda environment if specified
if [ ! -z "$CONDA_ENV" ]; then
    echo "🐍 Activating conda environment: $CONDA_ENV"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
fi

# Check Python dependencies
echo "🔍 Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchdiffeq; print('torchdiffeq: OK')" 2>/dev/null || {
    echo "❌ torchdiffeq not found. Installing..."
    pip install torchdiffeq
}

# Start training
echo ""
echo "🚂 Starting training..."
echo "Press Ctrl+C to stop training gracefully"
echo ""

python train_neural_ode.py \
    --dataset_root "$DATASET_ROOT" \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LEARNING_RATE \
    --latent_dim $LATENT_DIM \
    --num_events $NUM_EVENTS \
    --num_time_steps $NUM_TIME_STEPS \
    --save_dir "$EXPERIMENT_DIR" \
    --save_every 10 \
    --save_best \
    --val_split 0.2 \
    --val_every 5 \
    --num_workers 4 \
    --position_weight 1.0 \
    --orientation_weight 0.1 \
    --use_stereo

echo ""
echo "🎉 Training completed!"
echo "📁 Check results in: $EXPERIMENT_DIR"