#!/bin/bash -l
#SBATCH --job-name=ADP_multi_node_training
#SBATCH --time=24:00:00                    # 24 hours max
#SBATCH --nodes=1                          # 1 node (partition limit)
#SBATCH --ntasks-per-node=4                # 4 tasks per node (one per GPU)
#SBATCH --cpus-per-task=32                 # 32 CPUs per task (128/4)
#SBATCH --mem=187G                          # 187GB per task (750/4)
#SBATCH --gres=gpu:h100:4                  # 4 H100s per node
#SBATCH --partition=msigpu                 # H100 partition
#SBATCH --mail-type=BEGIN,END,FAIL         # Email notifications
#SBATCH --mail-user=batta083@umn.edu       # Your UMN email
#SBATCH --output=ADP_multi_%j.out          # Output log with job ID
#SBATCH --error=ADP_multi_%j.err           # Error log with job ID


echo "Multi-node training job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Working directory: $(pwd)"


module load conda
module load cuda/12.0
module load gcc/8.2.0


echo "Changing to Autonomous Driving Predictor directory..."
cd /scratch.global/batta083/autonomous_driving_predictor || exit 1
echo "Current directory: $(pwd)"

echo "Activating waymo_env..."
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate waymo_env

export PATH="/users/6/batta083/.conda/envs/waymo_env/bin:$PATH"
export PYTHONPATH="/users/6/batta083/.conda/envs/waymo_env/lib/python3.9/site-packages:$PYTHONPATH"

export LD_LIBRARY_PATH="/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH"

echo "Active environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "PyTorch Lightning check:"
python -c "import pytorch_lightning; print(f'PyTorch Lightning {pytorch_lightning.__version__} found')" || echo "PyTorch Lightning not found"

echo "PyTorch CUDA check:"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

if [ ! -f "execution_scripts/multi_node_training.py" ]; then
    echo "ERROR: multi_node_training.py not found in execution_scripts/"
    exit 1
fi

if [ ! -d "waymo_open_dataset/train" ] || [ -z "$(find waymo_open_dataset/train -name "*.pkl" -type f | head -1)" ]; then
    echo "ERROR: Preprocessed training data not found!"
    echo "   Directory: waymo_open_dataset/train"
    exit 1
fi

LATEST_CHECKPOINT_DIR=""
for dir in $(ls -td checkpoints/*/ 2>/dev/null); do
    if [ -n "$(ls "$dir"*.ckpt 2>/dev/null)" ]; then
        LATEST_CHECKPOINT_DIR="$dir"
        break
    fi
done

LATEST_CHECKPOINT=""
if [ -n "$LATEST_CHECKPOINT_DIR" ]; then
    LATEST_CHECKPOINT=$(ls -t "$LATEST_CHECKPOINT_DIR"*.ckpt 2>/dev/null | head -1)
    if [ -f "$LATEST_CHECKPOINT" ]; then
        echo "Found existing checkpoint: $LATEST_CHECKPOINT"
        
        if [[ "$LATEST_CHECKPOINT" == *"step_"* ]]; then
            echo "✓ Compatible step-based checkpoint found"
            echo "Resuming from checkpoint: $LATEST_CHECKPOINT"
            RESUME_FROM_CHECKPOINT=true
        else
            echo " Found epoch-based checkpoint (incompatible with multi-node)"
            echo "   Checkpoint: $LATEST_CHECKPOINT"
            echo "   Starting fresh training to avoid compatibility issues"
            RESUME_FROM_CHECKPOINT=false
        fi
    else
        echo "No valid checkpoint files found in $LATEST_CHECKPOINT_DIR"
        RESUME_FROM_CHECKPOINT=false
    fi
else
    echo "No checkpoint directories found. Starting fresh training."
    RESUME_FROM_CHECKPOINT=false
fi

echo "GPU Information:"
nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355
export WORLD_SIZE=4  
export NCCL_DEBUG=INFO  
export NCCL_IB_DISABLE=1 
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export TORCH_FLOAT32_MATMUL_PRECISION=medium
export TORCH_GEOMETRIC_WARN_ONCE=1
export CUDA_LAUNCH_BLOCKING=0

echo "Starting ADP single-node multi-GPU training with preprocessed data..."
echo "Configuration:"
echo "- Nodes: 1 node × 4 H100s = 4 total GPUs"
echo "- Tasks: 4 tasks × 1 H100 each = 4 total GPUs"
echo "- Batch size: 16 per GPU (effective: 64 with grad accumulation x1)"
echo "- CPUs: 32 cores per task, Memory: 187GB per task"
echo "- Dataset: Preprocessed Waymo pkl data ($(find waymo_open_dataset/train -name "*.pkl" -type f | wc -l) training files)"
echo "- Data workers: 4 parallel loaders per node"
echo "- Precision: Mixed 16-bit for speed and H100 optimization"
echo "- Validation: DISABLED (training-only mode)"
echo "- Checkpointing: Every 1000 steps"
echo "- Advanced Features: Lane Tokens + Relational Attention ENABLED"
if [ "$RESUME_FROM_CHECKPOINT" = true ]; then
    echo "- RESUME MODE: Resuming from checkpoint: $LATEST_CHECKPOINT"
else
    echo "- FRESH START: No existing checkpoints found"
fi

if [ "$RESUME_FROM_CHECKPOINT" = true ]; then
    CHECKPOINT_DIR="$LATEST_CHECKPOINT_DIR"
    echo "Using existing checkpoint directory: $CHECKPOINT_DIR"
else
    CHECKPOINT_DIR="/scratch.global/batta083/autonomous_driving_predictor/checkpoints/multi_node_$(date +%Y%m%d_%H%M%S)"
    mkdir -p $CHECKPOINT_DIR
    echo "Created new checkpoint directory: $CHECKPOINT_DIR"
fi

echo "Final Python path: $(which python)"

if [ "$RESUME_FROM_CHECKPOINT" = true ]; then
    echo "Resuming training from checkpoint: $LATEST_CHECKPOINT"
    python execution_scripts/multi_node_training.py \
        --config configs/training_config_multi_node.yaml \
        --save_ckpt_path="$CHECKPOINT_DIR" \
        --ckpt_path="$LATEST_CHECKPOINT"
else
    echo "Starting fresh training"
    python execution_scripts/multi_node_training.py \
        --config configs/training_config_multi_node.yaml \
        --save_ckpt_path="$CHECKPOINT_DIR"
fi

if [ $? -eq 0 ]; then
    echo "Multi-node training completed successfully at: $(date)"
    echo "Checkpoints saved in: $CHECKPOINT_DIR"
else
    echo "Multi-node training failed or was interrupted at: $(date)"
    echo "Check error logs for details"
fi

echo "Job completed at: $(date)"
echo "Resource usage:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,State
