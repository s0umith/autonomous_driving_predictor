#!/bin/bash -l
#SBATCH --job-name=ADP_multi_node_validation
#SBATCH --time=4:00:00                     # 4 hours max (validation is faster)
#SBATCH --nodes=1                          # 1 node
#SBATCH --ntasks-per-node=1                # 1 task per node
#SBATCH --cpus-per-task=32                 # 32 CPUs per task
#SBATCH --mem=64G                           # 64GB memory (less than training)
#SBATCH --gres=gpu:h100:1                   # 1 H100 for validation
#SBATCH --partition=msigpu                 # H100 partition
#SBATCH --mail-type=BEGIN,END,FAIL         # Email notifications
#SBATCH --mail-user=batta083@umn.edu       # Your UMN email
#SBATCH --output=ADP_validation_%j.out     # Output log with job ID
#SBATCH --error=ADP_validation_%j.err      # Error log with job ID



echo "Multi-node validation job started at: $(date)"
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


if [ ! -f "execution_scripts/multi_node_validation.py" ]; then
    echo "ERROR: multi_node_validation.py not found in execution_scripts/"
    exit 1
fi


if [ ! -d "waymo_open_dataset/validation" ] || [ -z "$(find waymo_open_dataset/validation -name "*.pkl" -type f | head -1)" ]; then
    echo "ERROR: Preprocessed validation data not found!"
    echo "   Directory: waymo_open_dataset/validation"
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
        echo "Found checkpoint for validation: $LATEST_CHECKPOINT"
    else
        echo "No valid checkpoint files found in $LATEST_CHECKPOINT_DIR"
        echo "ERROR: No checkpoint found for validation"
        exit 1
    fi
else
    echo "ERROR: No checkpoint directories found for validation"
    exit 1
fi


echo "GPU Information:"
nvidia-smi

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export TORCH_FLOAT32_MATMUL_PRECISION=medium
export TORCH_GEOMETRIC_WARN_ONCE=1
export CUDA_LAUNCH_BLOCKING=0


echo "Starting ADP validation..."
echo "Configuration:"
echo "- GPU: 1 H100"
echo "- CPUs: 32 cores"
echo "- Memory: 64GB"
echo "- Dataset: Preprocessed Waymo pkl data ($(find waymo_open_dataset/validation -name "*.pkl" -type f | wc -l) validation files)"
echo "- Data workers: 8 parallel loaders"
echo "- Precision: Mixed 16-bit for H100 optimization"
echo "- Batch size: 32 (optimized for validation)"
echo "- Advanced Features: Lane Tokens + Relational Attention ENABLED"
echo "- Checkpoint: $LATEST_CHECKPOINT"


VALIDATION_DIR="/scratch.global/batta083/autonomous_driving_predictor/validation_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p $VALIDATION_DIR
echo "Created validation results directory: $VALIDATION_DIR"


echo "Final Python path: $(which python)"

echo "Starting validation with checkpoint: $LATEST_CHECKPOINT"
python execution_scripts/multi_node_validation.py \
    --config configs/validation_config_multi_node.yaml \
    --ckpt_path="$LATEST_CHECKPOINT" \
    --output_dir="$VALIDATION_DIR"


if [ $? -eq 0 ]; then
    echo "Validation completed successfully at: $(date)"
    echo "Results saved in: $VALIDATION_DIR"
else
    echo "Validation failed or was interrupted at: $(date)"
    echo "Check error logs for details"
fi


echo "Job completed at: $(date)"
echo "Resource usage:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,State
