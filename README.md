# Autonomous Driving Predictor (ADP)

**Advanced Motion Prediction Framework with Lane Tokens & Relational Attention**

ADP is a state-of-the-art motion prediction system that generates realistic vehicle trajectories through advanced transformer-based architecture with two key innovations:

## Key Innovations

### 1. Lane Tokens
- **What**: Discrete tokens encoding closest lane ID for each agent
- **Why**: Forces model to stay grounded in road geometry
- **Effect**: Eliminates unrealistic off-road trajectories, ensures map-consistent motion

### 2. Relational Attention
- **What**: Attention bias based on relative position, velocity, and distance between agents
- **Why**: Enables realistic social interactions between vehicles
- **Effect**: Captures behaviors like yielding, overtaking, collision avoidance

## Performance Improvements

| **Baseline** | **With Advanced Features** |
|--------------|---------------------------|
| Agents sometimes drift off-road | Map-consistent trajectories |
| Weak interaction modeling | Realistic social behaviors |
| Generic attention patterns | Context-aware attention |

## Technical Specs

- **Architecture**: Transformer-based with advanced attention mechanisms
- **Data**: Waymo Open Motion Dataset (TF to Pickle conversion included)
- **Training**: Multi-GPU distributed training (configured for H100s)
- **Advanced Features**: Lane tokens + Relational attention enabled by default

## Installation & Setup

### Prerequisites
```bash
# Install required packages
pip install torch torch-geometric pytorch-lightning
pip install numpy pandas tqdm easydict pyyaml
```

### Data Preparation
```bash
# Convert Waymo TF files to pickle format
python execution_scripts/data_preparation.py \
    --input_dir /path/to/waymo/tf/files \
    --output_dir waymo_open_dataset/train
```

**Important**: The dataset must be preprocessed and organized in the following structure:
```
waymo_open_dataset/
├── train/          # Training data (.pkl files)
├── validation/     # Validation data (.pkl files)  
└── test/          # Test data (.pkl files)
```

## Configuration for Your System

**IMPORTANT**: This system is designed for HPC environments. You must reconfigure for your local PC specs.

### 1. Update Training Configuration

Edit `configs/training_config_multi_node.yaml`:

```yaml
# Adjust these parameters for your system
Trainer:
  accelerator: "gpu"  # or "cpu" if no GPU
  devices: 1          # Number of GPUs available
  strategy: "auto"    # or "ddp" for multi-GPU
  
Dataset:
  batch_size: 4       # Reduce for limited GPU memory
  num_workers: 2       # Adjust based on CPU cores
  
# Reduce model size if needed
Model:
  hidden_dim: 128     # Reduce from 256
  num_layers: 3       # Reduce from 6
```

### 2. Update Validation Configuration

Edit `configs/validation_config_multi_node.yaml`:

```yaml
Trainer:
  accelerator: "gpu"  # or "cpu"
  devices: 1          # Number of GPUs
  
Dataset:
  batch_size: 8       # Adjust for your GPU memory
  num_workers: 2      # Adjust for CPU cores
```

### 3. System-Specific Adjustments

**For CPU-only systems:**
```yaml
Trainer:
  accelerator: "cpu"
  devices: 1
  precision: "32"     # Use full precision
```

**For single GPU:**
```yaml
Trainer:
  accelerator: "gpu"
  devices: 1
  strategy: "auto"
```

**For limited memory:**
```yaml
Dataset:
  batch_size: 1       # Start with 1, increase gradually
  num_workers: 0      # Disable multiprocessing if needed
```

## Running the System

### Local Training
```bash
# Activate your environment
conda activate your_env

# Run training
python execution_scripts/multi_node_training.py \
    --config configs/training_config_multi_node.yaml
```

### Local Validation
```bash
# Run validation
python execution_scripts/multi_node_validation.py \
    --config configs/validation_config_multi_node.yaml \
    --checkpoint path/to/checkpoint.ckpt
```

### Data Conversion
```bash
# Convert Waymo data
python execution_scripts/data_preparation.py \
    --input_dir /path/to/waymo/training \
    --output_dir waymo_open_dataset/train
```

## Directory Structure

```
autonomous_driving_predictor/
├── autonomous_driving_predictor/     # Core package
├── waymo_open_dataset/               # Pre-split data (train/val/test)
├── configs/                         # Training configurations
├── execution_scripts/               # Training/validation scripts
├── train_adp_multi_node.slurm       # Multi-node training script
└── validate_adp_multi_node.slurm    # Multi-node validation script
```

## Advanced Features Details

### Lane Token System
- 100 discrete lane tokens (LANE_0 to LANE_99)
- Bucketed lane IDs for efficient encoding
- Integrated into agent state representation

### Relational Attention Mechanism
- 5-dimensional relational features (pos_x, pos_y, vel_x, vel_y, distance)
- Learnable bias projection to attention heads
- Dynamic attention based on agent proximity and motion

## Troubleshooting

### Common Issues

**Out of Memory:**
- Reduce `batch_size` in config
- Reduce `hidden_dim` and `num_layers`
- Use `precision: "16-mixed"` for GPU training

**Slow Training:**
- Increase `num_workers` (but not more than CPU cores)
- Use GPU if available
- Reduce `num_workers` to 0 if multiprocessing causes issues

**Import Errors:**
- Ensure all dependencies are installed
- Check Python path includes the package directory

---

**The combination of Lane Tokens + Relational Attention directly improves realism metrics, making ADP competitive on motion prediction benchmarks.**
