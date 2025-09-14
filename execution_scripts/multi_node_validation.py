#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from autonomous_driving_predictor.model_components.autonomous_motion_predictor import AutonomousMotionPredictor
from autonomous_driving_predictor.data_processing.data_module import DataModule
from autonomous_driving_predictor.core_algorithms.config import load_config

def main():
    parser = argparse.ArgumentParser(description="ADP Multi-Node Validation")
    parser.add_argument("--config", type=str, required=True, help="Path to validation configuration file")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint to validate")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes (default: 1)")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Validation precision")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override number of workers")
    parser.add_argument("--output_dir", type=str, default="validation_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.batch_size:
        config.Dataset.batch_size = args.batch_size
    if args.num_workers:
        config.Dataset.num_workers = args.num_workers
    
    print("=" * 80)
    print("ADP MULTI-NODE VALIDATION CONFIGURATION")
    print("=" * 80)
    print(f"Config file: {args.config}")
    print(f"Checkpoint path: {args.ckpt_path}")
    print(f"GPUs: {args.gpus}")
    print(f"Nodes: {args.nodes}")
    print(f"Precision: {args.precision}")
    print(f"Batch size: {config.Dataset.batch_size}")
    print(f"Workers: {config.Dataset.num_workers}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    if not os.path.exists(args.ckpt_path):
        print(f"ERROR: Checkpoint file not found: {args.ckpt_path}")
        sys.exit(1)
    
    print("ADVANCED FEATURES STATUS:")
    if hasattr(config.Model, 'advanced_features'):
        af = config.Model.advanced_features
        print(f"✓ Lane Tokens: {af.use_lane_tokens}")
        if af.use_lane_tokens:
            print(f"  - Count: {af.lane_tokens.count}")
            print(f"  - Bucket size: {af.lane_tokens.bucket_size}")
            print(f"  - Embedding dim: {af.lane_tokens.embedding_dim}")
        print(f"✓ Relational Attention: {af.use_relational_attention}")
        if af.use_relational_attention:
            print(f"  - Heads: {af.relational_attention.num_heads}")
            print(f"  - Head dim: {af.relational_attention.head_dim}")
            print(f"  - Rel feature dim: {af.relational_attention.rel_feature_dim}")
    else:
        print(" Advanced features configuration not found")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Initializing data module...")
    data_module = DataModule(config.Dataset)
    
    print("Initializing model...")
    model = AutonomousMotionPredictor(config.Model)
    
    if args.gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            sync_batchnorm=True,
            static_graph=False
        )
    else:
        strategy = "auto"
    
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="adp_validation",
        version=None,
        log_graph=False,
        default_hp_metric=True
    )
    
    print("Initializing trainer...")
    trainer = Trainer(
        accelerator="gpu",
        devices=args.gpus,
        num_nodes=args.nodes,
        strategy=strategy,
        precision=args.precision,
        enable_checkpointing=False,  
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=logger,
        sync_batchnorm=True if args.gpus > 1 else False,
        find_unused_parameters=False if args.gpus > 1 else None
    )
    
    print("TRAINER CONFIGURATION:")
    print(f"- Accelerator: {trainer.accelerator}")
    print(f"- Devices: {trainer.devices}")
    print(f"- Num nodes: {trainer.num_nodes}")
    print(f"- Strategy: {trainer.strategy}")
    print(f"- Precision: {trainer.precision}")
    print(f"- Checkpointing: {trainer.enable_checkpointing}")
    print(f"- Progress bar: {trainer.enable_progress_bar}")
    print(f"- Model summary: {trainer.enable_model_summary}")
    print("=" * 80)
    
    print("Starting validation...")
    print("=" * 80)
    
    try:
        print(f"Loading checkpoint: {args.ckpt_path}")
        trainer.validate(model, data_module, ckpt_path=args.ckpt_path)
        
        print("=" * 80)
        print("Validation completed successfully!")
        print(f"Results saved in: {args.output_dir}")
        
        if hasattr(trainer, 'callback_metrics'):
            print("VALIDATION METRICS:")
            for key, value in trainer.callback_metrics.items():
                print(f"- {key}: {value}")
        
    except Exception as e:
        print("=" * 80)
        print(f"Validation failed: {e}")
        print("Check error logs for details")
        sys.exit(1)
    
    print("=" * 80)
    print("FINAL VALIDATION STATISTICS:")
    print(f"- Checkpoint: {args.ckpt_path}")
    print(f"- GPUs used: {args.gpus}")
    print(f"- Nodes used: {args.nodes}")
    print(f"- Batch size: {config.Dataset.batch_size}")
    print(f"- Workers: {config.Dataset.num_workers}")
    print(f"- Results directory: {args.output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
