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
    parser = argparse.ArgumentParser(description="ADP Multi-Node Training")
    parser.add_argument("--config", type=str, required=True, help="Path to training configuration file")
    parser.add_argument("--save_ckpt_path", type=str, required=True, help="Path to save checkpoints")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    print("=" * 80)
    print("ADP MULTI-NODE TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Config file: {args.config}")
    print(f"Checkpoint path: {args.save_ckpt_path}")
    print(f"Resume from: {args.ckpt_path}")
    print(f"GPUs: {args.gpus}")
    print(f"Nodes: {args.nodes}")
    print(f"Precision: {args.precision}")
    print(f"Max epochs: {args.max_epochs}")
    print("=" * 80)
    
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
        print("⚠️  Advanced features configuration not found")
    print("=" * 80)
    
    print("Initializing data module...")
    data_module = DataModule(config.Dataset)
    
    print("Initializing model...")
    model = AutonomousMotionPredictor(config.Model)
    
    strategy = DDPStrategy(
        find_unused_parameters=False,
        sync_batchnorm=True,
        static_graph=False
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_ckpt_path,
        filename="advanced_features_step_{step:06d}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=1000,
        save_on_train_epoch_end=False
    )
    
    
    logger = TensorBoardLogger(
        save_dir="logs",
        name="adp_multi_node",
        version=None,
        log_graph=False,
        default_hp_metric=True
    )
    
    
    print("Initializing trainer...")
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.gpus,
        num_nodes=args.nodes,
        strategy=strategy,
        precision=args.precision,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        log_every_n_steps=100,
        val_check_interval=1000,
        limit_val_batches=0.1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=[checkpoint_callback],
        logger=logger,
        sync_batchnorm=True,
        find_unused_parameters=False
    )
    
    
    print("TRAINER CONFIGURATION:")
    print(f"- Max epochs: {trainer.max_epochs}")
    print(f"- Accelerator: {trainer.accelerator}")
    print(f"- Devices: {trainer.devices}")
    print(f"- Num nodes: {trainer.num_nodes}")
    print(f"- Strategy: {trainer.strategy}")
    print(f"- Precision: {trainer.precision}")
    print(f"- Gradient clip: {trainer.gradient_clip_val}")
    print(f"- Log every n steps: {trainer.log_every_n_steps}")
    print(f"- Val check interval: {trainer.val_check_interval}")
    print(f"- Limit val batches: {trainer.limit_val_batches}")
    print("=" * 80)
    
    
    print("Starting multi-node training...")
    print("=" * 80)
    
    try:
        if args.ckpt_path:
            print(f"Resuming from checkpoint: {args.ckpt_path}")
            trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
        else:
            print("Starting fresh training")
            trainer.fit(model, data_module)
        
        print("=" * 80)
        print("Multi-node training completed successfully!")
        print(f"Checkpoints saved in: {args.save_ckpt_path}")
        
    except Exception as e:
        print("=" * 80)
        print(f"Multi-node training failed: {e}")
        print("Check error logs for details")
        sys.exit(1)
    
    
    print("=" * 80)
    print("FINAL TRAINING STATISTICS:")
    print(f"- Total epochs: {trainer.current_epoch}")
    print(f"- Total steps: {trainer.global_step}")
    print(f"- Best validation loss: {checkpoint_callback.best_model_score}")
    print(f"- Checkpoints saved: {len(os.listdir(args.save_ckpt_path)) if os.path.exists(args.save_ckpt_path) else 0}")
    print("=" * 80)

if __name__ == "__main__":
    main()
