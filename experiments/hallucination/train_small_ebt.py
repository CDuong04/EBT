"""
Minimal training script for small EBT on SQuAD for hallucination experiments
This trains a tiny model quickly for proof-of-concept
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from data.nlp.squad_dataloader import SQuADDataset
from data.nlp.collator import NLP_HF_Collator
from model.nlp.ebt import EBT_NLP

class EBTWrapper(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = EBT_NLP(self.hparams)

    def training_step(self, batch, batch_idx):
        metrics = self.model.forward_loss_wrapper(batch, "train")
        loss = metrics["loss"]
        self.log_dict({f"train_{k}": v for k, v in metrics.items()}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        metrics = self.model.forward_loss_wrapper(batch, "val")
        self.log_dict({f"val_{k}": v for k, v in metrics.items()}, prog_bar=True)
        return metrics["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_steps,
            eta_min=self.hparams.lr / 10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

    def train_dataloader(self):
        dataset = SQuADDataset(self.hparams, split="train")
        collate_fn = NLP_HF_Collator(self.hparams)
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size_per_device,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        dataset = SQuADDataset(self.hparams, split="validation")
        collate_fn = NLP_HF_Collator(self.hparams)
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size_per_device,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
        )

def main():
    # Disable SLURM environment detection to avoid ntasks issue
    # PyTorch Lightning tries to use SLURM settings which may conflict
    os.environ.pop('SLURM_NTASKS', None)
    os.environ.pop('SLURM_JOB_NAME', None)

    # Basic hyperparameters - very small model for quick training
    hparams = dict(
        # Training
        lr=1e-3,
        weight_decay=0.01,
        batch_size_per_device=8,
        num_workers=4,
        max_steps=5000,  # Quick training for proof of concept

        # Model size - tiny for fast iteration
        embedding_dim=256,
        num_transformer_blocks=4,
        multiheaded_attention_heads=4,
        ffn_dim_multiplier=1,
        weight_initialization_method="xavier",
        weight_initialization_gain=1.0,

        # Data
        dataset_dir="",
        dataset_name="squad",
        context_length=512,  # Increased to 512 to avoid truncating [[Answer]]: tag
        pretokenize_dataset=False,  # SQuAD returns raw strings, not pre-tokenized
        tokenizer="EleutherAI/gpt-neox-20b",
        execution_mode="finetune",  # Use finetune mode (only trains on answer tokens after [[Answer]]:)

        # EBT-specific parameters
        model_name="ebt",
        mcmc_step_size=500.0,
        mcmc_step_size_lr_multiplier=1500.0,
        mcmc_num_steps=2,
        ebt_type="default",  # Use default type (time_embed has issues with variable lengths)
        normalize_initial_condition=True,
        denoising_initial_condition="random_noise",
        mcmc_step_size_learnable=True,
        no_mcmc_detach=False,

        # Additional EBT params (defaults)
        ebt_norm="rms",
        ebt_act_func="silu",
        dyt_alpha_init=0.5,
        mcmc_replay_buffer=False,
        gaussian_random_noise_scaling=1.0,
        normalize_initial_condition_only_first_step=False,
        randomize_mcmc_step_size_scale=1.0,
        randomize_mcmc_num_steps=0,
        randomize_mcmc_num_steps_min=0,
        randomize_mcmc_num_steps_final_landscape=False,
        langevin_dynamics_noise=0.0,
        langevin_dynamics_noise_learnable=False,
        vocab_to_embed_uses_prob_dist=False,
        num_modality_processing_mlp_layers=1,
        truncate_mcmc=False,
        clamp_futures_grad=False,
        clamp_futures_grad_max_change=9.0,
        absolute_clamp=0.0,
        clamp_max_after_warm_up=0.0,
        sharpen_predicted_distribution=0.0,
        reconstruction_coeff=1.0,
        contrastive_loss=False,
        contrastive_loss_coeff=0.0005,
        soften_target_prob_dist=0.0,
        debug_unused_parameters=False,
        discrete_contrastive_loss_true_logit_val=0,
    )

    # Create model
    model = EBTWrapper(hparams)

    # Logger
    logger = WandbLogger(
        name="small_ebt_squad_hallucination_exp",
        project="hallucination_experiments",
        entity=""  # Add your wandb entity if needed
    )

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="experiments/hallucination/checkpoints",
        filename="ebt_squad_{step}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    # Trainer
    trainer = pl.Trainer(
        max_steps=hparams["max_steps"],
        devices=1,  # Single GPU for simplicity
        logger=logger,
        callbacks=[checkpoint_callback],
        val_check_interval=500,  # Validate every 500 steps
        enable_model_summary=True,
        gradient_clip_val=1.0,
    )

    # Train
    print("Starting training...")
    trainer.fit(model)
    print(f"Training complete! Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Last checkpoint: {checkpoint_callback.last_model_path}")

if __name__ == "__main__":
    main()
