"""
@file CommonMetaDynamics.py

A common class that each meta latent dynamics function inherits.
Holds the training + validation step logic and the VAE components for reconstructions.
Has a testing step for holdout steps that handles all metric calculations and visualizations.
"""
import os
import json
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning

from utils import metrics
from utils.plotting import show_images
from models.CommonVAE import LatentStateEncoder, EmissionDecoder
from utils.utils import CosineAnnealingWarmRestartsWithDecayAndLinearWarmup, determine_annealing_factor


class LatentMetaDynamicsModel(pytorch_lightning.LightningModule):
    def __init__(self, args):
        """ Generic training and testing boilerplate for the dynamics models """
        super().__init__()
        self.save_hyperparameters(args)
        self.cfg = args

        # Encoder + Decoder
        self.encoder = LatentStateEncoder(args)
        self.decoder = EmissionDecoder(args)

        # Recurrent dynamics function
        self.dynamics_func = None
        self.dynamics_out = None

        # Losses
        self.reconstruction_loss = nn.MSELoss(reduction='none')

        # General trackers
        self.n_updates = 0

        # Accumulation of outputs over the logging interval
        self.outputs = list()

    def forward(self, x, D):
        """ Placeholder function for the dynamics forward pass """
        raise NotImplementedError("In forward: Latent Dynamics function not specified.")

    def model_specific_loss(self, x, domain, preds):
        """ Placeholder function for any additional loss terms a dynamics function may have """
        return 0.0

    def model_specific_plotting(self, version_path, outputs):
        """ Placeholder function for any additional plots a dynamics function may have """
        return None

    def configure_optimizers(self):
        """ Configure the AdamW optimizer and CosineAnnealing scheduler """
        optim = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate)
        scheduler = CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(
            optim,
            T_0=self.cfg.scheduler['restart_interval'], T_mult=1,
            eta_min=self.cfg.learning_rate * 1e-2,
            warmup_steps=self.cfg.scheduler['warmup_steps'],
            decay=self.cfg.scheduler['decay']
        )

        # Explicit dictionary to state how often to ping the scheduler
        scheduler = {
            'scheduler': scheduler,
            'frequency': 1,
            'interval': 'step'
        }
        return [optim], [scheduler]

    def on_train_start(self):
        """ Boilerplate experiment logging setup pre-training """
        # Get total number of parameters for the model and save
        self.log("total_num_parameters", float(sum(p.numel() for p in self.parameters() if p.requires_grad)), prog_bar=False)

        # Make image dir in lightning experiment folder if it doesn't exist
        if not os.path.exists(f"{self.logger.log_dir}/images/"):
            os.mkdir(f"{self.logger.log_dir}/images/")

    def get_step_outputs(self, batch):
        """ Handles processing a batch and getting model predictions """
        # Get batch
        images, domains, states, _, labels = batch

        # Get predictions
        preds, zt = self(images, domains)
        return images, domains, states, labels, preds, zt

    def get_step_losses(self, images, domains, preds):
        """ For a given batch, compute all potential loss terms from components """
        # Reconstruction loss for the sequence and z0
        likelihoods = self.reconstruction_loss(preds, images)
        likelihood = likelihoods.reshape([likelihoods.shape[0] * likelihoods.shape[1], -1]).sum([-1]).mean()

        # Initial encoder loss, KL[q(z_K|x_0:K) || p(z_K)]
        klz = self.cfg.betas.z0 * self.encoder.kl_z_term()

        # Get the loss terms from the specific latent dynamics loss
        model_specific_loss = self.model_specific_loss(images, domains, preds)

        # Return all loss terms
        return likelihood, klz, model_specific_loss

    def get_epoch_metrics(self, outputs, length=20):
        """
        Takes the dictionary of saved batch metrics, stacks them, and gets outputs to log in the Tensorboard.
        :param outputs: list of dictionaries with outputs from each back
        :return: dictionary of metrics aggregated over the epoch
        """
        # Convert outputs to Tensors and then Numpy arrays
        images = torch.vstack([out["images"] for out in outputs]).cpu().numpy()
        preds = torch.vstack([out["preds"] for out in outputs]).cpu().numpy()
        
        states = torch.vstack([out["states"] for out in outputs]).cpu().numpy()
        zt = torch.vstack([out["zt"] for out in outputs]).cpu().numpy()

        # Iterate through each metric function and add to a dictionary
        out_metrics = {}
        for met in self.cfg.metrics:
            metric_function = getattr(metrics, met)
            out_metrics[met] = metric_function(images, preds, cfg=self.cfg, length=length)[1]

        # TODO - testing MCC
        out_metrics["mcc_score"] = metrics.mean_corr_coef(zt, states)

        # Return a dictionary of metrics
        return out_metrics

    def training_step(self, batch, batch_idx):
        """ Training step, getting loss and returning it to optimizer """
        # Reshuffle context/query sets
        self.trainer.train_dataloader.dataset.datasets.split()

        # Get model outputs from batch
        images, domains, states, labels, preds, zt = self.get_step_outputs(batch)

        # Get model loss terms for the step
        likelihood, klz, dynamics_loss = self.get_step_losses(images, domains, preds)

        # Determine KL annealing factor for the current step
        kl_factor = determine_annealing_factor(self.n_updates, anneal_update=1000)

        # Build the full loss
        loss = likelihood + kl_factor * ((self.cfg.betas.z0 * klz) + dynamics_loss)

        # Log ELBO loss terms
        self.log_dict({
            "likelihood": likelihood,
            "kl_z": self.cfg.betas.z0 * klz,
            "dynamics_loss": dynamics_loss,
            "kl_factor": kl_factor
        })
        
        # Log metrics every N batches
        self.outputs.append({
            "loss": loss, "labels": labels.detach(), 
            "preds": preds.detach(), "images": images.detach(),
            "zt": zt.detach(), "states": states
        })

        # Return the loss for updating and track the iteration number
        self.n_updates += 1
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """ Given the iterative training, check on every batch's end whether it is evaluation time or not """
        # Show side-by-side reconstructions
        if batch_idx % self.cfg.log_interval == 0 and batch_idx != 0:
            show_images(self.outputs[0]["images"], self.outputs[0]["preds"], f'{self.logger.log_dir}/images/recon{batch_idx}train.png', num_out=5)

            # Get per-dynamics plots
            self.model_specific_plotting(self.logger.log_dir, self.outputs)
            
            # Get metrics
            metrics = self.get_epoch_metrics(self.outputs, length=self.cfg.generation_length)
            for metric in metrics.keys():
                self.log(f"train_{metric}", metrics[metric], prog_bar=True)

            self.outputs = list()

    def validation_step(self, batch, batch_idx):
        """
        PyTorch-Lightning validation step. Similar to the training step but on the given val set under torch.no_grad()
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        # Get model outputs from batch
        images, domains, states, labels, preds, zt = self.get_step_outputs(batch)

        # Get model loss terms for the step
        likelihood, klz, dynamics_loss = self.get_step_losses(images, domains, preds)

        # Log validation likelihood and metrics
        self.log("val_likelihood", likelihood, prog_bar=True)

        # Return outputs as dict
        out = {"loss": likelihood}
        out["preds"] = preds
        out["images"] = images
        return out

    def validation_epoch_end(self, outputs):
        """
        Every N epochs, get a validation reconstruction sample
        :param outputs: list of outputs from the validation steps at batch 0
        """
        # Log epoch metrics on saved batches
        metrics = self.get_epoch_metrics(self.outputs, length=self.cfg.generation_length)
        for metric in metrics.keys():
            self.log(f"val_{metric}", metrics[metric], prog_bar=True)

        # Get image reconstructions
        if self.n_updates % self.cfg.val_log_interval == 0 and self.n_updates != 0:
            show_images(outputs[0]["images"], outputs[0]["preds"], f'{self.logger.log_dir}/images/recon{self.n_updates}val.png', num_out=5)

    def test_step(self, batch, batch_idx):
        """
        PyTorch-Lightning testing step.
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        # Get model outputs from batch
        images, domains, states, labels, preds, zt = self.get_step_outputs(batch)

        # Build output dictionary
        out = {"states": states.detach().cpu(), "embeddings": self.dynamics_func.embeddings.detach().cpu(),
               "preds": preds.detach().cpu(), "images": images.detach().cpu(), "labels": labels.detach().cpu(),
               "zt": states.detach().cpu()}
        return out

    def test_epoch_end(self, batch_outputs):
        """ For testing end, save the predictions, gt, and MSE to NPY files in the respective experiment folder """
        # Set up output path and create dir
        output_path = f"{self.logger.log_dir}/test_{self.cfg.split}"
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        # Stack all output types and convert to numpy
        outputs = dict()
        for key in batch_outputs[0].keys():
            outputs[key] = torch.vstack([output[key] for output in batch_outputs]).numpy()

        # Save to files
        for key in outputs.keys():
            np.save(f"{output_path}/test_{self.cfg.split}_{key}.npy", outputs[key])

        # Iterate through each metric function and add to a dictionary
        out_metrics = {}
        for met in self.cfg.test_metrics:
            metric_function = getattr(metrics, met)
            _, metric_mean, metric_std = metric_function(outputs["images"], outputs["preds"], cfg=self.cfg, length=self.cfg.generation_length)
            out_metrics[f"{met}_mean"], out_metrics[f"{met}_std"] = float(metric_mean), float(metric_std)
            print(f"=> {met}: {metric_mean:4.5f}+-{metric_std:4.5f}")
            
        # Add MCC to test metrics
        out_metrics["mcc_score"] = metrics.mean_corr_coef(outputs["zt"], outputs["states"])

        # Save some examples
        show_images(outputs["images"][:10], outputs["preds"][:10], f"{output_path}/test_{self.cfg.split}_examples.png", num_out=5)

        # Save metrics to JSON in checkpoint folder
        with open(f"{output_path}/test_{self.cfg.split}_metrics.json", 'w') as f:
            json.dump(out_metrics, f)

        # Save metrics to an easy excel conversion style
        with open(f"{output_path}/test_{self.cfg.dataset}_excel.txt", 'w') as f:
            for metric in self.cfg.metrics:
                f.write(f"{out_metrics[f'{metric}_mean']:0.3f}({out_metrics[f'{metric}_std']:0.3f}),")
