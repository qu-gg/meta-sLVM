"""
@file meta_main.py

Holds the general training script for the meta-learning models, defining a dataset and model to train on
"""
import os
import json
import hydra
import torch
import numpy as np
import pytorch_lightning
import matplotlib.pyplot as plt
import pytorch_lightning.loggers as pl_loggers

from omegaconf import DictConfig
from utils.dataloader import SSMDataModule
from utils.plotting import get_embedding_tsne
from utils.utils import find_best_step, get_model, flatten_cfg
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(cfg.seed, workers=True)

    # Limit number of CPU workers
    torch.set_num_threads(8)

    # Flatten the Hydra config
    cfg.exptype = cfg.exptype
    cfg = flatten_cfg(cfg)

    # Initialize model
    model = get_model(cfg.model)(cfg)

    # Build consistent colors for active task IDs
    colors = []
    for i in range(16):
        colors.append(next(plt.gca()._get_lines.prop_cycler)['color'])
    cfg.colors = colors

    # Initialize data module
    datamodule = SSMDataModule(cfg, cfg.task_ids)
    print(f"=> Dataset 'train' shape: {datamodule.train_dataloader().dataset.images.shape}")
    print(f"=> Dataset 'val' shape: {datamodule.val_dataloader().dataset.images.shape}")

    # Set up the logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"experiments/{cfg.exptype}/", name=f"{cfg.model}")
    
    # Callbacks for checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(monitor='val_dst',
                                          filename='step{step:02d}-val_dst{val_dst:.4f}',
                                          auto_insert_metric_name=False, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize trainer
    trainer = pytorch_lightning.Trainer(
        callbacks=[
            lr_monitor,
            checkpoint_callback
        ],
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=1,
        max_steps=cfg.num_steps * cfg.batch_size,
        gradient_clip_val=cfg.gradient_clip,
        val_check_interval=cfg.val_log_interval,
        num_sanity_val_steps=0,
        logger=tb_logger
    )

    # Train the model
    trainer.fit(model, datamodule)

    """ Testing Block """
    # Build the model path to test on, the passed in config path or the current logging directory
    cfg.model_path = f"{tb_logger.log_dir}/"
    print(f"=> Built model path: {cfg.model_path}.")

    # If a checkpoint is not given, get the best one from training
    ckpt_path = f"{cfg.model_path}/checkpoints/{find_best_step(f'{cfg.model_path}/checkpoints/')[0]}"
    print(f"=> Loading in checkpoint path: {ckpt_path}.")

    # Test on the training set
    cfg.split = "train"
    trainer.test(model, datamodule.evaluate_train_dataloader(), ckpt_path=ckpt_path)

    # Test the model, regardless of training, over all given data splits
    for split in cfg.testing_splits:
        cfg.split = split
        datamodule = SSMDataModule(cfg, task_ids=[split])
        trainer.test(model, datamodule, ckpt_path=ckpt_path)

    # Aggregate all metrics into list
    metrics = dict()
    query_idx = []
    for query in os.listdir(tb_logger.log_dir):
        if "test" not in query or ".json" in query or ".txt" in query or "train" in query:
            continue

        query_idx.append(int(query.split('_')[-1]))
        metric_json = json.load(open(f"{cfg.model_path}/{query}/{query}_metrics.json", 'r'))

        for metric in cfg.metrics:
            if f"{metric}_mean" in metrics.keys():
                metrics[f"{metric}_mean"].append(metric_json[f"{metric}_mean"])
                metrics[f"{metric}_std"].append(metric_json[f"{metric}_std"])
            else:
                metrics[f"{metric}_mean"] = [metric_json[f"{metric}_mean"]]
                metrics[f"{metric}_std"] = [metric_json[f"{metric}_std"]]

    # Sort indices by folder idx rather than listdir order
    sorted_indices = np.argsort(query_idx)
    for mkey in metrics.keys():
        metrics[mkey] = np.array(metrics[mkey])[sorted_indices]

    with open(f"{cfg.model_path}/test_all_excel.txt", 'a') as f:
        f.write("\n")
        for mkey in cfg.metrics:
            f.write(f"{np.mean(metrics[f'{mkey}_mean']):0.4f}({np.mean(metrics[f'{mkey}_std']):0.4f}) & ")

    # For meta-models, get the tSNE of meta-embeddings
    all_train_sets = dict()
    all_test_sets = dict()

    # Stack all sets of embeddings and assignments
    all_train_sets['embeddings'] = np.load(f"{tb_logger.log_dir}/test_train/test_train_embeddings.npy")
    all_train_sets['labels'] = np.load(f"{tb_logger.log_dir}/test_train/test_train_labels.npy")
    all_test_sets["embeddings"] = np.vstack([np.load(f"{tb_logger.log_dir}/test_{pid}/test_{pid}_embeddings.npy") for pid in cfg.task_ids])
    all_test_sets["labels"] = np.vstack([np.load(f"{tb_logger.log_dir}/test_{pid}/test_{pid}_labels.npy") for pid in cfg.task_ids])
    get_embedding_tsne(f"{tb_logger.log_dir}/embedding_tsne.png", cfg, all_train_sets, all_test_sets)

    # Remove preds and image npy files
    os.system(f"find experiments/ -path 'experiments/{cfg.exptype}*' -name '*_images.npy' -delete")
    os.system(f"find experiments/ -path 'experiments/{cfg.exptype}*' -name '*_preds.npy' -delete")


if __name__ == '__main__':
    main()
