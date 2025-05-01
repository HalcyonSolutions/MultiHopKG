"""
Supports training via Wandb parameter sweep. (Doesn't work with bash for some reason)
"""

import wandb
import subprocess
import os
import time

def main():
    wandb.init()

    config = wandb.config

    # Print for sanity check
    print("=== Sanity Check: Collected Arguments ===")
    print(f"Mode: {config.mode}")
    print(f"Model: {config.model}")
    print(f"Dataset: {config.dataset}")
    print(f"GPU_ID: {config.gpu_id}")
    print(f"Seed: {config.seed}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Negative Sample Size: {config.negative_sample_size}")
    print(f"Hidden Dim: {config.hidden_dim}")
    print(f"Gamma: {config.gamma}")
    print(f"Alpha: {config.alpha}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Max Steps: {config.max_steps}")
    print(f"Test Batch Size: {config.test_batch_size}")
    print(f"WandB Project: {config.wandb_project}")
    print(f"Track: {config.track}")
    if 'autoencoder_flag' in config:
        print(f"Autoencoder Flag: {config.autoencoder_flag}")
        print(f"Autoencoder Hidden Dim: {config.autoencoder_hidden_dim}")
        print(f"Autoencoder Lambda: {config.autoencoder_lambda}")
    print(f"Saving Metric: {config.saving_metric}")
    print(f"Saving Threshold: {config.saving_threshold}")
    print(f"Additional Params: {config.additional_params}")
    print("==========================================")

    if config.mode == "train":
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)

        cmd = [
            "python", "kge_train.py",
            "--do_train",
            "--cuda",
            "--do_valid",
            "--do_test",
            "--data_path", f"data/{config.dataset}",
            "--model", config.model,
            "-n", str(config.negative_sample_size),
            "-b", str(config.batch_size),
            "-d", str(config.hidden_dim),
            "-g", str(config.gamma),
            "-a", str(config.alpha),
            "-adv",
            "-lr", str(config.learning_rate),
            "--max_steps", str(config.max_steps),
            "-save", f"models/{config.model}_{config.dataset}_dim{config.hidden_dim}_{timestamp}",
            "--test_batch_size", str(config.test_batch_size),
            "--valid_steps", str(config.valid_steps),
            "--saving_metric", config.saving_metric,
            "--saving_threshold", str(config.saving_threshold),
            "--random_seed", str(config.seed),
            "--timestamp", timestamp,
        ]

        if config.additional_params:
            cmd += config.additional_params.split()

        if str(config.track).lower() == "true":
            cmd += ["-track", "--wandb_project", config.wandb_project]
        
        if 'autoencoder_flag' in config:
            cmd += ["--autoencoder_flag", "--autoencoder_hidden_dim", str(config.autoencoder_hidden_dim), "--autoencoder_lambda", str(config.autoencoder_lambda)]

    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
