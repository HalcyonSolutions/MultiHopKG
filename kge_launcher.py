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
    print(f"Lambda LP: {config.lambda_lp}")
    print(f"Lambda RP: {config.lambda_rp}")
    print(f"Lambda DP: {config.lambda_dp}")
    print(f"Lambda NBE: {config.lambda_nbe}")
    print(f"Lambda NBR: {config.lambda_nbr}")
    print(f"Max Steps: {config.max_steps}")
    print(f"Task: {config.task}")
    print(f"Test Batch Size: {config.test_batch_size}")
    print(f"WandB Project: {config.wandb_project}")
    print(f"Track: {config.track}")
    if 'autoencoder_flag' in config:
        print(f"Autoencoder Flag: {config.autoencoder_flag}")
        print(f"Autoencoder Hidden Dim: {config.autoencoder_hidden_dim}")
        print(f"Autoencoder Lambda: {config.autoencoder_lambda}")
    if 'clean_up' in config:
        print(f"Clean Up: {config.clean_up}")
    print(f"Saving Metric: {config.saving_metric}")
    print(f"Saving Threshold: {config.saving_threshold}")
    print(f"Additional Params: {config.additional_params}")
    print("==========================================")

    if config.mode == "train":
        local_time = time.localtime()
        timestamp = time.strftime("%Y%m%d_%H%M%S", local_time)

        task = str(config.task).lower()
        
        #
        if task == "basic":
            config.lambda_dp = 0.0
            config.lambda_nbe = 0.0
            config.lambda_nbr = 0.0
        elif task == "wild":
            config.lambda_lp = 0.0
            config.lambda_rp = 0.0
        elif task == "relation_prediction":
            config.lambda_lp = 0.0
            config.lambda_dp = 0.0
            config.lambda_nbe = 0.0
            config.lambda_nbr = 0.0
        elif task == "domain_prediction":
            config.lambda_lp = 0.0
            config.lambda_rp = 0.0
            config.lambda_nbe = 0.0
            config.lambda_nbr = 0.0
        elif task == "entity_neighborhood_prediction":
            config.lambda_lp = 0.0
            config.lambda_rp = 0.0
            config.lambda_dp = 0.0
            config.lambda_nbr = 0.0
        elif task == "relation_neighborhood_prediction":
            config.lambda_lp = 0.0
            config.lambda_rp = 0.0
            config.lambda_dp = 0.0
            config.lambda_nbe = 0.0

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
            "--task", str(config.task),
            "--test_batch_size", str(config.test_batch_size),
            "--valid_steps", str(config.valid_steps),
            "--saving_metric", config.saving_metric,
            "--saving_threshold", str(config.saving_threshold),
            "--random_seed", str(config.seed),
            "--timestamp", timestamp,
            "--lambda_lp", str(config.lambda_lp),
            "--lambda_rp", str(config.lambda_rp),
            "--lambda_dp", str(config.lambda_dp),
            "--lambda_nbe", str(config.lambda_nbe),
            "--lambda_nbr", str(config.lambda_nbr),
        ]

        if config.additional_params:
            cmd += config.additional_params.split()

        if str(config.model).lower() == "rotate":
            cmd += ["--double_entity_embedding"]

        if str(config.clean_up).lower() == "true":
            cmd += ["--clean_up", "--clean_up_folder"]

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
