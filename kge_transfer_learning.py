"""
kge_transfer_learning.py

Calls kge_train.py for each dataset in order, passing transfer-learning args as appropriate.
"""

import subprocess
import time
import os

import random

def prepare_command(dataset, model, negative_sample_size, batch_size, hidden_dim, gamma, alpha, learning_rate,
                    max_steps, test_batch_size, valid_steps, saving_metric, saving_threshold, seed, timestamp):
    save_dir = f"models/{model}_{dataset}_dim{hidden_dim}_{timestamp}"
    cmd = [
        "python", "kge_train.py",
        "--do_train",
        "--cuda",
        "--do_valid",
        "--do_test",
        "--data_path", f"data/{dataset}",
        "--model", model,
        "-n", str(negative_sample_size),
        "-b", str(batch_size),
        "-d", str(hidden_dim),
        "-g", str(gamma),
        "-a", str(alpha),
        "-adv",
        "-lr", str(learning_rate),
        "--max_steps", str(max_steps),
        "-save", save_dir,
        "--test_batch_size", str(test_batch_size),
        "--valid_steps", str(valid_steps),
        "--saving_metric", saving_metric,
        "--saving_threshold", str(saving_threshold),
        "--random_seed", str(seed),
        "--timestamp", timestamp,
    ]
    return cmd, save_dir

def main():
    # === USER SETTINGS ===
    datasets_train = [
        "FamilyBodon/subset1",
        "FamilyBodon/subset2",
        "FamilyBodon/subset3",
        "FamilyBodon/subset4",
        "FamilyBodon/subset5",
        "FamilyBodon/subset6",
        "FamilyBodon/subset7"
    ]

    dataset_valid = [
        "FamilyBodon/subset8",
        "FamilyBodon/subset9",
        "FamilyBodon/subset10",
    ]

    dataset_test = [
        "FamilyBodon",
    ]

    # Set these True/False as needed for transfer
    reload_entities = False
    reload_relationship = True

    model = "pRotatE"
    negative_sample_size = 128
    batch_size = 32
    hidden_dim = 200
    gamma = 12.0
    alpha = 1.0
    learning_rate = 0.0001
    max_dataset_loop = 10  # Number of times to loop through the dataset
    max_steps = 50000
    test_batch_size = 4
    valid_steps = 5000
    saving_metric = "MRR"
    saving_threshold = 0.0
    clean_up = True
    track = False  # set to True to enable wandb tracking
    wandb_project = "KGE_Transfer"
    seed = 42
    additional_params = ""  # set to anything you want appended to the args

    # For transfer learning, use output from previous model as init_checkpoint for next
    prev_save_dir = None
    local_time = time.localtime()
    timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
    random.seed(seed)  # Set random seed for reproducibility
    for _ in range(max_dataset_loop):
        # randomize the datasets_train order for each loop
        random.shuffle(datasets_train)
        for _, dataset in enumerate(datasets_train):
            cmd, save_dir = prepare_command(
                dataset, model, negative_sample_size, batch_size, hidden_dim, gamma, alpha,
                learning_rate, max_steps, test_batch_size, valid_steps, saving_metric,
                saving_threshold, seed, timestamp
            )

            # If this is not the first dataset, add transfer learning options
            if prev_save_dir:
                cmd += ["--init_checkpoint", prev_save_dir]
                if reload_entities:
                    cmd += ["--reload_entities"]
                if reload_relationship:
                    cmd += ["--reload_relationship"]

            if additional_params:
                cmd += additional_params.split()

            if str(model).lower() == "rotate":
                cmd += ["--double_entity_embedding"]

            if clean_up:
                cmd += ["--clean_up"]

            if track:
                cmd += ["-track", "--wandb_project", wandb_project]

            print("\n===============================")
            print(f"Training on dataset: {dataset}")
            print("Command:", " ".join(cmd))
            print("===============================")
            subprocess.run(cmd, check=True)

            # After training, update prev_save_dir for transfer learning
            prev_save_dir = save_dir
        
        for dataset in dataset_valid:
            cmd, save_dir = prepare_command(
                dataset, model, negative_sample_size, batch_size, hidden_dim, gamma, alpha,
                learning_rate, max_steps, test_batch_size, valid_steps, saving_metric,
                saving_threshold, seed, timestamp
            )

            # If this is not the first dataset, add transfer learning options
            if prev_save_dir:
                cmd += ["--init_checkpoint", prev_save_dir]
                if reload_entities:
                    cmd += ["--reload_entities", "--freeze_entities"]
                if reload_relationship:
                    cmd += ["--reload_relationship", "--freeze_relationship"]

            if additional_params:
                cmd += additional_params.split()

            if str(model).lower() == "rotate":
                cmd += ["--double_entity_embedding"]

            if clean_up:
                cmd += ["--clean_up"]

            if track:
                cmd += ["-track", "--wandb_project", wandb_project]

            print("\n===============================")
            print(f"Validating on dataset: {dataset}")
            print("Command:", " ".join(cmd))
            print("===============================")
            subprocess.run(cmd, check=True)
    
    for dataset in dataset_test:
        cmd, save_dir = prepare_command(
            dataset, model, negative_sample_size, batch_size, hidden_dim, gamma, alpha,
            learning_rate, max_steps, test_batch_size, valid_steps, saving_metric,
            saving_threshold, seed, timestamp
        )

        # If this is not the first dataset, add transfer learning options
        if prev_save_dir:
            cmd += ["--init_checkpoint", prev_save_dir]
            if reload_entities:
                cmd += ["--reload_entities", "--freeze_entities"]
            if reload_relationship:
                cmd += ["--reload_relationship", "--freeze_relationship"]

        if additional_params:
            cmd += additional_params.split()

        if str(model).lower() == "rotate":
            cmd += ["--double_entity_embedding"]

        if clean_up:
            cmd += ["--clean_up"]

        if track:
            cmd += ["-track", "--wandb_project", wandb_project]

        print("\n===============================")
        print(f"Testing on dataset: {dataset}")
        print("Command:", " ".join(cmd))
        print("===============================")
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
