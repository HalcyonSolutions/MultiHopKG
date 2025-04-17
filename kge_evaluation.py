"""
Script to evaluate a Knowledge Graph Embedding (KGE) model on a KGE test dataset (link prediction).
This script loads a pre-trained KGE model, prepares the test dataset, and computes evaluation metrics.

Use this script to ensure that the model is performing the same after saving and loading or transfering.
"""
import argparse
import json
import os

import logging

import numpy as np
import torch

import multihopkg.data_utils as data_utils
from multihopkg.utils.data_splitting import read_triple

from multihopkg.exogenous.sun_models import KGEModel

def get_args() -> argparse.Namespace:
    """
    Get the command line arguments.
    """
    parser = argparse.ArgumentParser(description="KGE Evaluation (inference) script")

    # Dataset Parameters
    parser.add_argument("--data_dir", type=str, default="data/FB15k", help="Path to the KGE dataset directory")
    parser.add_argument("--countries", action="store_true", help="Use countries dataset")

    # Model Parameters
    parser.add_argument("--model", type=str, default="pRotatE", help="Model name")
    parser.add_argument("--trained_model_path", type=str, default="models/protatE_FB15k_dim1000", help="Path to the trained KGE model directory")
    parser.add_argument("--gamma", type=float, default=12.0, help="Gamma value for the model")

    # Evaluation Parameters
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for evaluation (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument("--cpu_num", type=int, default=10, help="Number of CPU threads to use")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Batch size for testing")
    parser.add_argument("--test_log_steps", type=int, default=1000, help="Log steps for testing")
    
    args = parser.parse_args()
    return args

def override_args(args: argparse.Namespace, model: KGEModel) -> None:
    """
    Override the default arguments with any additional ones you want to set.
    """
    args.nrelation = model.nrelation
    args.nentity = model.nentity
    args.cuda = args.device
    return args

def setup_logging(log_dir: str) -> None:
    """
    Set up logging to write to a file in the specified directory.
    """
    log_file = os.path.join(log_dir, "evaluation_metrics.log")
    logging.basicConfig(
        filename=log_file,               # Log file path
        filemode="w",                    # Overwrite the log file
        level=logging.INFO,              # Log level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    )
    print(f"Logging metrics to {log_file}")

def save_metrics(metrics: dict, output_file: str) -> None:
    """
    Save the evaluation metrics to a JSON file.
    """
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {output_file}")

def logging_metrics(metrics: dict) -> None:
    """
    Log the evaluation metrics to a file.
    """
    logging.info(f"Evaluation Metrics for {metrics['model_name']} on {metrics['data_dir']}")
    for key, value in metrics.items():
        logging.info(f"{key}: {value}")

def print_metrics(metrics: dict) -> None:
    """
    Print the evaluation metrics to the console.
    """
    print(f"Evaluation Metrics for {metrics['model_name']} on {metrics['data_dir']}")
    for key, value in metrics.items():
        print(f"{key}: {value}")

def main():

    args = get_args()

    # Set up logging
    setup_logging(args.trained_model_path)

    # Set up JSON file for saving metrics
    json_file = os.path.join(args.trained_model_path, "evaluation_metrics.json")
    #-------------------------------------
    'Loading Phase'

    # Load the dictionaries
    _, ent2id, _, rel2id =  data_utils.load_dictionaries(args.data_dir)

    # Load the model's embeddings
    entity_embeddings = np.load(os.path.join(args.trained_model_path, "entity_embedding.npy"))
    relation_embeddings = np.load(os.path.join(args.trained_model_path, "relation_embedding.npy"))
    checkpoint = torch.load(os.path.join(args.trained_model_path , "checkpoint"))
    
    # Load the model
    kge_model = KGEModel.from_pretrained(
        model_name=args.model,
        entity_embedding=entity_embeddings,
        relation_embedding=relation_embeddings,
        gamma=args.gamma,
        state_dict=checkpoint["model_state_dict"]
    ).to(args.device)

    # Override the arguments with the model's parameters
    args = override_args(args, kge_model)

    # Information computed by knowldege graph for future dependency injection
    dim_entity = kge_model.get_entity_dim()
    dim_relation = kge_model.get_relation_dim()

    # Load the train, validation, and test triples
    train_triples = read_triple(os.path.join(args.data_dir, 'train.txt'), ent2id, rel2id)
    valid_triples = read_triple(os.path.join(args.data_dir, 'valid.txt'), ent2id, rel2id)
    test_triples = read_triple(os.path.join(args.data_dir, 'test.txt'), ent2id, rel2id)

    #----------------------------------------------------
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    del train_triples
    del valid_triples

    #-------------------------------------
    'Evaluation Phase'
    print(f":: Running Evaluation on {args.model} on {args.data_dir} with embedding size of {dim_entity}")

    metrics = KGEModel.test_step(
        model=kge_model,
        test_triples=test_triples,
        all_true_triples=all_true_triples,
        args=args
    )

    # Additional Sanity Metrics
    metrics.update({
        "model_name": kge_model.model_name,
        "gamma": args.gamma,
        "embedding-range": kge_model.embedding_range.detach().cpu().item(),
        "entity-embedding-size": kge_model.get_entity_dim(),
        "relation-embedding-size": kge_model.get_relation_dim(),
        "epsilon": kge_model.epsilon,
        "trained_model_path": args.trained_model_path,
        "data_dir": args.data_dir,
    })

    #-------------------------------------
    'Printing and Logging Phase'
    
    print_metrics(metrics)
    
    logging_metrics(metrics)

    save_metrics(metrics, json_file)



if __name__ == "__main__":
    main(),