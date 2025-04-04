#!/usr/bin/env python3

"""
 Copyright (c) 2018, salesforce.com, inc.
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Experiment Portal.
"""

import argparse
import json
import logging
import os
import pdb
from typing import List, Tuple, Dict, Any, DefaultDict
import debugpy
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from transformers.models.oneformer.image_processing_oneformer import (
    convert_segmentation_map_to_binary_masks,
)
from torch.utils.tensorboard import SummaryWriter 

import wandb
from rich import traceback
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BartConfig,
    BertModel,
    PreTrainedTokenizer,
)

import multihopkg.data_utils as data_utils
from multihopkg.environments import Observation
from multihopkg.exogenous.sun_models import KGEModel, get_embeddings_from_indices
from multihopkg.models_language.classical import HunchBart, collate_token_ids_batch
from multihopkg.logging import setup_logger
from multihopkg.rl.graph_search.cpg import ContinuousPolicyGradient
from multihopkg.rl.graph_search.pn import ITLGraphEnvironment
from multihopkg.run_configs import alpha
from multihopkg.run_configs.common import overload_parse_defaults_with_yaml
from multihopkg.utils.convenience import tensor_normalization
from multihopkg.utils.setup import set_seeds
from multihopkg.vector_search import ANN_IndexMan, ANN_IndexMan_pRotatE
from multihopkg.logs import torch_module_logging
from multihopkg.utils.wandb import histogram_all_modules
from multihopkg.emb.operations import angular_difference
from multihopkg.utils_debug.dump_evals import dump_evaluation_metrics

# PCA
from sklearn.decomposition import PCA

import io
from PIL import Image

traceback.install()
wandb_run = None

# TODO: Remove before final realease, this is purely for debugging
in_dev_mode = False


def initialize_model_directory(args, random_seed=None):
    # add model parameter info to model directory
    # TODO: We might2ant our implementation of something like this later
    raise NotImplementedError


def initial_setup() -> Tuple[argparse.Namespace, PreTrainedTokenizer, PreTrainedTokenizer, logging.Logger]:
    global logger
    args = alpha.get_args()
    args = overload_parse_defaults_with_yaml(args.preferred_config, args)

    set_seeds(args.seed)
    logger = setup_logger("__MAIN__")

    # Get Tokenizer
    question_tokenizer = AutoTokenizer.from_pretrained(args.question_tokenizer_name)
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_tokenizer_name)

    assert isinstance(args, argparse.Namespace)

    return args, question_tokenizer, answer_tokenizer, logger


def prep_questions(questions: List[torch.Tensor], model: BertModel):
    embedded_questions = model(questions)
    return embedded_questions


def batch_loop_dev(
    env: ITLGraphEnvironment,
    mini_batch: pd.DataFrame,  # Perhaps change this ?
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    steps_in_episode: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Specifically for computing any extra metrics on the dev set.
    Otherwise, this is the same as `batch_loop`.
    """

    ########################################
    # Start the batch loop with zero grad
    ########################################
    nav_agent.zero_grad()
    device = nav_agent.fc1.weight.device

    # Deconstruct the batch
    questions = mini_batch["Question"].tolist()
    answers = mini_batch["Answer"].tolist()
    relevant_entities = mini_batch["Relevant-Entities"].tolist()
    relevant_rels = mini_batch["Relevant-Relations"].tolist()
    answer_id = mini_batch["Answer-Entity"].tolist()
    question_embeddings = env.get_llm_embeddings(questions, device)
    answer_ids_padded_tensor = collate_token_ids_batch(answers, pad_token_id).to(torch.int32).to(device)
    pad_mask = answer_ids_padded_tensor.ne(pad_token_id)

    logger.warning(f"About to go into rollout")
    log_probs, llm_rewards, kg_rewards, eval_extras = rollout(
        steps_in_episode,
        nav_agent,
        hunch_llm,
        env,
        question_embeddings,
        answer_ids_padded_tensor,
        relevant_entities = relevant_entities,
        relevant_rels = relevant_rels,
        answer_id = answer_id,
        dev_mode=True,
    )

    ########################################
    # Calculate Reinforce Objective
    ########################################
    logger.warning(f"We just left dev rollout")
    #-------------------------------------------------------------------------
    'LLM Rewards'
    llm_rewards_t = (
        torch.stack(llm_rewards)
    ).permute(1,0,2)  # TODO: I think I need to add the gamma here

    assert not torch.isnan(llm_rewards_t).any(), "NaN detected in the llm rewards (batch_loop_dev). Aborting training."

    # Get only masked, then mean
    llm_rewards_t_unpacked = []
    for i, reward_batch_element in enumerate(llm_rewards_t):
        mask_for_element = pad_mask[i][1:].unsqueeze(0).repeat(steps_in_episode,1)
        filtered_rewards = reward_batch_element[mask_for_element].reshape(steps_in_episode, -1)
        mean_reward = torch.mean(filtered_rewards, dim=-1)
        llm_rewards_t_unpacked.append(mean_reward)
    llm_rewards_t = torch.stack(llm_rewards_t_unpacked)

    log_probs_t = torch.stack(log_probs).T
    num_steps = log_probs_t.shape[-1]

    assert not torch.isnan(log_probs_t).any(), "NaN detected in the log probs (batch_loop_dev). Aborting training."

    # TODO: Check if this is not bad. 
    llm_rewards_t = llm_rewards_t.expand_as(log_probs_t) # TOREM: This is a hack to make the shapes match
    #-------------------------------------------------------------------------
    'Knowledge Graph Environment Rewards'
    kg_rewards_t = (
        torch.stack(kg_rewards)
    ).permute(1,0,2) # Correcting to Shape: (batch_size, num_steps, reward_type)
    kg_rewards_t = kg_rewards_t.squeeze(2) # Shape: (batch_size, num_steps)

    assert not torch.isnan(kg_rewards_t).any(), "NaN detected in the kg rewards (batch_loop_dev). Aborting training."

    #-------------------------------------------------------------------------
    gamma = nav_agent.gamma
    discounted_rewards = torch.zeros_like(llm_rewards_t.clone()).to(llm_rewards_t.device) # Shape: (batch_size, num_steps)
    discounted_rewards[:,-1] = llm_rewards_t[:,-1] + kg_rewards_t[:,-1]
    for t in reversed(range(num_steps - 1)):
        discounted_rewards[:,t] += gamma * (llm_rewards_t[:,t + 1] + kg_rewards_t[:,t + 1])

    # Sample-wise normalization of the rewards for stability

    discounted_rewards = (discounted_rewards - discounted_rewards.mean(axis=-1)[:, torch.newaxis]) / (discounted_rewards.std(axis=-1)[:, torch.newaxis] + 1e-8)
    
    pg_loss = -discounted_rewards * log_probs_t # Have to negate it into order to do gradient ascent

    # logger.info(f"Does pg_loss require grad? {pg_loss.requires_grad}")

    return pg_loss, eval_extras


def batch_loop(
    env: ITLGraphEnvironment,
    mini_batch: pd.DataFrame,  # Perhaps change this ?
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    steps_in_episode: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    ########################################
    # Start the batch loop with zero grad
    ########################################
    nav_agent.zero_grad()
    device = nav_agent.fc1.weight.device

    # Deconstruct the batch
    questions = mini_batch["Question"].tolist()
    answers = mini_batch["Answer"].tolist()
    relevant_entities = mini_batch["Relevant-Entities"].tolist()
    relevant_rels = mini_batch["Relevant-Relations"].tolist()
    answer_id = mini_batch["Answer-Entity"].tolist()
    question_embeddings = env.get_llm_embeddings(questions, device)
    answer_ids_padded_tensor = collate_token_ids_batch(answers, pad_token_id).to(torch.int32).to(device)
    pad_mask = answer_ids_padded_tensor.ne(pad_token_id)

    log_probs, llm_rewards, kg_rewards, eval_extras = rollout(
        steps_in_episode,
        nav_agent,
        hunch_llm,
        env,
        question_embeddings,
        answer_ids_padded_tensor,
        relevant_entities = relevant_entities,
        relevant_rels = relevant_rels,
        answer_id = answer_id,
    )

    ########################################
    # Calculate Reinforce Objective
    ########################################
    logger.debug("About to calculate rewards")
    #-------------------------------------------------------------------------
    'LLM Rewards'
    llm_rewards_t = (
        torch.stack(llm_rewards)
    ).permute(1,0,2)  # TODO: I think I need to add the gamma here

    # Get only masked, then mean
    llm_rewards_t_unpacked = []
    for i, reward_batch_element in enumerate(llm_rewards_t):
        mask_for_element = pad_mask[i][1:].unsqueeze(0).repeat(steps_in_episode,1)
        filtered_rewards = reward_batch_element[mask_for_element].reshape(steps_in_episode, -1)
        mean_reward = torch.mean(filtered_rewards, dim=-1)
        llm_rewards_t_unpacked.append(mean_reward)
    llm_rewards_t = torch.stack(llm_rewards_t_unpacked)

    log_probs_t = torch.stack(log_probs).T
    num_steps = log_probs_t.shape[-1]

    # TODO: Check if this is not bad. 
    llm_rewards_t = llm_rewards_t.expand_as(log_probs_t) # TOREM: This is a hack to make the shapes match
    #-------------------------------------------------------------------------
    'Knowledge Graph Environment Rewards'
    kg_rewards_t = (
        torch.stack(kg_rewards)
    ).permute(1,0,2) # Correcting to Shape: (batch_size, num_steps, reward_type)
    kg_rewards_t = kg_rewards_t.squeeze(2) # Shape: (batch_size, num_steps)

    #-------------------------------------------------------------------------
    # ! Modifying the rewards to stabilize training
    # ! Approach 1: Use the rewards as is

    # ! Approach 2: Use the discounted rewards
    gamma = nav_agent.gamma
    discounted_rewards = torch.zeros_like(llm_rewards_t.clone()).to(llm_rewards_t.device) # Shape: (batch_size, num_steps)
    discounted_rewards[:,-1] = llm_rewards_t[:,-1] + kg_rewards_t[:,-1]
    for t in reversed(range(num_steps - 1)):
        discounted_rewards[:,t] += gamma * (llm_rewards_t[:,t + 1] + kg_rewards_t[:,t + 1])

    # Sample-wise normalization of the rewards for stability

    discounted_rewards = (discounted_rewards - discounted_rewards.mean(axis=-1)[:, torch.newaxis]) / (discounted_rewards.std(axis=-1)[:, torch.newaxis] + 1e-8)

    # ! Approach 3: Use the rewards as is but scale them
    # Scale rewards instead of normalizing
    # rewards_t = rewards_t / (torch.abs(rewards_t).max() + 1e-8)

    # ! Approach 4: Use the rewards as is but normalize them with the mean and std
    # Normalize the rewards
    # rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

    pg_loss = -discounted_rewards * log_probs_t # Have to negate it into order to do gradient ascent
    # TODO: Perhaps only use the first few steps ?

    # ! Approach 2: Use the discounted rewards
    # pg_loss = -1 * (discounted_rewards * log_probs_t)

    # TOREM: Maybe we don't need this visualization
    # ########################################
    # # Adding TorchViz Dot here
    # ########################################
    # dot = make_dot(pg_loss.sum())
    # dot.render("model_graph", format="png")

    return pg_loss, eval_extras


def evaluate_training(
    env: ITLGraphEnvironment,
    dev_df: pd.DataFrame,
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    steps_in_episode: int,
    batch_size_dev: int,
    batch_count: int,
    verbose: bool,
    visualize: bool,
    writer: SummaryWriter,
    question_tokenizer: PreTrainedTokenizer,
    answer_tokenizer: PreTrainedTokenizer,
    wandb_on: bool,
    answer_id: List[int] = None,
):
    print("Running evalute_training")

    global in_dev_mode
    num_batches = len(dev_df) // batch_size_dev
    nav_agent.eval()
    hunch_llm.eval()
    in_dev_mode = True  # TOREM: This is only for debugging
    env.eval()
    # env.question_embedding_module.eval()
    assert (
        not env.question_embedding_module.training
    ), "The question embedding module must not be in training mode"

    batch_cumulative_metrics = {
        "dev/batch_count": [batch_count],
        "dev/pg_loss": [],
    }  # For storing results from all batches
    current_evaluations = (
        {}
    )  # For storing results from last batch. Otherwise too much info

    with torch.no_grad():
        # for batch_id in range(num_batches):

        # We will only evaluate on the last batch
        batch_id = num_batches - 1
        # TODO: Get the rollout working
        mini_batch = dev_df[
            batch_id * batch_size_dev : (batch_id + 1) * batch_size_dev
        ]
        if not isinstance(  # TODO: Remove this assertion once it is never ever met again
            mini_batch, pd.DataFrame
        ):  # For the lsp to give me a break
            raise RuntimeError(
                f"The mini batch is not a pd.DataFrame, but a {type(mini_batch)}. Please check the data loading code."
            )
        # if (
        #     len(mini_batch) < batch_size_dev
        # ):  # We dont want to evaluate on incomplete batches
        #     continue

        current_evaluations["reference_questions"] = mini_batch["Question"]
        current_evaluations["true_answer"] = mini_batch["Answer"]
        current_evaluations["relevant_entities"] = mini_batch["Relevant-Entities"]
        current_evaluations["relevant_relations"] = mini_batch["Relevant-Relations"]
        current_evaluations["true_answer_id"] = mini_batch["Answer-Entity"]

        # Get the Metrics
        bos_token_id = answer_tokenizer.bos_token_id
        eos_token_id = answer_tokenizer.eos_token_id
        pad_token_id = answer_tokenizer.pad_token_id
        if bos_token_id is None or eos_token_id is None or pad_token_id is None:
            raise ValueError("Assumptions Wrong. The answer_tokenizer must have a bos_token_id, eos_token_id and pad_token_id")
        pg_loss, eval_extras = batch_loop_dev(
            env,
            mini_batch,
            nav_agent,
            hunch_llm,
            steps_in_episode,
            pad_token_id,
        )

        'Extract all the variables from eval_extras'
        for k, v in eval_extras.items():
            current_evaluations[k] = v

        # Accumlate the metrics
        batch_cumulative_metrics["dev/pg_loss"].append(pg_loss.mean().item())

    ########################################
    # Take `current_evaluations` as
    # a sample of batches and dump its results
    ########################################
    if verbose and logger:
        graph_annotation = []
        if env.entity2title:
            for i0 in range(len(env.graph_annotation)):
                if env.graph_annotation[i0] in env.entity2title.keys():
                    graph_annotation.append(env.entity2title[env.graph_annotation[i0]])
                else:
                    graph_annotation.append("")

        # eval_extras has variables that we need
        just_dump_it_here = "./logs/evaluation_dumps.log"

        answer_kge_tensor = get_embeddings_from_indices(
            env.knowledge_graph.entity_embedding,
            torch.tensor(answer_id, dtype=torch.int),
        ).unsqueeze(1) # Shape: (batch, 1, embedding_dim)

        logger.warning(f"About to go into dump_evaluation_metrics")
        dump_evaluation_metrics(
            path_to_log=just_dump_it_here,
            evaluation_metrics_dictionary=current_evaluations,
            vector_entity_searcher=env.ann_index_manager_ent,															 
            vector_rel_searcher=env.ann_index_manager_rel,
            question_tokenizer=question_tokenizer,
            answer_tokenizer=answer_tokenizer,
            answer_kge_tensor=answer_kge_tensor,
            embedding_range=env.knowledge_graph.embedding_range.item(),
            id2entity=env.id2entity,					   
            id2relations=env.id2relation,
            entity2title=env.entity2title,
            relation2title=env.relation2title,
            writer=writer,						  
            wandb_on=wandb_on,
            logger=logger,
        )
        logger.warning(f"We just left dump_evaluation_metrics")
        # TODO: Maybe dump the language metrics in wandb ?
        # table = wandb.Table(
        #     columns=["Question", "Path Taken", "Real Answer", "Given Answer"]
        # )


    ########################################
    # Average out all metrics across batches
    # The dump to wandb
    ########################################
    """
    for k, v in batch_cumulative_metrics.items():
        metric_to_report = 0
        if isinstance(v[0],torch.Tensor):
            metric_to_report = torch.stack(v).mean()
        elif isinstance(v[0], int) or isinstance(v[0], float):
            metric_to_report = v[0]
        else:
            raise ValueError(f"The metric to report is not a tensor or int but rather {type(v[0])}")

        if wandb_run is not None:
            wandb.log({k: metric_to_report})
        logger.debug(f"Metric '{k}' has value {metric_to_report}")


    nav_agent.train()
    hunch_llm.train()
    env.train()
    dev_mode = False
    logger.info("Done with Evaluation")
    """
    # TODO: Implement this

def train_multihopkg(
    batch_size: int,
    batch_size_dev: int,
    epochs: int,
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    learning_rate: float,
    steps_in_episode: int,
    env: ITLGraphEnvironment,
    start_epoch: int,
    train_data: pd.DataFrame,
    dev_df: pd.DataFrame,
    mbatches_b4_eval: int,
    verbose: bool,
    visualize: bool,
    question_tokenizer: PreTrainedTokenizer,
    answer_tokenizer: PreTrainedTokenizer,
    track_gradients: bool,
    num_batches_till_eval: int,
    wandb_on: bool,
):
    # TODO: Get the rollout working

    # Print Model Parameters + Perhaps some more information
    print(
        "--------------------------\n" "Model Parameters\n" "--------------------------"
    )
    for name, param in nav_agent.named_parameters():
        print(name, param.numel(), "requires_grad={}".format(param.requires_grad))

    for name, param in env.named_parameters():
        if param.requires_grad: print(name, param.numel(), "requires_grad={}".format(param.requires_grad))

    writer = SummaryWriter(log_dir=f'runs')
    
    # Just use Adam Optimizer by default
    # optimizer = torch.optim.Adam(  # type: ignore
    #     filter(lambda p: p.requires_grad, nav_agent.parameters()), lr=learning_rate
    # )

    named_param_map = {param: name for name, param in (list(nav_agent.named_parameters()) + list(env.named_parameters()) + list(hunch_llm.named_parameters()))}
    optimizer = torch.optim.Adam(  # type: ignore
        filter(
            lambda p: p.requires_grad,
            list(env.concat_projector.parameters()) + list(nav_agent.parameters()) + list(hunch_llm.embedding_translator.parameters())
        ),
        lr=learning_rate
    )

    modules_to_log: List[nn.Module] = [nav_agent]

    # Variable to pass for logging
    batch_count = 0
    bos_token_id = answer_tokenizer.bos_token_id
    eos_token_id = answer_tokenizer.eos_token_id
    pad_token_id = answer_tokenizer.pad_token_id
    if bos_token_id is None or eos_token_id is None or pad_token_id is None:
        raise ValueError("Assumptions Wrong. The answer_tokenize must have a bos_token_id, eos_token_id and pad_token_id")

    # variables to track vanishing gradient for nav_agent
    mu_tracker = [[], []] # mean, and std
    sigma_tracker = [[], []]
    fc1_tracker = [[], []]

    # Replacement for the hooks
    if track_gradients:
        grad_logger = torch_module_logging.ModuleSupervisor({
            "navigation_agent" : nav_agent, 
            "hunch_llm" : hunch_llm
        })

    ########################################
    # Epoch Loop
    ########################################
    for epoch_id in tqdm(range(start_epoch, epochs), desc="Epoch"):

        logger.info("Epoch {}".format(epoch_id))
        # TODO: Perhaps evaluate the epochs?

        # Set in training mode
        nav_agent.train()
        batch_rewards = []
        entropies = []

        ##############################
        # Batch Loop
        ##############################
        # TODO: update the parameters.
        for sample_offset_idx in tqdm(range(0, len(train_data), batch_size), desc="Training Batches", leave=False):
            mini_batch = train_data[sample_offset_idx : sample_offset_idx + batch_size]
            # pdb.set_trace()
            assert isinstance(
                mini_batch, pd.DataFrame
            )  # For the lsp to give me a break
            ########################################
            # Evaluation
            ########################################
            if batch_count % mbatches_b4_eval == 0:
                evaluate_training(
                    env,
                    dev_df,
                    nav_agent,
                    hunch_llm,
                    steps_in_episode,
                    batch_size_dev,
                    batch_count,
                    verbose,
                    visualize,
                    writer,
                    question_tokenizer,
                    answer_tokenizer,
                    wandb_on,
                    answer_id=mini_batch["Answer-Entity"].tolist(),  # Extract answer_id from mini_batch
                )

            ########################################
            # Training
            ########################################
            optimizer.zero_grad()
            pg_loss, _ = batch_loop(
                env, mini_batch, nav_agent, hunch_llm, steps_in_episode, bos_token_id, eos_token_id, pad_token_id
            )

            if torch.isnan(pg_loss).any():
                logger.error("NaN detected in the loss. Aborting training.")
                # pdb.set_trace()

            # Logg the mean, std, min, max of the rewards
            reinforce_terms_mean = pg_loss.mean()
            reinforce_terms_mean_item = reinforce_terms_mean.item()
            reinforce_terms_std_item = pg_loss.std().item()
            reinforce_terms_min_item = pg_loss.min().item()
            reinforce_terms_max_item = pg_loss.max().item()
            logger.debug(f"Reinforce terms mean: {reinforce_terms_mean_item}, std: {reinforce_terms_std_item}, min: {reinforce_terms_min_item}, max: {reinforce_terms_max_item}")

            # TODO: Uncomment and try: 
            pg_loss = tensor_normalization(pg_loss)

            batch_rewards.append(reinforce_terms_mean_item)
            logger.debug("Bout to go backwords")
            reinforce_terms_mean.backward()

            # TODO: get grad distribution parameters,
            # Inspecting vanishing gradient
            
            if sample_offset_idx == 0:

                # Ask for the DAG to be dumped
                if track_gradients:
                    grad_logger.dump_visual_dag(destination_path=f"./figures/grads/dag_{epoch_id:02d}.png", figsize=(10, 100)) # type: ignore

            optimizer.step()

            if torch.all(nav_agent.mu_layer.weight.grad == 0):
                print("Gradients are zero for mu_layer!")

            # TODO: get grad distribution parameters,
            # Inspecting vanishing gradient
            
            if sample_offset_idx % num_batches_till_eval == 0 and verbose:
                # Retrieve named parameters from the optimizer
                named_params = [
                    (named_param_map[param], param)
                    for group in optimizer.param_groups
                    for param in group['params']
                ]

                # Wandb hisotram of modules
                histograms = histogram_all_modules(modules_to_log, num_buckets=20)
                # Report the histograms to wandb
                if wandb_on:
                    for name, histogram in histograms.items():
                        wandb.log({f"{name}/Histogram": wandb.Histogram(np_histogram=histogram)})


                # Iterate and calculate gradients as needed
                for name, param in named_params:
                    if param.requires_grad and ('bias' not in name) and (param.grad is not None):
                        if name == 'weight': name = 'concat_projector.weight'               
                        grads = param.grad.detach().cpu()
                        weights = param.detach().cpu()

                        write_parameters(grads, name, "Gradient", writer, epoch_id)
                        write_parameters(weights, name, "Weights", writer, epoch_id)

                        if wandb_on:
                            wandb.log({f"{name}/Gradient": wandb.Histogram(grads.numpy().flatten())})
                            wandb.log({f"{name}/Weights": wandb.Histogram(weights.numpy().flatten())})
                        elif visualize:
                            write_histogram(grads.numpy().flatten(), name, 'g', "Gradient Histogram", "Grad Value", "Frequency", writer, epoch_id)
                            write_histogram(weights.numpy().flatten(), name, 'b', "Weights Histogram", "Weight Value", "Frequency", writer, epoch_id)

            optimizer.step()
            if wandb_on:
                loss_item = pg_loss.mean().item()
                logger.info(f"Submitting train/pg_loss: {loss_item} to wandb")
                wandb.log({"train/pg_loss": loss_item})

            batch_count += 1

def write_parameters(data: torch.Tensor, layer_name: str, value_type: str, writer: SummaryWriter, epoch_id: int):
    mean = data.mean().item()
    var = data.var().item()

    writer.add_scalar(f'{layer_name}/{value_type} Mean', mean, epoch_id)
    writer.add_scalar(f'{layer_name}/{value_type} Var', var, epoch_id)

def write_histogram(data: np.ndarray, layer_name: str, color: str, title: str, xlabel: str, ylabel: str, writer: SummaryWriter, epoch_id: int):

    plt.figure(1)
    plt.hist(data, bins=50, alpha=0.75, color=color)
    plt.title(f"{layer_name} {title}")
    plt.xlabel(f"{xlabel}")
    plt.ylabel(f"{ylabel}")

    # Show the grid
    plt.grid(True)

    # Save the histogram to a BytesIO buffer
    hist_buf = io.BytesIO()
    plt.savefig(hist_buf, format="png")
    hist_buf.seek(0)

    # Convert the buffer content to an image and then to a NumPy array in HWC format
    hist_image = Image.open(hist_buf)
    hist_image_np = np.array(hist_image)

    # Add the histogram to TensorBoard
    writer.add_image(
        f"{layer_name}/{title}", hist_image_np, epoch_id, dataformats="HWC"
    )

    # Close the buffer
    hist_buf.close()
    plt.close()

def initialize_path(questions: torch.Tensor):
    # Questions must be turned into queries
    raise NotImplementedError


def calculate_reward(
    hunch_llm: nn.Module,
    obtained_state: torch.Tensor,
    answers_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Will take the answers and give an idea of how close we were.
    This will of course require us to have a language model that will start giving us the  answer.
    """
    batch_size = answers_ids.size(0)
    seq_max_len = answers_ids.size(1)
    hidden_dim = obtained_state.shape[-1]

    # From the obtained_state we will try to find an answer
    conditioning_labels = answers_ids[:, :-1].contiguous().to(dtype=torch.int64)
    teacher_forcing_labels = answers_ids[:, 1:].contiguous().to(dtype=torch.int64)

    answers_inf_softmax = hunch_llm(graph_embeddings=obtained_state, decoder_input_ids=conditioning_labels)

    _, logits = answers_inf_softmax.loss, answers_inf_softmax.logits

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    loss = loss_fn(logits.view(-1, logits.shape[-1]), teacher_forcing_labels.view(-1))

    # TODO: Perhaps Stabilize the loss. Normalize it or SMTH like that
    reward = -loss # We expect this reward function to be concave rather than convex. 

    # Reshape the reward to the batch size
    reward = reward.view(batch_size, -1)

    # # Get indices of the max value of the final output
    # answers_inf_ids = torch.argmax(logits, dim=-1)

    return reward, logits


def rollout(
    # TODO: self.mdl should point to (policy network)
    steps_in_episode,
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    env: ITLGraphEnvironment,
    questions_embeddings: torch.Tensor,
    answers_ids: torch.Tensor,
    relevant_entities: List[List[int]],
    relevant_rels: List[List[int]],
    answer_id: List[int],
    dev_mode: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
    """
    Will execute RL episode rollouts in parallel.
    args:
        kg: Knowledge graph environment.
        num_steps: Number of rollout steps.
        navigator_agent: Policy network.
        graphman: Graph search policy network.
        questions: Questions already pre-embedded to be answered (num_rollouts, question_dim)
        visualize_action_probs: If set, save action probabilities for visualization.
    returns:
        - log_action_probs (torch.TEnsor): For REINFORCE
        - rewards (torch.Tensor):  I mean, also for REINFOCE
    """

    assert steps_in_episode > 0

    ########################################
    # Prepare lists to be returned
    ########################################
    log_action_probs = []
    llm_rewards = []
    kg_rewards = []
    eval_metrics = DefaultDict(list)

    # Dummy nodes ? TODO: Figur eout what they do.
    # TODO: Perhaps here we can enter through the centroid.
    # For now we still with these dummy
    # NOTE: We repeat these entities until we get the right shape:
    # TODO: make sure we keep all seen nodes up to date
    answer_tensor = get_embeddings_from_indices(
            env.knowledge_graph.entity_embedding,
            torch.tensor(answer_id, dtype=torch.int),
    ).unsqueeze(1) # Shape: (batch, 1, embedding_dim)


    # Get initial observation. A concatenation of centroid and question atm. Passed through the path encoder
    observations = env.reset(
        questions_embeddings,
        answer_ent = answer_id,
        relevant_ent = relevant_entities
    )

    cur_position, cur_state = observations.position, observations.state
    # Should be of shape (batch_size, 1, hidden_dim)

    # pn.initialize_path(kg) # TOREM: Unecessasry to ask pn to form it for us.
    states_so_far = []
    for t in range(steps_in_episode):

        # Ask the navigator to navigate, agent is presented state, not position
        # State is meant to summrized path history.
        sampled_actions, log_probs, entropies = nav_agent(cur_state)

        # TODO:Make sure we are gettign rewards from the environment.
        observations, kg_extrinsic_rewards, kg_dones = env.step(sampled_actions)
        # Ah ssampled_actions are the ones that have to go against the knowlde garph.

        states = observations.state
        visited_embeddings = observations.position.clone()
        position_ids = observations.position_id.clone()
        # For now, we use states given by the path encoder and positions mostly for debugging
        states_so_far.append(states)

        # VISITED EMBEDDINGS IS THE ENCODER

        ########################################
        # Calculate the Reward
        ########################################
        stacked_states = torch.stack(states_so_far).permute(1, 0, 2)
        # Calculate how close we are
        llm_reward, logits = calculate_reward(
            hunch_llm, stacked_states, answers_ids
        )

        llm_rewards.append(llm_reward)

        kg_intrinsic_reward = angular_difference(
            observations.kge_cur_pos.unsqueeze(1)/(env.knowledge_graph.embedding_range.item()/torch.pi),
            answer_tensor/(env.knowledge_graph.embedding_range.item()/torch.pi),
            smooth=True
        ).norm(dim=-1)

        # TODO: Ensure the that the model stays within range of answer, otherwise set kg_done back to false so intrinsic reward kicks back in.
        kg_rewards.append(kg_dones*kg_extrinsic_rewards - torch.logical_not(kg_dones)*kg_intrinsic_reward) # Merging positive environment rewards with negative intrinsic ones

        # TODO: Make obseervations not rely on the question

        ########################################
        # Log Stuff for across batch
        ########################################
        cur_state = states
        log_action_probs.append(log_probs)

        ########################################
        # Stuff that we will only use for evaluation
        ########################################
        if dev_mode:
            eval_metrics["sampled_actions"].append(sampled_actions.detach().cpu())
            eval_metrics["visited_embeddings"].append(visited_embeddings.detach().cpu())
            eval_metrics["position_ids"].append(position_ids.detach().cpu())
            eval_metrics["kge_cur_pos"].append(observations.kge_cur_pos.detach().cpu())
            eval_metrics["kge_prev_pos"].append(observations.kge_prev_pos.detach().cpu())
            eval_metrics["kge_action"].append(observations.kge_action.detach().cpu())

            'LLM Metrics'
            eval_metrics["hunch_llm_final_guesses"].append(logits.argmax(dim=-1))
            llm_softmax = torch.nn.functional.softmax(logits, dim=-1)
            llm_entropies = -torch.sum(llm_softmax * torch.log(llm_softmax), dim=-1)
            eval_metrics["hunch_llm_entropy"].append(llm_entropies.mean().detach().cpu())

            'KGE Metrics'
            eval_metrics["kg_extrinsic_rewards"].append(kg_extrinsic_rewards.detach().cpu())
            eval_metrics["kg_intrinsic_reward"].append(kg_intrinsic_reward.detach().cpu())
            eval_metrics["kg_dones"].append(kg_dones.detach().cpu())

    if dev_mode:
        eval_metrics = {k: torch.stack(v) for k, v in eval_metrics.items()}

    # Return Rewards of Rollout as a Tensor
    return log_action_probs, llm_rewards, kg_rewards, eval_metrics


def load_qa_data(
    cached_metadata_path: str,
    raw_QAData_path,
    question_tokenizer_name: str,
    answer_tokenizer_name: str,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int], 
    force_recompute: bool = False,
):

    if os.path.exists(cached_metadata_path) and not force_recompute:
        logger.info(
            f"\033[93m Found cache for the QA data {cached_metadata_path} will load it instead of working on {raw_QAData_path}. \033[0m"
        )
        # Read the first line of the raw csv to count the number of columns
        train_metadata = json.load(open(cached_metadata_path.format(question_tokenizer_name, answer_tokenizer_name)))
        saved_paths: Dict[str, str] = train_metadata["saved_paths"]

        train_df = pd.read_parquet(saved_paths["train"])
        # TODO: Eventually use this to avoid data leakage
        dev_df = pd.read_parquet(saved_paths["dev"])
        test_df = pd.read_parquet(saved_paths["test"])

        # Ensure that we are not reading them integers as strings, but also not as floats
        logger.info(
            f"Loaded cached data from \033[93m\033[4m{json.dumps(cached_metadata_path,indent=4)} \033[0m"
        )
    else:
        ########################################
        # Actually compute the data.
        ########################################
        logger.info(
            f"\033[93m Did not find cache for the QA data {cached_metadata_path}. Will now process it from {raw_QAData_path} \033[0m"
        )
        question_tokenizer = AutoTokenizer.from_pretrained(question_tokenizer_name)
        answer_tokenzier   = AutoTokenizer.from_pretrained(answer_tokenizer_name)
        df_split, train_metadata = (
            data_utils.process_and_cache_triviaqa_data(  # TOREM: Same here, might want to remove if not really used
                raw_QAData_path,
                cached_metadata_path,
                question_tokenizer,
                answer_tokenzier,
                entity2id,
                relation2id,
            )
        )
        train_df, dev_df, test_df = df_split.train, df_split.dev, df_split.test
        logger.info(
            f"Done. Result dumped at : \n\033[93m\033[4m{train_metadata['saved_paths']}\033[0m"
        )

    ########################################
    # Train Validation Test split
    ########################################

    # All of this was unecessary it was already beign done before.
    # shuffled_train_df = train_df.sample(frac=1).reset_index(drop=True)
    # train_df, val_df = train_test_split(
    #     shuffled_train_df, test_size=0.2, random_state=42
    # )

    return train_df, dev_df, train_metadata


def main():
    # By default we run the config
    # Process data will determine by itself if there is any data to process
    args, question_tokenizer, answer_tokenizer, logger = initial_setup()
    global wandb_run

    if args.debug:
        logger.info("\033[1;33m Waiting for debugger to attach...\033[0m")
        debugpy.listen(("0.0.0.0", 42020))
        debugpy.wait_for_client()

        # USe debugpy to listen

    ########################################
    # Get the data
    ########################################
    logger.info(":: Setting up the data")

    # Load the KGE Dictionaries
    id2ent, ent2id, id2rel, rel2id =  data_utils.load_dictionaries(args.data_dir)

    # Load the QA Dataset
    train_df, dev_df, train_metadata = load_qa_data(
        cached_metadata_path=args.cached_QAMetaData_path,
        raw_QAData_path=args.raw_QAData_path,
        question_tokenizer_name=args.question_tokenizer_name,
        answer_tokenizer_name=args.answer_tokenizer_name,
        entity2id=ent2id,
        relation2id=rel2id,
        force_recompute=args.force_data_prepro
    )
    if not isinstance(dev_df, pd.DataFrame) or not isinstance(train_df, pd.DataFrame):
        raise RuntimeError(
            "The data was not loaded properly. Please check the data loading code."
        )

    # TODO: Muybe ? (They use it themselves)
    # initialize_model_directory(args, args.seed)
    if args.wandb:
        logger.info(
            f"🪄 Initializing Weights and Biases. Under project name {args.wandb_project_name} and run name {args.wr_name}"
        )
        wandb_run = wandb.init(
            project=args.wandb_project_name,
            name=args.wr_name,
            config=vars(args),
            notes=args.wr_notes,
        )

    ## Agent needs a Knowledge graph as well as the environment
    logger.info(":: Setting up the knowledge graph")

    # TODO: Load the weighs ?
    # knowledge_graph = SunKnowledgeGraph(
    #     model=args.model,
    #     pretrained_sun_model_path=args.pretrained_sun_model_loc,
    #     data_path=args.data_dir,
    #     graph_embed_model_name=args.graph_embed_model_name,
    #     gamma=args.gamma,
    #     id2entity=id2ent,
    #     entity2id=ent2id,
    #     id2relation=id2rel,
    #     relation2id=rel2id,
    #     device=args.device,
    # )

    # TODO: Test this replacement for SunKnowledgeGraph
    entity_embeddings = np.load(os.path.join(args.trained_model_path, "entity_embedding.npy"))
    relation_embeddings = np.load(os.path.join(args.trained_model_path, "relation_embedding.npy"))
    checkpoint = torch.load(os.path.join(args.trained_model_path , "checkpoint"))
    kge_model = KGEModel.from_pretrained(
        model_name=args.model,
        entity_embedding=entity_embeddings,
        relation_embedding=relation_embeddings,
        gamma=args.gamma,
        state_dict=checkpoint["model_state_dict"]
    )

    # Information computed by knowldege graph for future dependency injection
    dim_entity = kge_model.get_entity_dim()
    dim_relation = kge_model.get_relation_dim()
    logger.info("You have reached the exit")

    # Paths for triples
    train_triplets_path = os.path.join(args.data_dir, "train.triples")
    dev_triplets_path = os.path.join(args.data_dir, "dev.triples")
    entity_index_path = os.path.join(args.data_dir, "entity2id.txt")
    relation_index_path = os.path.join(args.data_dir, "relation2id.txt")

    # Get the Module for Approximate Nearest Neighbor Search
    ########################################
    # Setup the ann index.
    # Will be needed for obtaining observations.
    ########################################
    logger.info(":: Setting up the ANN Index")

    ########################################
    # Setup the Vector Searchers
    ########################################
    # ! Currently using approximations, check if it this is the best way to go
    # ! Testing: exact computation
    if args.model == "pRotatE": # for rotational kge models
        ann_index_manager_ent = ANN_IndexMan_pRotatE(
            kge_model.get_all_entity_embeddings_wo_dropout(),
            embedding_range=kge_model.embedding_range.item(),
        )
        ann_index_manager_rel = ANN_IndexMan_pRotatE(
            kge_model.get_all_relations_embeddings_wo_dropout(),
            embedding_range=kge_model.embedding_range.item(),
        )
    else: # for non-rotational kge models
        ann_index_manager_ent = ANN_IndexMan(
            kge_model.get_all_entity_embeddings_wo_dropout(),
            exact_computation=True,
            nlist=100,
        )
        ann_index_manager_rel = ANN_IndexMan(
            kge_model.get_all_relations_embeddings_wo_dropout(),
            exact_computation=True,
            nlist=100,
        )

    if args.visualize:
        # Train the pca, and also get the emebeddings of the graph as an array
        pca = PCA(n_components=2)
        tmp_graph = (kge_model.get_all_entity_embeddings_wo_dropout() .cpu() .detach() .numpy())
        graph_mag = np.abs(tmp_graph[:,:1000] + 1j*tmp_graph[:,1000:])
        graph_pca = pca.fit(graph_mag).transform(graph_mag)

        # Sub-sample it to 100 elements
        random_idx = np.random.choice(graph_pca.shape[0], 100, replace=False)
        graph_pca = graph_pca[random_idx]

        # ! Extract annotations for the graph
        graph_annotation = []
        for i0 in range(min(random_idx.shape[0], 15)): # improve this with an argument
            graph_annotation.append(id2ent[random_idx[i0]])

        # For visualization
        plt.ion()
        fig = plt.figure(0, figsize=(8,6))
        ax = fig.add_subplot(111)
    else:
        graph_pca = None
        graph_annotation = []
        pca = None

    # Setup the pretrained language model
    logger.info(":: Setting up the pretrained language model")
    config = BartConfig.from_pretrained("facebook/bart-base")
    # Access the hidden size (hidden dimension)
    # TODO: Remove the hardcode. Perhaps
    embedding_hidden_size = config.d_model
    embedding_vocab_size = config.vocab_size
    print(
        f"The hidden dimension of the embedding layer is {embedding_hidden_size} and its vocab size is {embedding_vocab_size}"
    )

    # We prepare our custom encoder for Bart Here
    hunch_llm = HunchBart(
        pretrained_bart_model=args.pretrained_llm_for_hunch,
        answer_tokenizer=answer_tokenizer,
        # We convert the graph embeddings to state embeddings obeying current state dimensions
        graph_embedding_dim=args.llm_model_dim, 
    ).to(args.device)

    # # Freeze the Hunch LLM
    # for param in hunch_llm.parameters():
    #     param.requires_grad = False

    if args.further_train_hunchs_llm:
        # TODO: Ensure we dont have to freeze the model for this.
        hunch_llm.freeze_llm()

    # Setup the entity embedding module
    question_embedding_module = AutoModel.from_pretrained(args.question_embedding_model).to(args.device)

    # # Freeze the Question Embedding Module
    # for param in question_embedding_module.parameters():
    #     param.requires_grad = False

    # Setting up the models
    logger.info(":: Setting up the environment")
    env = ITLGraphEnvironment(
        question_embedding_module=question_embedding_module,
        question_embedding_module_trainable=args.question_embedding_module_trainable,
        entity_dim=dim_entity,
        ff_dropout_rate=args.ff_dropout_rate,
        history_dim=args.history_dim,
        history_num_layers=args.history_num_layers,
        knowledge_graph=kge_model,
        relation_dim=dim_relation,
        node_data=args.node_data_path,
        node_data_key=args.node_data_key,
        rel_data=args.relationship_data_path,
        rel_data_key=args.relationship_data_key,
        id2entity=id2ent,
        entity2id=ent2id,
        id2relation=id2rel,
        relation2id=rel2id,
        ann_index_manager_ent=ann_index_manager_ent,
        ann_index_manager_rel=ann_index_manager_rel,
        steps_in_episode=args.num_rollout_steps,
        trained_pca=pca,
        graph_pca=graph_pca,
        graph_annotation=graph_annotation,
        nav_start_emb_type=args.nav_start_emb_type,
    ).to(args.device)

    # Now we load this from the embedding models

    # TODO: Reorganizew the parameters lol
    logger.info(":: Setting up the navigation agent")
    nav_agent = ContinuousPolicyGradient(
        baseline=args.baseline,
        beta=args.beta,
        gamma=args.rl_gamma,
        action_dropout_rate=args.action_dropout_rate,
        action_dropout_anneal_factor=args.action_dropout_anneal_factor,
        action_dropout_anneal_interval=args.action_dropout_anneal_interval,
        num_rollout_steps=args.num_rollout_steps,
        dim_action=dim_relation,
        dim_hidden=args.rnn_hidden,
        dim_observation=args.history_dim,  # observation will be into history
    ).to(args.device)

    # ======================================
    # Visualizaing nav_agent models using Netron
    # Save a model into .onnx format
    # torch_input = torch.randn(12, 768)
    # onnx_program = torch.onnx.dynamo_export(nav_agent, torch_input)
    # onnx_program.save("models/images/nav_agent.onnx")
    # ======================================

    # TODO: Add checkpoint support
    # See args.start_epoch

    # TODO: Load the validation data
    # dev_path = os.path.join(args.data_dir, "dev.triples")
    # data_triple_split_dict = data_utils.sun_load_triples_and_dict( args.pretrained_sun_model_loc)

    data_triple_split_dict, id2entity, id2relation = data_utils.load_triples_and_dict(
        [train_triplets_path, dev_triplets_path],  # TODO: Load the test_data
        entity_index_path,
        relation_index_path,
        group_examples_by_query=False,
        add_reverse_relations=False,
    )

    print("Let me get the head of id2entity:")
    print(id2entity[0])

    # TODO: Make it take check for a checkpoint and decide what start_epoch
    # if args.checkpoint_path is not None:
    #     # TODO: Add it here to load the checkpoint separetely
    #     nav_agent.load_checkpoint(args.checkpoint_path)

    ######## ######## ########
    # Train:
    ######## ######## ########
    start_epoch = 0
    logger.info(":: Training the model")

    if args.visualize:
        args.verbose = True

    train_multihopkg(
        batch_size=args.batch_size,
        batch_size_dev=args.batch_size_dev,
        epochs=args.epochs,
        nav_agent=nav_agent,
        hunch_llm=hunch_llm,
        learning_rate=args.learning_rate,
        steps_in_episode=args.num_rollout_steps,
        env=env,
        start_epoch=args.start_epoch,
        train_data=train_df,
        dev_df=dev_df,
        mbatches_b4_eval=args.batches_b4_eval,
        verbose=args.verbose,
        visualize=args.visualize,
        question_tokenizer=question_tokenizer,
        answer_tokenizer=answer_tokenizer,
        track_gradients=args.track_gradients,
        num_batches_till_eval=args.num_batches_till_eval,
        wandb_on=args.wandb,
    )
    logger.info("Done with everything. Exiting...")

    # TODO: Evaluation of the model
    # metrics = inference(lf)


if __name__ == "__main__":
    main(),
