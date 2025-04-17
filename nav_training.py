"""
This script serves as a counterpart to the `mlm_training.py` file.

While `mlm_training.py` uses masked language modeling with three different embedding spaces 
(Question Textual Embedding with BERT, KG Embedding with a KGE model, and Answer Embedding with BART) 
to answer questions, this script focuses solely on using the KGE model for question answering.

Optionally, it may also incorporate Question Textual Embedding with BERT alongside the KG Embedding 
from the KGE model.
"""
# TODO: Make the question embeddings optional
# TODO: Update Summaries according to this new code (without LLM)
# TODO: Cleanup packages imports
# TODO: Improve dump_eval_metrics to support both `mlm_training.py` and `nav_training.py`

import argparse
import json
import logging
import os
from typing import List, Tuple, Dict, Any, DefaultDict
import debugpy
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter 

import wandb
from rich import traceback
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
import multihopkg.utils_debug.distribution_tracker as dist_tracker
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
from multihopkg.utils_debug.dump_evals import dump_evaluation_metrics


# PCA
from sklearn.decomposition import PCA

import io
from PIL import Image

traceback.install()
wandb_run = None

def initial_setup() -> Tuple[argparse.Namespace, PreTrainedTokenizer, PreTrainedTokenizer, logging.Logger]:
    global logger
    args = alpha.get_args()
    args = overload_parse_defaults_with_yaml(args.preferred_config, args)

    set_seeds(args.seed)
    logger = setup_logger("__NAV__")

    # Get Tokenizer
    question_tokenizer = AutoTokenizer.from_pretrained(args.question_tokenizer_name)

    # TODO: Remove from Navigation code
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_tokenizer_name)

    assert isinstance(args, argparse.Namespace)

    return args, question_tokenizer, answer_tokenizer, logger

def rollout(
    # TODO: self.mdl should point to (policy network)
    steps_in_episode: int,
    nav_agent: ContinuousPolicyGradient,
    env: ITLGraphEnvironment,
    questions_embeddings: torch.Tensor,
    relevant_entities: List[List[int]],
    relevant_rels: List[List[int]],
    answer_id: List[int],
    dev_mode: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
    """
    Executes reinforcement learning (RL) episode rollouts in parallel for a given number of steps.
    This function is the core of the training process, used by both `batch_loop` and `batch_loop_dev`.
    
    During the rollout:
    - The navigation agent (`nav_agent`) interacts with the environment (`env`) to take actions.
    - Rewards are computed from both the language model (`hunch_llm`) and the knowledge graph environment (KGE).
    - Evaluation metrics are optionally collected in development mode (`dev_mode`).

    args:
        steps_in_episode (int): 
            The number of steps to execute in each episode.
        nav_agent (ContinuousPolicyGradient): 
            The policy network responsible for deciding actions based on the current state.
        hunch_llm (nn.Module): 
            A language model used to compute rewards based on how well the agent's state aligns with the expected answers.
        env (ITLGraphEnvironment): 
            The knowledge graph environment that provides observations, rewards, and state transitions.
        questions_embeddings (torch.Tensor): 
            Pre-embedded representations of the questions to be answered. Shape: (batch_size, embedding_dim).
        answers_ids (torch.Tensor): 
            Tokenized IDs of the correct answers. Shape: (batch_size, sequence_length).
        relevant_entities (List[List[int]]): 
            A list of relevant entities for each question, represented as lists of entity IDs.
        relevant_rels (List[List[int]]): 
            A list of relevant relations for each question, represented as lists of relation IDs.
        answer_id (List[int]): 
            A list of IDs corresponding to the correct answer entities.
        dev_mode (bool, optional): 
            If `True`, additional evaluation metrics are collected for debugging or analysis. Defaults to `False`.
    returns:
        - log_action_probs (List[torch.Tensor]): 
            A list of log probabilities of the actions taken by the navigation agent at each step.
        - llm_rewards (List[torch.Tensor]): 
            A list of rewards computed by the language model for each step.
        - kg_rewards (List[torch.Tensor]): 
            A list of rewards computed by the knowledge graph environment for each step.
        - eval_metrics (Dict[str, Any]): 
            A dictionary of evaluation metrics collected during the rollout (only populated if `dev_mode=True`).
    """

    assert steps_in_episode > 0

    ########################################
    # Prepare lists to be returned
    ########################################
    log_action_probs = []
    kg_rewards = []
    eval_metrics = DefaultDict(list)

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

        # TODO: Make sure we are gettign rewards from the environment.
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

        kg_intrinsic_reward = env.knowledge_graph.absolute_difference(
            observations.kge_cur_pos.unsqueeze(1),
            answer_tensor,
        ).norm(dim=-1)

        # TODO: Ensure the that the model stays within range of answer, otherwise set kg_done back to false so intrinsic reward kicks back in.
        kg_rewards.append(kg_dones*kg_extrinsic_rewards - torch.logical_not(kg_dones)*kg_intrinsic_reward) # Merging positive environment rewards with negative intrinsic ones

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

            'KGE Metrics'
            eval_metrics["kg_extrinsic_rewards"].append(kg_extrinsic_rewards.detach().cpu())
            eval_metrics["kg_intrinsic_reward"].append(kg_intrinsic_reward.detach().cpu())
            eval_metrics["kg_dones"].append(kg_dones.detach().cpu())

    if dev_mode:
        eval_metrics = {k: torch.stack(v) for k, v in eval_metrics.items()}

    # Return Rewards of Rollout as a Tensor
    return log_action_probs, kg_rewards, eval_metrics

def batch_loop_dev(
    env: ITLGraphEnvironment,
    mini_batch: pd.DataFrame,  # Perhaps change this ?
    nav_agent: ContinuousPolicyGradient,
    steps_in_episode: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Executes a batch loop for the development set to compute additional evaluation metrics.
    This function is similar to `batch_loop` but focuses on collecting metrics for debugging
    and analysis during development.

    During the batch loop:
    - The navigation agent (`nav_agent`) interacts with the environment (`env`) to take actions inside `rollout`.
    - Rewards are computed from both the language model (`hunch_llm`) and the knowledge graph environment (KGE).
    - Evaluation metrics are collected for analysis.

    This function is found within `evaluate_training` calls upon `rollout`.

    Args:
        env (ITLGraphEnvironment): 
            The knowledge graph environment that provides observations, rewards, and state transitions.
        mini_batch (pd.DataFrame): 
            A batch of data containing questions, answers, relevant entities, and relations.
        nav_agent (ContinuousPolicyGradient): 
            The policy network responsible for deciding actions based on the current state.
        hunch_llm (nn.Module): 
            A language model used to compute rewards based on how well the agent's state aligns with the expected answers.
        steps_in_episode (int): 
            The number of steps to execute in each episode.
        pad_token_id (int): 
            The token ID used for padding sequences in the answer IDs.

    Returns:
        - `pg_loss` (torch.Tensor): 
            The policy gradient loss computed for the batch.
        - `eval_extras` (Dict[str, Any]): 
            A dictionary containing additional evaluation metrics collected during the batch loop.


    Notes:
        - This function is specifically designed for development and debugging purposes.
        - Rewards are normalized for stability before being used to compute the policy gradient loss.
    """

    ########################################
    # Start the batch loop with zero grad
    ########################################
    nav_agent.zero_grad()
    device = nav_agent.fc1.weight.device

    # Deconstruct the batch
    questions = mini_batch["Question"].tolist()
    relevant_entities = mini_batch["Relevant-Entities"].tolist()
    relevant_rels = mini_batch["Relevant-Relations"].tolist()
    answer_id = mini_batch["Answer-Entity"].tolist()
    question_embeddings = env.get_llm_embeddings(questions, device)

    logger.warning(f"About to go into rollout")
    log_probs, kg_rewards, eval_extras = rollout(
        steps_in_episode,
        nav_agent,
        env,
        question_embeddings,
        relevant_entities = relevant_entities,
        relevant_rels = relevant_rels,
        answer_id = answer_id,
        dev_mode=True,
    )

    ########################################
    # Calculate Reinforce Objective
    ########################################

    log_probs_t = torch.stack(log_probs).T
    num_steps = log_probs_t.shape[-1]

    assert not torch.isnan(log_probs_t).any(), "NaN detected in the log probs (batch_loop_dev). Aborting training."

    #-------------------------------------------------------------------------
    'Knowledge Graph Environment Rewards'

    kg_rewards_t = (
        torch.stack(kg_rewards)
    ).permute(1,0,2) # Correcting to Shape: (batch_size, num_steps, reward_type)
    kg_rewards_t = kg_rewards_t.squeeze(2) # Shape: (batch_size, num_steps)

    assert not torch.isnan(kg_rewards_t).any(), "NaN detected in the kg rewards (batch_loop_dev). Aborting training."

    #-------------------------------------------------------------------------
    'Discount and Merging of Rewards'

    # TODO: Check if a weight is needed for combining the rewards
    gamma = nav_agent.gamma
    discounted_rewards = torch.zeros_like(kg_rewards_t.clone()).to(kg_rewards_t.device) # Shape: (batch_size, num_steps)
    discounted_rewards[:,-1] +=  kg_rewards_t[:,-1]
    for t in reversed(range(num_steps - 1)):
        discounted_rewards[:,t] += gamma * (kg_rewards_t[:,t + 1])

    # Sample-wise normalization of the rewards for stability
    discounted_rewards = (discounted_rewards - discounted_rewards.mean(axis=-1)[:, torch.newaxis]) / (discounted_rewards.std(axis=-1)[:, torch.newaxis] + 1e-8)
    
    #--------------------------------------------------------------------------
    'Loss Calculation'

    pg_loss = -discounted_rewards * log_probs_t # Have to negate it into order to do gradient ascent

    logger.warning(f"We just left dev rollout")

    return pg_loss, eval_extras


def batch_loop(
    env: ITLGraphEnvironment,
    mini_batch: pd.DataFrame,  # Perhaps change this ?
    nav_agent: ContinuousPolicyGradient,
    steps_in_episode: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Executes a batch loop for training the navigation agent and language model.
    This function performs reinforcement learning (RL) rollouts for a batch of data
    and computes the policy gradient loss for training.

    During the batch loop:
    - The navigation agent (`nav_agent`) interacts with the environment (`env`) to take actions inside `rollout`.
    - Rewards are computed from both the language model (`hunch_llm`) and the knowledge graph environment (KGE).
    - The policy gradient loss is calculated based on the rewards and log probabilities of actions.

    This function is found within `train_multihokg` and calls upon `rollout`.

    Args:
        env (ITLGraphEnvironment): 
            The knowledge graph environment that provides observations, rewards, and state transitions.
        mini_batch (pd.DataFrame): 
            A batch of data containing questions, answers, relevant entities, and relations.
        nav_agent (ContinuousPolicyGradient): 
            The policy network responsible for deciding actions based on the current state.
        hunch_llm (nn.Module): 
            A language model used to compute rewards based on how well the agent's state aligns with the expected answers.
        steps_in_episode (int): 
            The number of steps to execute in each episode.
        bos_token_id (int): 
            The token ID representing the beginning of a sequence in the answer IDs.
        eos_token_id (int): 
            The token ID representing the end of a sequence in the answer IDs.
        pad_token_id (int): 
            The token ID used for padding sequences in the answer IDs.

    Returns:
        - `pg_loss` (torch.Tensor): 
            The policy gradient loss computed for the batch.
        - `eval_extras` (Dict[str, Any]): 
            A dictionary containing additional evaluation metrics collected during the batch loop.

    Notes:
        - Rewards are normalized for stability before being used to compute the policy gradient loss.
        - This function is designed for training and does not collect as many metrics as `batch_loop_dev`.
    """

    ########################################
    # Start the batch loop with zero grad
    ########################################
    nav_agent.zero_grad()
    device = nav_agent.fc1.weight.device

    # Deconstruct the batch
    questions = mini_batch["Question"].tolist()
    relevant_entities = mini_batch["Relevant-Entities"].tolist()
    relevant_rels = mini_batch["Relevant-Relations"].tolist()
    answer_id = mini_batch["Answer-Entity"].tolist()
    question_embeddings = env.get_llm_embeddings(questions, device)

    log_probs, kg_rewards, eval_extras = rollout(
        steps_in_episode,
        nav_agent,
        env,
        question_embeddings,
        relevant_entities = relevant_entities,
        relevant_rels = relevant_rels,
        answer_id = answer_id,
    )

    ########################################
    # Calculate Reinforce Objective
    ########################################
    logger.debug("About to calculate rewards")

    log_probs_t = torch.stack(log_probs).T
    num_steps = log_probs_t.shape[-1]

    #-------------------------------------------------------------------------
    'Knowledge Graph Environment Rewards'

    kg_rewards_t = (
        torch.stack(kg_rewards)
    ).permute(1,0,2) # Correcting to Shape: (batch_size, num_steps, reward_type)
    kg_rewards_t = kg_rewards_t.squeeze(2) # Shape: (batch_size, num_steps)

    #-------------------------------------------------------------------------
    'Discount and Merging of Rewards'

    # TODO: Check if a weight is needed for combining the rewards
    gamma = nav_agent.gamma
    discounted_rewards = torch.zeros_like(kg_rewards_t.clone()).to(kg_rewards_t.device) # Shape: (batch_size, num_steps)
    discounted_rewards[:,-1] += kg_rewards_t[:,-1]
    for t in reversed(range(num_steps - 1)):
        discounted_rewards[:,t] += gamma * (kg_rewards_t[:,t + 1])

    # Sample-wise normalization of the rewards for stability
    discounted_rewards = (discounted_rewards - discounted_rewards.mean(axis=-1)[:, torch.newaxis]) / (discounted_rewards.std(axis=-1)[:, torch.newaxis] + 1e-8)


    #--------------------------------------------------------------------------
    'Loss Calculation'

    pg_loss = -discounted_rewards * log_probs_t # Have to negate it into order to do gradient ascent

    return pg_loss, eval_extras

def evaluate_training(
    env: ITLGraphEnvironment,
    dev_df: pd.DataFrame,
    nav_agent: ContinuousPolicyGradient,
    steps_in_episode: int,
    batch_size_dev: int,
    batch_count: int,
    verbose: bool,
    visualize: bool,
    writer: SummaryWriter,
    question_tokenizer: PreTrainedTokenizer,
    wandb_on: bool,
    iteration: int,
    answer_id: List[int] = None,
):
    """
    Evaluates the performance of the navigation agent and language model on the development set.
    This function computes evaluation metrics, logs results, and optionally visualizes the evaluation process.

    This function is found within `train_multihopkg` and is called periodically during training. 
    This function calls upon `batch_loop_dev` and `dump_evaluation_metrics`.

    Args:
        env (ITLGraphEnvironment): 
            The knowledge graph environment that provides observations, rewards, and state transitions.
        dev_df (pd.DataFrame): 
            The development dataset containing questions, answers, relevant entities, and relations.
        nav_agent (ContinuousPolicyGradient): 
            The policy network responsible for deciding actions based on the current state.
        hunch_llm (nn.Module): 
            A language model used to compute rewards based on how well the agent's state aligns with the expected answers.
        steps_in_episode (int): 
            The number of steps to execute in each episode.
        batch_size_dev (int): 
            The batch size for the development set.
        batch_count (int): 
            The current batch count during training.
        verbose (bool): 
            If `True`, additional information is logged for debugging purposes.
        visualize (bool): 
            If `True`, visualizations of the evaluation process are generated.
        writer (SummaryWriter): 
            A TensorBoard writer for logging metrics and visualizations.
        question_tokenizer (PreTrainedTokenizer): 
            The tokenizer used for processing questions.
        answer_tokenizer (PreTrainedTokenizer): 
            The tokenizer used for processing answers.
        wandb_on (bool): 
            If `True`, logs metrics to Weights & Biases (wandb).
        iteration (int): 
            The current iteration number, used for logging and tracking progress.
        answer_id (List[int], optional): 
            A list of IDs corresponding to the correct answer entities. Defaults to `None`.

    Returns:
        None

    Notes:
        - This function evaluates only the last batch of the development set.
        - Metrics are logged to TensorBoard and optionally to wandb.
        - The function ensures that the environment and models are in evaluation mode during the process.
    """
    num_batches = len(dev_df) // batch_size_dev
    nav_agent.eval()

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

        # We will only evaluate on the last batch
        batch_id = num_batches - 1

        mini_batch = dev_df[
            batch_id * batch_size_dev : (batch_id + 1) * batch_size_dev
        ]

        if not isinstance(  # TODO: Remove this assertion once it is never ever met again
            mini_batch, pd.DataFrame
        ):  # For the lsp to give me a break
            raise RuntimeError(
                f"The mini batch is not a pd.DataFrame, but a {type(mini_batch)}. Please check the data loading code."
            )
        
        current_evaluations["reference_questions"] = mini_batch["Question"]
        current_evaluations["true_answer"] = mini_batch["Answer"]
        current_evaluations["relevant_entities"] = mini_batch["Relevant-Entities"]
        current_evaluations["relevant_relations"] = mini_batch["Relevant-Relations"]
        current_evaluations["true_answer_id"] = mini_batch["Answer-Entity"]

        # Get the Metrics
        pg_loss, eval_extras = batch_loop_dev(
            env,
            mini_batch,
            nav_agent,
            steps_in_episode,
        )

        'Extract all the variables from eval_extras'
        for k, v in eval_extras.items():
            current_evaluations[k] = v

        # Accumlate the metrics
        current_evaluations["pg_loss"] = pg_loss.detach().cpu()
        batch_cumulative_metrics["dev/pg_loss"].append(pg_loss.mean().item())

        ########################################
        # Take `current_evaluations` as
        # a sample of batches and dump its results
        ########################################
        if verbose and logger:

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
                answer_tokenizer=None, # TODO: Ensure this an optional parameter
                answer_kge_tensor=answer_kge_tensor,
                id2entity=env.id2entity,					   
                id2relations=env.id2relation,
                entity2title=env.entity2title,
                relation2title=env.relation2title,
                kg_model_name=env.knowledge_graph.model_name,
                kg_ent_distance_func=env.knowledge_graph.absolute_difference,
                kg_rel_denormalize_func=env.knowledge_graph.denormalize_relation,
                kg_rel_wrap_func=env.knowledge_graph.wrap_relation,
                iteration=iteration,
                writer=writer,						  
                wandb_on=wandb_on,
                logger=logger,
            )
            logger.warning(f"We just left dump_evaluation_metrics")

            logger.warning(f"Cleaning up the dev dictionaries")
            
            current_evaluations.clear()
            eval_extras.clear()

            if not mini_batch._is_view: # if a copy was created, delete after usage
                del mini_batch

def train_nav_multihopkg(
    batch_size: int,
    batch_size_dev: int,
    epochs: int,
    nav_agent: ContinuousPolicyGradient,
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
    track_gradients: bool,
    num_batches_till_eval: int,
    wandb_on: bool,
):
    """
    Trains the navigation agent and language model using reinforcement learning (RL) on a knowledge graph environment.
    This function performs training over multiple epochs and evaluates the model periodically on a development set.

    During training:
    - The navigation agent (`nav_agent`) interacts with the environment (`env`) to take actions in `rollout`.
    - Rewards are computed from both the language model (`hunch_llm`) and the knowledge graph environment (KGE) in `batch_loop`.
    - The policy gradient loss is calculated and used to update the model parameters.
    - Evaluation is performed periodically using the `evaluate_training` function.

    This function is found within `main` and is called to initiate the training process.
    This function calls upon `batch_loop` and `evaluate_training`.

    Args:
        batch_size (int): 
            The batch size for training.
        batch_size_dev (int): 
            The batch size for the development set.
        epochs (int): 
            The total number of epochs to train the model.
        nav_agent (ContinuousPolicyGradient): 
            The policy network responsible for deciding actions based on the current state.
        hunch_llm (nn.Module): 
            A language model used to compute rewards based on how well the agent's state aligns with the expected answers.
        learning_rate (float): 
            The learning rate for the optimizer.
        steps_in_episode (int): 
            The number of steps to execute in each episode.
        env (ITLGraphEnvironment): 
            The knowledge graph environment that provides observations, rewards, and state transitions.
        start_epoch (int): 
            The epoch to start training from (useful for resuming training).
        train_data (pd.DataFrame): 
            The training dataset containing questions, answers, relevant entities, and relations.
        dev_df (pd.DataFrame): 
            The development dataset for periodic evaluation.
        mbatches_b4_eval (int): 
            The number of mini-batches to process before performing evaluation.
        verbose (bool): 
            If `True`, additional information is logged for debugging purposes.
        visualize (bool): 
            If `True`, visualizations of gradients and weights histograms are tracked along with navigation movements.
        question_tokenizer (PreTrainedTokenizer): 
            The tokenizer used for processing questions.
        answer_tokenizer (PreTrainedTokenizer): 
            The tokenizer used for processing answers. Note: Not the same as the question tokenizer.
        track_gradients (bool): 
            If `True`, tracks and logs gradient information for debugging.
        num_batches_till_eval (int): 
            The number of batches to process before inspecting vanishing gradients.
        wandb_on (bool): 
            If `True`, logs metrics to Weights & Biases (wandb).

    Returns:
        None

    Notes:
        - The function uses reinforcement learning to train the navigation agent and language model.
        - Periodic evaluation is performed using the `evaluate_training` function.
        - Metrics and visualizations are logged to TensorBoard and optionally to wandb.
        - The function ensures that the environment and models are in training mode during the process.
    """
    # TODO: Get the rollout working

    # Print Model Parameters + Perhaps some more information
    if verbose:
        print(
            "--------------------------\n" "Model Parameters\n" "--------------------------"
        )
        for name, param in nav_agent.named_parameters():
            print(name, param.numel(), "requires_grad={}".format(param.requires_grad))

        for name, param in env.named_parameters():
            if param.requires_grad: print(name, param.numel(), "requires_grad={}".format(param.requires_grad))

    writer = SummaryWriter(log_dir=f'runs')

    named_param_map = {param: name for name, param in (list(nav_agent.named_parameters()) + list(env.named_parameters()))}
    optimizer = torch.optim.Adam(  # type: ignore
        filter(
            lambda p: p.requires_grad,
            list(env.concat_projector.parameters()) + list(nav_agent.parameters())
        ),
        lr=learning_rate
    )

    modules_to_log: List[nn.Module] = [nav_agent]

    # Variable to pass for logging
    batch_count = 0

    # Replacement for the hooks
    if track_gradients:
        grad_logger = torch_module_logging.ModuleSupervisor({
            "navigation_agent" : nav_agent, 
        })

    ########################################
    # Epoch Loop
    ########################################
    for epoch_id in tqdm(range(start_epoch, epochs), desc="Epoch"):

        logger.info("Epoch {}".format(epoch_id))
        # TODO: Perhaps evaluate the epochs?

        # Set in training mode
        nav_agent.train()

        ##############################
        # Batch Loop
        ##############################
        # TODO: update the parameters.
        for sample_offset_idx in tqdm(range(0, len(train_data), batch_size), desc="Training Batches", leave=False):
            mini_batch = train_data[sample_offset_idx : sample_offset_idx + batch_size]

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
                    steps_in_episode,
                    batch_size_dev,
                    batch_count,
                    verbose,
                    visualize,
                    writer,
                    question_tokenizer,
                    wandb_on,
                    iteration = epoch_id * (len(train_data) // batch_size // mbatches_b4_eval) + (batch_count // mbatches_b4_eval),
                    answer_id=mini_batch["Answer-Entity"].tolist(),  # Extract answer_id from mini_batch
                )

            ########################################
            # Training
            ########################################
            'Forward pass'

            optimizer.zero_grad()
            pg_loss, _ = batch_loop(
                env, mini_batch, nav_agent, steps_in_episode
            )

            if torch.isnan(pg_loss).any():
                logger.error("NaN detected in the loss. Aborting training.")

            # Logg the mean, std, min, max of the rewards
            reinforce_terms_mean = pg_loss.mean()
            reinforce_terms_mean_item = reinforce_terms_mean.item()
            reinforce_terms_std_item = pg_loss.std().item()
            reinforce_terms_min_item = pg_loss.min().item()
            reinforce_terms_max_item = pg_loss.max().item()
            logger.debug(f"Reinforce terms mean: {reinforce_terms_mean_item}, std: {reinforce_terms_std_item}, min: {reinforce_terms_min_item}, max: {reinforce_terms_max_item}")

            # TODO: Uncomment and try: (but comment out the normalization in batch_loop and bacth_loop_dev)
            # pg_loss = tensor_normalization(pg_loss)

            #---------------------------------
            'Backward pass'
            logger.debug("Bout to go backwords")
            reinforce_terms_mean.backward()
            
            #---------------------------------
            'Gradient Tracking'

            if sample_offset_idx == 0:

                # Ask for the DAG to be dumped
                if track_gradients:
                    grad_logger.dump_visual_dag(destination_path=f"./figures/grads/dag_{epoch_id:02d}.png", figsize=(10, 100)) # type: ignore

            if torch.all(nav_agent.mu_layer.weight.grad == 0):
                logger.warning("Gradients are zero for mu_layer!")


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

                        dist_tracker.write_dist_parameters(grads, name, "Gradient", writer, epoch_id)
                        dist_tracker.write_dist_parameters(weights, name, "Weights", writer, epoch_id)

                        if wandb_on:
                            wandb.log({f"{name}/Gradient": wandb.Histogram(grads.numpy().flatten())})
                            wandb.log({f"{name}/Weights": wandb.Histogram(weights.numpy().flatten())})
                        elif visualize:
                            dist_tracker.write_dist_histogram(
                                grads.numpy().flatten(),
                                name, 
                                'g', 
                                "Gradient Histogram", 
                                "Grad Value", 
                                "Frequency", 
                                writer, 
                                epoch_id
                            )
                            dist_tracker.write_dist_histogram(
                                weights.numpy().flatten(), 
                                name, 
                                'b', 
                                "Weights Histogram", 
                                "Weight Value", 
                                "Frequency", 
                                writer, 
                                epoch_id
                            )
            
            if wandb_on:
                loss_item = pg_loss.mean().item()
                logger.info(f"Submitting train/pg_loss: {loss_item} to wandb")
                wandb.log({"train/pg_loss": loss_item})

            #---------------------------------
            'Optimizer step'

            optimizer.step()

            batch_count += 1

def main():
    """
    The main entry point for training and evaluating the MultiHopKG model.

    This function orchestrates the entire process, including:
    - Initializing configurations, tokenizers, and logging.
    - Loading and preprocessing the training and development datasets.
    - Setting up the knowledge graph environment and navigation agent.
    - Training the navigation agent and language model using reinforcement learning.
    - Optionally logging metrics and visualizations to Weights & Biases (wandb).

    Workflow:
    1. **Initial Setup**:
       - Parses command-line arguments and configuration files.
       - Initializes tokenizers and logging.
       - Optionally waits for a debugger to attach if in debug mode.

    2. **Data Loading**:
       - Loads the knowledge graph dictionaries (entities and relations).
       - Loads and preprocesses the QA datasets (training and development).

    3. **Environment and Model Setup**:
       - Loads the pretrained knowledge graph embeddings and initializes the KGE model.
       - Sets up the approximate nearest neighbor (ANN) index for entity and relation embeddings.
       - Initializes the pretrained language model (`HunchBart`) and the navigation agent (`ContinuousPolicyGradient`).
       - Configures the ITLGraphEnvironment for reinforcement learning.

    4. **Training**:
       - Calls the `train_multihopkg` function to train the navigation agent and language model.
       - Periodically evaluates the model on the development set using `evaluate_training`.

    5. **Logging and Visualization**:
       - Optionally logs metrics and visualizations to TensorBoard and wandb.
       - Supports visualization of the knowledge graph embeddings.

    Args:
        None

    Returns:
        None

    Notes:
        - The function assumes that all required configurations and paths are provided via command-line arguments or configuration files.
        - The function supports debugging, visualization, and logging for enhanced monitoring and analysis.
    """
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
    # TODO: Modify code so it doesn't require answer_tokenizer
    train_df, dev_df, train_metadata = data_utils.load_qa_data(
        cached_metadata_path=args.cached_QAMetaData_path,
        raw_QAData_path=args.raw_QAData_path,
        question_tokenizer_name=args.question_tokenizer_name,
        answer_tokenizer_name=args.answer_tokenizer_name, 
        entity2id=ent2id,
        relation2id=rel2id,
        logger=logger,
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

    ########################################
    # Set the KG Environment
    ########################################
    # Agent needs a Knowledge graph as well as the environment
    logger.info(":: Setting up the knowledge graph")

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
    # TODO: Improve the ANN index manager for rotational models
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

    # Setup the entity embedding module
    question_embedding_module = AutoModel.from_pretrained(args.question_embedding_model).to(args.device)

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
        trained_pca=None,
        graph_pca=None,
        graph_annotation=None,
        nav_start_emb_type=args.nav_start_emb_type,
        epsilon = args.nav_epsilon_error,
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

    train_nav_multihopkg(
        batch_size=args.batch_size,
        batch_size_dev=args.batch_size_dev,
        epochs=args.epochs,
        nav_agent=nav_agent,
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
        track_gradients=args.track_gradients,
        num_batches_till_eval=args.num_batches_till_eval,
        wandb_on=args.wandb,
    )
    logger.info("Done with everything. Exiting...")

    # TODO: Evaluation of the model
    # metrics = inference(lf)

if __name__ == "__main__":
    main()