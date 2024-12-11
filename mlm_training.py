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

import numpy as np
import pandas as pd
import torch
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
from multihopkg.knowledge_graph import ITLKnowledgeGraph
from multihopkg.language_models import HunchLLM, collate_token_ids_batch
from multihopkg.logging import setup_logger
from multihopkg.rl.graph_search.cpg import ContinuousPolicyGradient
from multihopkg.rl.graph_search.pn import ITLGraphEnvironment
from multihopkg.run_configs import alpha
from multihopkg.utils.setup import set_seeds
from multihopkg.vector_search import ANN_IndexMan

traceback.install()
wandb_run = None

# TODO: Remove before final realease, this is purely for debugging
in_dev_mode = False


def initialize_model_directory(args, random_seed=None):
    # add model parameter info to model directory
    # TODO: We might2ant our implementation of something like this later
    raise NotImplementedError

def initial_setup() -> Tuple[argparse.Namespace, PreTrainedTokenizer, logging.Logger]:
    global logger
    args = alpha.get_args()
    torch.cuda.set_device(args.gpu)
    set_seeds(args.seed)
    logger = setup_logger("__MAIN__")

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    assert isinstance(args, argparse.Namespace)

    return args, tokenizer, logger

def prep_questions(questions: List[torch.Tensor], model: BertModel):
    embedded_questions = model(questions)
    return embedded_questions


def batch_loop_dev(
    env: ITLGraphEnvironment,
    mini_batch: pd.DataFrame,  # Perhaps change this ?
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    steps_in_episode: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Specifically for computing any extra metrics on the dev set.
    Otherwise, this is the same as `batch_loop`.
    """

    ########################################
    # Start the batch loop with zero grad
    ########################################
    nav_agent.zero_grad()

    # Deconstruct the batch
    questions = mini_batch['question'].tolist()
    answers = mini_batch['answer'].tolist()
    question_embeddings = env.get_llm_embeddings(questions)
    answer_ids_padded_tensor = collate_token_ids_batch(answers).to(torch.int32)

    logger.warn(f"About to go into rollout")
    log_probs, rewards, eval_extras = rollout(
        steps_in_episode,
        nav_agent,
        hunch_llm,
        env,
        question_embeddings,
        answer_ids_padded_tensor,
        dev_mode=True,
    )

    ########################################
    # Calculate Reinforce Objective
    ########################################
    logger.warn(f"We just left dev rollout")
    # Compute policy gradient
    rewards_t = torch.stack(rewards).sum(dim=1)
    log_probs_t = torch.stack(log_probs).sum(dim=1)

    assert (
        not torch.isnan(rewards_t).any() and not torch.isnan(log_probs_t).any()
    ), "NaN detected in the rewards or log probs (batch_loop_dev). Aborting training."

    pg_loss = -1 * rewards_t * log_probs_t

    logger.info(f"Does pg_loss require grad? {pg_loss.requires_grad}")

    return pg_loss, eval_extras

def batch_loop(
    env: ITLGraphEnvironment,
    mini_batch: pd.DataFrame,  # Perhaps change this ?
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    steps_in_episode: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    ########################################
    # Start the batch loop with zero grad
    ########################################
    nav_agent.zero_grad()

    # Deconstruct the batch
    questions = mini_batch['question'].tolist()
    answers = mini_batch['answer'].tolist()
    question_embeddings = env.get_llm_embeddings(questions)
    answer_ids_padded_tensor = collate_token_ids_batch(answers).to(torch.int32)

    log_probs, rewards, eval_extras = rollout(
        steps_in_episode,
        nav_agent,
        hunch_llm,
        env,
        question_embeddings,
        answer_ids_padded_tensor,
    )

    ########################################
    # Calculate Reinforce Objective
    ########################################
    # Compute policy gradient
    num_steps = len(log_probs)
    rewards_t = torch.stack(rewards).sum(dim=1)
    log_probs_t = torch.stack(log_probs).sum(dim=1)

    pg_loss = -1*rewards_t * log_probs_t

    return pg_loss, eval_extras


def evaluate_training(
    env: ITLGraphEnvironment,
    dev_df: pd.DataFrame,
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    steps_in_episode: int,
    batch_size: int,
    batch_count: int
):

    global in_dev_mode
    num_batches = len(dev_df) // batch_size
    nav_agent.eval()
    hunch_llm.eval()
    in_dev_mode = True  # TOREM: This is only for debugging
    env.eval()
    # env.question_embedding_module.eval()
    assert (
        not env.question_embedding_module.training
    ), "The question embedding module must not be in training mode"

    metrics = {
        "dev/batch_count": [batch_count],
        "dev/pg_loss": []
    }

    eval_extras = {}

    with torch.no_grad():
        for batch_id in range(num_batches):
            # TODO: Get the rollout working
            mini_batch = dev_df[batch_id*batch_size:(batch_id+1)*batch_size]
            if not isinstance(mini_batch, pd.DataFrame): # For the lsp to give me a break
                raise RuntimeError(f"The mini batch is not a pd.DataFrame, but a {type(mini_batch)}. Please check the data loading code.")
            if len(mini_batch) < batch_size: # We dont want to evaluate on incomplete batches
                continue

            pg_loss,eval_extras = batch_loop_dev(
                env, mini_batch, nav_agent, hunch_llm, steps_in_episode
            )
            logger.info("Finishing inner looop of batch")
            metrics['dev/pg_loss'].append(pg_loss.mean().item())
        logger.info("Done with all batches")

    # Average out all metrics in side metrics
    for k,v in metrics.items():
        metrics[k] = np.mean(v)

    if wandb_run is not None:
        wandb_run.log(metrics)
        if len(eval_extras) > 0:
            ########################################
            # Textual Evaluation 
            ########################################
            table = wandb.Table(columns=["Question","Path Taken","Real Answer","Given Answer"])
            pg_loss = batch_loop_dev(
                env, mini_batch, nav_agent, hunch_llm, steps_in_episode
            )
    




    nav_agent.train()
    hunch_llm.train()
    env.train()
    dev_mode = False
    logger.info("Done with Evaluation")
    # TODO: Implement this

def train_multihopkg(
    batch_size: int,
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
):
    # TODO: Get the rollout working

    # Print Model Parameters + Perhaps some more information
    print(
        "--------------------------\n"
    "Model Parameters\n"
    "--------------------------"
    )
    for name, param in nav_agent.named_parameters():
        print(name, param.numel(), "requires_grad={}".format(param.requires_grad))

    # Just use Adam Optimizer by default
    optimizer = torch.optim.Adam(  # type: ignore
        filter(lambda p: p.requires_grad, nav_agent.parameters()), lr=learning_rate
    )

    # Variable to pass for logging
    batch_count = 0

    ########################################
    # Epoch Loop
    ########################################
    for epoch_id in range(start_epoch, epochs):
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
        for sample_offset_idx in tqdm(range(0, len(train_data), batch_size)):
            ########################################
            # Training
            ########################################
            mini_batch = train_data[sample_offset_idx : sample_offset_idx + batch_size]
            assert isinstance(mini_batch, pd.DataFrame) # For the lsp to give me a break
            optimizer.zero_grad()
            pg_loss,_ = batch_loop(
                 env, mini_batch, nav_agent, hunch_llm, steps_in_episode
            )
            if torch.isnan(pg_loss).any():
                logger.error("NaN detected in the loss. Aborting training.")
                pdb.set_trace()
            reinforce_terms_mean = pg_loss.mean()

            batch_rewards.append(reinforce_terms_mean.item())
            reinforce_terms_mean.backward()
            optimizer.step()

            batch_count += 1 

            ########################################
            # Evaluation 
            ########################################
            logger.warn("Entering evaluation")
            if batch_count % mbatches_b4_eval == 0:
                evaluate_training(env, dev_df, nav_agent, hunch_llm, steps_in_episode, batch_size, batch_count)





def initialize_path(questions: torch.Tensor):
    # Questions must be turned into queries
    raise NotImplementedError


def calculate_reward(hunch_llm: nn.Module, obtained_state: torch.Tensor, answers_ids: torch.Tensor) -> torch.Tensor:
    """
    Will take the answers and give an idea of how close we were.
    This will of course require us to have a language model that will start giving us the  answer.
    """
    batch_size = answers_ids.size(0)
    seq_max_len = answers_ids.size(1)

    # From the obtained_state we will try to find an answer
    answers_inf_softmax = hunch_llm(obtained_state, answers_ids)
    # Get indices of the max value of the final output
    answers_inf_ids = torch.argmax(answers_inf_softmax, dim=-1)
    answers_inf_embeddings = hunch_llm.decoder_embedding(answers_inf_ids.unsqueeze(1)).reshape(batch_size, seq_max_len, -1)
    # attempt_at_answer.shape = (batch_size, seq_len, vocab_size)

    # Compare with the correct answer
    answers_embeddings = hunch_llm.decoder_embedding(answers_ids)
    answer_scores = torch.nn.functional.cosine_similarity(answers_inf_embeddings, answers_embeddings, dim=-1)
    answer_score = answer_scores.mean(-1)

    return answer_score

def rollout(
    # TODO: self.mdl should point to (policy network)
    steps_in_episode,
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    env: ITLGraphEnvironment,
    questions_embeddings: torch.Tensor,
    answers_ids: torch.Tensor,
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
    rewards = []
    dev_dictionary = DefaultDict(list)

    # Dummy nodes ? TODO: Figur eout what they do.
    # TODO: Perhaps here we can enter through the centroid.
    # For now we still with these dummy
    # NOTE: We repeat these entities until we get the right shape:
    # TODO: make sure we keep all seen nodes up to date

    # Get initial observation. A concatenation of centroid and question atm. Passed through the path encoder
    observations = env.reset(questions_embeddings)
    cur_position, cur_state = observations.position, observations.state
    # Should be of shape (batch_size, 1, hidden_dim)

    # pn.initialize_path(kg) # TOREM: Unecessasry to ask pn to form it for us.
    states_so_far = []
    for t in range(steps_in_episode):

        # Ask the navigator to navigate, agent is presented state, not position
        # State is meant to summrized path history.
        sampled_actions, log_probs, entropies  = nav_agent(cur_state)

        # TODO:Make sure we are gettign rewards from the environment.
        observations = env.step(sampled_actions)

        visited_embeddings, states = (observations.position, observations.state)
        # For now, we use states given by the path encoder and positions mostly for debugging
        states_so_far.append(states)

        ########################################
        # Calculate the Reward
        ########################################
        stacked_states = torch.stack(states_so_far).permute(1,0,2)
        similarity_scores = calculate_reward(hunch_llm, stacked_states, answers_ids)
        rewards.append(similarity_scores)

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
            dev_dictionary["sampled_actions"].append(sampled_actions)
            dev_dictionary["visited_embeddings"].append(visited_embeddings)
            meep = observations.position

    if dev_mode:
        pdb.set_trace()
    dev_dictionary = { k: torch.stack(v) for k,v in dev_dictionary.items()}
    # dev_dictionary["sampled_actions"] = torch.stack(dev_dictionary["sampled_actions"])
    # dev_dictionary["visited_position"] = torch.stack(dev_dictionary["visited_position"])


    return log_action_probs, rewards, dev_dictionary

def load_qa_data(cached_metadata_path: str, raw_QAData_path, tokenizer_name: str):
    if os.path.exists(cached_metadata_path):
        logger.info(
            f"\033[93m Found cache for the QA data {cached_metadata_path} will load it instead of working on {raw_QAData_path}. \033[0m"
        )
        # Read the first line of the raw csv to count the number of columns
        train_metadata = json.load(open(cached_metadata_path))
        cached_csv_data_path = train_metadata["saved_path"]
        train_df = pd.read_parquet(cached_csv_data_path)
        # Ensure that we are not reading them integers as strings, but also not as floats
        logger.info(f"Loaded cached data from \033[93m\033[4m{json.dumps(cached_metadata_path,indent=4)} \033[0m")
    else:
        logger.info(
            f"\033[93m Did not find cache for the QA data {cached_metadata_path}. Will now process it from {raw_QAData_path} \033[0m"
        )
        text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        train_df, train_metadata = data_utils.process_qa_data( #TOREM: Same here, might want to remove if not really used
            raw_QAData_path,
            cached_metadata_path,
            text_tokenizer,
        )
        logger.info(f"Done. Result dumped at : \n\033[93m\033[4m{train_metadata['saved_path']}\033[0m")

    ########################################
    # Train Validation Test split
    ########################################

    # Shuffle the Questions
    shuffled_train_df = train_df.sample(frac=1).reset_index(drop=True)
    train_df, val_df = train_test_split(shuffled_train_df, test_size=0.2, random_state=42)

    return train_df, val_df, train_metadata

def main():
    # By default we run the config
    # Process data will determine by itself if there is any data to process
    args, tokenizer, logger = initial_setup()
    global wandb_run

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
    knowledge_graph = ITLKnowledgeGraph(
        data_dir=args.data_dir,
        model=args.model,
        emb_dropout_rate=args.emb_dropout_rate,
        use_action_space_bucketing=args.use_action_space_bucketing,
        pretrained_embedding_type=args.pretrained_embedding_type,
        pretrained_embedding_weights_path=args.pretrained_embedding_weights_path,
    )

    # Information computed by knowldege graph for future dependency injection
    dim_entity = knowledge_graph.get_entity_dim()
    dim_relation = knowledge_graph.get_relation_dim()
    logger.info("You have reached the exit")

    # Paths for triples
    train_triplets_path = os.path.join(args.data_dir, "train.triples")
    dev_triplets_path   = os.path.join(args.data_dir, "dev.triples")
    entity_index_path   = os.path.join(args.data_dir, "entity2id.txt")
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')


    # Get the Module for Approximate Nearest Neighbor Search
    ########################################
    # Setup the ann index. 
    # Will be needed for obtaining observations.
    ########################################
    logger.info(":: Setting up the ANN Index") 
    ann_index_manager = ANN_IndexMan(
        knowledge_graph.get_all_entity_embeddings_wo_dropout(),
        exact_computation=False,
        nlist=100,
    )

    # Setup the pretrained language model
    logger.info(":: Setting up the pretrained language model")
    config = BartConfig.from_pretrained("facebook/bart-base")
    # Access the hidden size (hidden dimension)
    bart_padding_token_id = config.pad_token_id
    # TODO: Remove the hardcode. Perhaps 
    embedding_hidden_size = config.d_model
    embedding_vocab_size = config.vocab_size
    print(f"The hidden dimension of the embedding layer is {embedding_hidden_size} and its vocab size is {embedding_vocab_size}") 
    hunch_llm = HunchLLM(
        pretrained_transformer_weights_path = args.pretrained_llm_transformer_ckpnt_path,
        xattn_left_dim = args.history_dim,
        llm_model_dim = args.llm_model_dim,
        llm_num_heads = args.llm_num_heads,
        llm_num_layers = args.llm_num_layers,
        llm_ff_dim = args.llm_ff_dim,
        llm_max_seq_length = args.max_seq_length,
        xattn_left_max_seq_length = args.steps_in_episode,
        dropout = args.llm_dropout_rate,
        embedding_padding_id = bart_padding_token_id,
        embedding_dim = embedding_hidden_size,
        embedding_vocab_size = embedding_vocab_size,
    )
    if args.further_train_hunchs_llm:
        # TODO: Ensure we dont have to freeze the model for this.
        hunch_llm.freeze_llm()

    # Setup the entity embedding module
    question_embedding_module = AutoModel.from_pretrained(args.question_embedding_model)
    # Setting up the models
    logger.info(":: Setting up the environment")
    env = ITLGraphEnvironment(
        question_embedding_module=question_embedding_module,
        question_embedding_module_trainable=args.question_embedding_module_trainable,
        entity_dim=dim_entity,
        ff_dropout_rate=args.ff_dropout_rate,
        history_dim=args.history_dim,
        history_num_layers=args.history_num_layers,
        knowledge_graph=knowledge_graph,
        relation_dim=dim_relation,
        ann_index_manager=ann_index_manager,
        steps_in_episode=args.num_rollout_steps
    )

    # Now we load this from the embedding models

    # TODO: Reorganizew the parameters lol
    logger.info(":: Setting up the navigation agent")
    nav_agent = ContinuousPolicyGradient(
        baseline=args.baseline,
        beta=args.beta,
        gamma=args.gamma,
        action_dropout_rate=args.action_dropout_rate,
        action_dropout_anneal_factor=args.action_dropout_anneal_factor,
        action_dropout_anneal_interval=args.action_dropout_anneal_interval,
        num_rollout_steps=args.num_rollout_steps,
        dim_action=dim_relation,
        dim_hidden=args.rnn_hidden,
        dim_observation=args.history_dim, # observation will be into history
    )


    # TODO: Add checkpoint support
    # See args.start_epoch

    ########################################
    # Get the data
    ########################################
    logger.info(":: Setting up the data")
    train_df,dev_df, train_metadata = load_qa_data(args.cached_QAMetaData_path, args.raw_QAData_path, args.tokenizer_name)
    if not isinstance(dev_df, pd.DataFrame) or not isinstance(train_df, pd.DataFrame):
        raise RuntimeError("The data was not loaded properly. Please check the data loading code.")

    # TODO: Load the validation data
    # dev_path = os.path.join(args.data_dir, "dev.triples")
    data_triple_split_dict: Dict[str, Any] = data_utils.load_triples_and_dict(
        [train_triplets_path, dev_triplets_path], # TODO: Load the test_data
        entity_index_path,
        relation_index_path,
        group_examples_by_query=False,
        add_reverse_relations=False,
    )

    # TODO: Make it take check for a checkpoint and decide what start_epoch
    # if args.checkpoint_path is not None:
    #     # TODO: Add it here to load the checkpoint separetely
    #     nav_agent.load_checkpoint(args.checkpoint_path)

    ######## ######## ########
    # Train:
    ######## ######## ########
    start_epoch = 0
    logger.info(":: Training the model")
    train_multihopkg(
        batch_size = args.batch_size,
        epochs = args.epochs,
        nav_agent = nav_agent,
        hunch_llm = hunch_llm,
        learning_rate = args.learning_rate,
        steps_in_episode = args.num_rollout_steps,
        env = env, 
        start_epoch = args.start_epoch,
        train_data = train_df,
        dev_df = dev_df,
        mbatches_b4_eval = args.batches_b4_eval,
    )
    logger.info("Done with everything. Exiting...")

    # TODO: Evaluation of the model
    # metrics = inference(lf)


if __name__ == "__main__":
    main(),
