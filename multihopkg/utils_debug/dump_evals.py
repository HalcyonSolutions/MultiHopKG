import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter 
from transformers import PreTrainedTokenizer

import wandb
import logging
import sys

from multihopkg.vector_search import ANN_IndexMan, ANN_IndexMan_pRotatE

from typing import Dict, Any, Union, Callable

def dump_evaluation_metrics(
    path_to_log: str,
    evaluation_metrics_dictionary: Dict[str, Any],
	vector_entity_searcher: Union[ANN_IndexMan, ANN_IndexMan_pRotatE],						  
    vector_rel_searcher: Union[ANN_IndexMan, ANN_IndexMan_pRotatE],  
    question_tokenizer: PreTrainedTokenizer,
    answer_tokenizer: PreTrainedTokenizer,
    answer_kge_tensor:torch.Tensor,
    id2entity: Dict[int, str],
    id2relations: Dict[int, str],
    entity2title: Dict[str, str],
    relation2title: Dict[str, str],
    kg_model_name: str,
    kg_ent_distance_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    kg_rel_denormalize_func: Callable[[torch.Tensor], torch.Tensor],
    kg_rel_wrap_func: Callable[[torch.Tensor], torch.Tensor],
    iteration: int,
    wandb_on: bool,
    logger: logging.Logger,
    writer: SummaryWriter,
) -> None:
    """
    Will output all of the metrics in a very detailed way for each specific key, ONLY for both the LLM + KG Navigation Agent
    """

    # TODO: Here we are missing dic keys: sampled_actions, position_ids, hunch_llm_final_guesses
    assertions = [
        "sampled_actions" in evaluation_metrics_dictionary,
        "position_ids" in evaluation_metrics_dictionary,
        "reference_questions" in evaluation_metrics_dictionary,
        "true_answer" in evaluation_metrics_dictionary,
        "relevant_entities" in evaluation_metrics_dictionary,
        "relevant_relations" in evaluation_metrics_dictionary,
        "true_answer_id" in evaluation_metrics_dictionary, 
        "kge_cur_pos" in evaluation_metrics_dictionary,
        "kge_prev_pos" in evaluation_metrics_dictionary,
        "kge_action" in evaluation_metrics_dictionary,
        "hunch_llm_entropy" in evaluation_metrics_dictionary,
        "hunch_llm_rewards" in evaluation_metrics_dictionary,
        "hunch_llm_final_guesses" in evaluation_metrics_dictionary,
        "kg_extrinsic_rewards" in evaluation_metrics_dictionary,
        "kg_intrinsic_reward" in evaluation_metrics_dictionary,
        "kg_dones" in evaluation_metrics_dictionary,
        "pg_loss" in evaluation_metrics_dictionary,
    ]
    if not all(assertions):
        raise ValueError("Evaluation metrics dictionary is missing some keys")

    if kg_model_name == "pRotatE":
        unit_type = "(deg)"
        translation_type = "Rotation"
        conversion_constant = 180/torch.pi # convert radians to degrees
    else:
        unit_type = ""
        translation_type = "Translation"
        conversion_constant = 1

    log_file = open(path_to_log, "w")

    batch_size = evaluation_metrics_dictionary["sampled_actions"].shape[1]
    num_reasoning_steps = evaluation_metrics_dictionary["hunch_llm_final_guesses"].shape[0]
    reasoning_steps_str = [ f"{i}." for i in range(num_reasoning_steps)]
    log_file.write(f"Batch size: {batch_size}\n")

    wandb_questions = []
    wandb_predAnswer = []
    wandb_realAnswer = []
    wandb_steps = []
    wandb_positions = []
    distance_between_position_avg = []
    distance_to_answer_avg = []
    with open(path_to_log) as f:
        for element_id in range(batch_size):

            log_file.write(f"********************************\n")
            log_file.write(f"Evaluation for {element_id}\n")

            # Extracting the data from the dictionary
            kge_cur_pos = evaluation_metrics_dictionary["kge_cur_pos"][:, element_id]
            kge_prev_pos = evaluation_metrics_dictionary["kge_prev_pos"][:, element_id]
            kge_action = evaluation_metrics_dictionary["kge_action"][:, element_id]
            kge_extrinsic_rewards = evaluation_metrics_dictionary["kg_extrinsic_rewards"][:, element_id].squeeze(1)
            kge_intrinsic_rewards = evaluation_metrics_dictionary["kg_intrinsic_reward"][:, element_id].squeeze(1)
            kge_dones = evaluation_metrics_dictionary["kg_dones"][:, element_id].squeeze(1)
            hunch_llm_final_guesses = evaluation_metrics_dictionary["hunch_llm_final_guesses"][:, element_id, :]
            questions = evaluation_metrics_dictionary["reference_questions"].iloc[element_id]
            answer = evaluation_metrics_dictionary["true_answer"].iloc[element_id]
            relevant_entities = evaluation_metrics_dictionary["relevant_entities"].iloc[element_id]
            relevant_rels = evaluation_metrics_dictionary["relevant_relations"].iloc[element_id]
            answer_id = evaluation_metrics_dictionary["true_answer_id"].iloc[element_id]
            ans_emb = answer_kge_tensor[element_id,0].cpu()

            #-------------------------------------
            'LLM'
            # Reconstruct the language output
            # for every element in the batch, we decode the 3 actions with 4 elements in them
            predicted_answer = answer_tokenizer.batch_decode(hunch_llm_final_guesses)

            questions_txt = question_tokenizer.decode(questions)
            answer_txt = answer_tokenizer.decode(answer)
            log_file.write(f"#LLM Evaluation Data ------------\n")
            log_file.write(f"Question: {questions_txt}\n")
            log_file.write(f"Answer: {answer_txt}\n")
            
            log_file.write(f"#LLM Predictions ----------------\n")
            log_file.write(f"HunchLLM Answer: {predicted_answer}\n")

            #--------------------------------
            'KGE'

            # Match the relation that are closest to positions we visit
            _, relation_indices = vector_rel_searcher.search(kge_action, 1)
            entity_emb, entity_indices = vector_entity_searcher.search(kge_cur_pos, 1)
            prev_emb, start_index = vector_entity_searcher.search(kge_prev_pos, 1)
            # answer_emb, answer_indices = vector_entity_searcher.search(answer_tensor.squeeze(1).cpu().numpy(), 1)

            # combine index of start_index with the rest of entity_indices into pos_ids
            pos_ids = np.concatenate((start_index[0][:, None], entity_indices), axis=0)

            # -----------------------------------
            'KGE Context Tokens'
            log_file.write(f"#KGE Evaluation Data ------------\n")
            log_file.write(f"Answer ID: {answer_id}\n")
            answer_token = id2entity[answer_id]
            log_file.write(f"Answer Entity Token: {answer_token}\n") # This must match with the answer above
            if entity2title:
                answer_name = entity2title[answer_token]
                log_file.write(f"Answer Entity Name: {answer_name}\n")

            relevant_entities_tokens = [id2entity[index] for index in relevant_entities]
            log_file.write(f"Relevant Entity Tokens: \n{relevant_entities_tokens}\n")

            if entity2title:
                entities_names = [entity2title[index] for index in relevant_entities_tokens]
                log_file.write(f"Relevant Entity Names: \n{entities_names}\n")
                wandb_steps.append(" , ".join(entities_names))

            relevant_relations_tokens = [id2relations[int(index)] for index in relevant_rels]
            log_file.write(f"Relevant Relations Tokens: \n{relevant_relations_tokens}\n")

            if relation2title: 
                relations_names = [relation2title[index] for index in relevant_relations_tokens]
                log_file.write(f"Relevant Relations Names: \n{relations_names}\n")
                wandb_steps.append(" -- ".join(relations_names))

            # -----------------------------------
            'KGE Navigation Agent Tokens'

            log_file.write(f"#NAV Agent Inference ------------\n")
            relations_tokens = [id2relations[int(index)] for index in relation_indices.squeeze()]
            log_file.write(f"Closest Relations Tokens: \n{relations_tokens}\n")

            if relation2title: 
                relations_names = [relation2title[index] for index in relations_tokens]
                log_file.write(f"Closest Relations Names: \n{relations_names}\n")
                wandb_steps.append(" -- ".join(relations_names))

            entities_tokens = [id2entity[index] for index in pos_ids.squeeze()]
            log_file.write(f"Closest Entity Tokens: \n{entities_tokens}\n")

            if entity2title: 
                entities_names = [entity2title[index] for index in entities_tokens]
                log_file.write(f"Closest Entity Names: \n{entities_names}\n")
                wandb_steps.append(" --> ".join(entities_names))

            # --------------------------------
            'KGE Extrinsic and Intrinsic Rewards'
            log_file.write(f"#NAV Agent Rewards --------------\n")
            log_file.write(f"Extrinsic Rewards: \n{kge_extrinsic_rewards.tolist()}\n")
            log_file.write(f"Intrinsic Rewards: \n{(-kge_intrinsic_rewards).tolist()}\n")
            log_file.write(f"Dones: \n{kge_dones.tolist()}\n")

            # -----------------------------------
            'Navigation Agent Distance Metrics'

            # TODO: Make sure this is usuable for both pRotatE and TransE, currently only for pRotatE

            action_distance = []
            for i0 in range(kge_action.shape[0]):
                action_normalized = kg_rel_wrap_func(kg_rel_denormalize_func(kge_action[i0])).abs().sum().item()
                action_distance.append(f"{conversion_constant*action_normalized:.2e}") # calculates how much translation was done
            
            log_file.write(f"Cummulative Action {translation_type} {unit_type}:\n {action_distance} \n")

            position_distance = []
            position_distance_avg = []
            position_distance_ans = []
            closest_emb_distance = []
            
            for i0 in range(kge_cur_pos.shape[0]):
                diff = kg_ent_distance_func(kge_cur_pos[i0], kge_prev_pos[i0])
                diff_total = conversion_constant*diff.sum().item()
                diff_avg = conversion_constant*diff.mean().item()

                diff_ans = kg_ent_distance_func(kge_cur_pos[i0], ans_emb)
                diff_ans = conversion_constant*diff_ans.mean().item()

                diff_closest = kg_ent_distance_func(kge_cur_pos[i0], entity_emb[i0])
                diff_closest = conversion_constant*diff_closest.mean().item()

                position_distance.append(f"{diff_total:.2e}")
                position_distance_avg.append(f"{diff_avg:.2e}")
                position_distance_ans.append(f"{diff_ans:.2e}")
                closest_emb_distance.append(f"{diff_closest:.2e}")
                distance_between_position_avg.append(diff_avg)
                distance_to_answer_avg.append(diff_ans)

            log_file.write(f"Cummulative {translation_type} between KGE Positions {unit_type}:\n {position_distance} \n") # This results should match action distance for pRotatE, otherwise something is wrong, sort of a sanity check
            log_file.write(f"Avg. {translation_type} between KGE Positions {unit_type}:\n {position_distance_avg} \n")
            log_file.write(f"Avg. {translation_type} Debt between Current KGE and Answer {unit_type}:\n {position_distance_ans} \n")
            log_file.write(f"Avg. {translation_type} Debt between Current KGE Pos. and Closest Entity {unit_type}:\n {closest_emb_distance} \n")

            wandb_positions.append(" --> ".join(position_distance))

            # We will write predicted_answer into a wandb table:
            if wandb_on:
                wandb_questions.append(questions_txt)
                wandb_realAnswer.append(answer_txt)
                wandb_predAnswer.append("\n".join([f"{step} {answer}" for step,answer in zip(reasoning_steps_str,predicted_answer)]))
    
    ########################################
    # Report to SummaryWriter
    ########################################
    'LLM Rewards'
    hunch_llm_entropy = evaluation_metrics_dictionary["hunch_llm_entropy"]
    llm_rewards = evaluation_metrics_dictionary["hunch_llm_rewards"]

    writer.add_scalar('LLM Reward/Entropy', hunch_llm_entropy[-1].item(), iteration) # Only consider the last step
    writer.add_scalar('LLM Reward/Reward', llm_rewards[-1].mean(), iteration) # Only consider the last step

    'KG Rewards'
    kge_extrinsic_rewards = evaluation_metrics_dictionary["kg_extrinsic_rewards"]
    kge_intrinsic_rewards = evaluation_metrics_dictionary["kg_intrinsic_reward"]
    kge_dones = evaluation_metrics_dictionary["kg_dones"]

    writer.add_scalar('KG Reward/Extrinsic', kge_extrinsic_rewards[-1].mean(), iteration) # Only consider the last step
    writer.add_scalar('KG Reward/Intrinsic', -kge_intrinsic_rewards[-1].mean(), iteration) # Only consider the last step
    writer.add_scalar('KG Reward/Dones', kge_dones[-1].sum()/len(kge_dones[-1]), iteration) # Only consider the last step

    'Loss'
    pg_loss = evaluation_metrics_dictionary["pg_loss"]
    writer.add_scalar('PG Loss', pg_loss[:,-1].mean(), iteration) # Only consider the last step

    'Distance Metrics'
    distance_between_position_avg = sum(distance_between_position_avg)/len(distance_between_position_avg)
    distance_to_answer_avg = sum(distance_to_answer_avg)/len(distance_to_answer_avg)
    writer.add_scalar('Distance/Position', distance_between_position_avg, iteration)
    writer.add_scalar('Distance/Answer', distance_to_answer_avg, iteration)
    ########################################
    # Report to Wandb
    ########################################
    # TODO: Add KGE navigation agent metrics to wandb
    if wandb_on:
        wandb_df = pd.DataFrame({
            "questions" : wandb_questions,
            "answer_true" : wandb_realAnswer,
            "answer_pred" : wandb_predAnswer,
            "positions" : wandb_positions if wandb_positions else "",
            "relations" : wandb_steps if wandb_positions else ""
        })
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({"qa_results": wandb_table})

        # wandb.log({"entropy": wandb.Histogram(evaluation_metrics_dictionary["hunch_llm_entropy"])})
        wandb.log({"hunchllm/entropy": evaluation_metrics_dictionary["hunch_llm_entropy"].mean().item()})

    # TODO: Some other  metrics ?