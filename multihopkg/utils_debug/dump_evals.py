import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter 
from transformers import PreTrainedTokenizer
from PIL import Image

import wandb
import logging
import io
import time
import sys

from multihopkg.vector_search import ANN_IndexMan

from typing import Dict, Any, List

# for visualization
initial_pos_flag = True
frame_count = 1
ax = None
fig = None

def dump_evaluation_metrics(
    path_to_log: str,
    evaluation_metrics_dictionary: Dict[str, Any],
    possible_relation_embeddings: torch.Tensor,
	vector_entity_searcher: ANN_IndexMan,								  
    vector_rel_searcher: ANN_IndexMan,
    question_tokenizer: PreTrainedTokenizer,
    answer_tokenizer: PreTrainedTokenizer,
    answer_tensor:torch.Tensor,
    id2entity: Dict[int, str],
    id2relations: Dict[int, str],
    entity2title: Dict[str, str],
    relation2title: Dict[str, str],
    trained_pca,
    graph_pca,
    graph_annotation: List[str],
    visualize: bool,
    wandb_on: bool,
    logger: logging.Logger,
    writer: SummaryWriter,
) -> None:
    """
    Will output all of the metrics in a very detailed way for each specific key, ONLY for both the LLM + KG Navigation Agent
    """
    
    global initial_pos_flag
    global frame_count

    # TODO: Here we are missing dic keys: sampled_actions, position_ids, hunch_llm_final_guesses
    assertions = [
        "sampled_actions" in evaluation_metrics_dictionary,
        "position_ids" in evaluation_metrics_dictionary,
        "hunch_llm_final_guesses" in evaluation_metrics_dictionary,
    ]
    if not all(assertions):
        raise ValueError("Evaluation metrics dictionary is missing some keys")

    log_file = open(path_to_log, "w")

    batch_size = evaluation_metrics_dictionary["sampled_actions"].shape[1]
    num_reasoning_steps = evaluation_metrics_dictionary["hunch_llm_final_guesses"].shape[0]
    reasoning_steps_str = [ f"{i}." for i in range(num_reasoning_steps)]
    log_file.write(f"Batch size: {batch_size}\n")

    final_str_output = ""

    final_columns = ["questions", "answer_true", "answ"]
    wandb_questions = []
    wandb_predAnswer = []
    wandb_realAnswer = []
    wandb_steps = []
    wandb_positions = []
    with open(path_to_log) as f:
        for element_id in range(batch_size):

            log_file.write(f"********************************\n")
            log_file.write(f"Evaluation for {element_id}\n")

            # The main values to work with
            sampled_actions = evaluation_metrics_dictionary["sampled_actions"][:, element_id]
            position_ids = evaluation_metrics_dictionary["position_ids"][:, element_id]
            kge_cur_pos = evaluation_metrics_dictionary["kge_cur_pos"][:, element_id]
            kge_prev_pos = evaluation_metrics_dictionary["kge_prev_pos"][:, element_id]
            kge_action = evaluation_metrics_dictionary["kge_action"][:, element_id]
            hunch_llm_final_guesses = evaluation_metrics_dictionary["hunch_llm_final_guesses"][:, element_id, :]
            questions = evaluation_metrics_dictionary["reference_questions"].iloc[element_id]
            answer = evaluation_metrics_dictionary["true_answer"].iloc[element_id]
            relevant_entities = evaluation_metrics_dictionary["relevant_entities"].iloc[element_id]
            relevant_rels = evaluation_metrics_dictionary["relevant_relations"].iloc[element_id]
            answer_id = evaluation_metrics_dictionary["true_answer_id"].iloc[element_id]

            # Get the pca of the initial position
            if visualize and initial_pos_flag:
                initial_pos = kge_prev_pos[0][None, :]
                initial_pos_mag = np.abs(initial_pos[:, :1000].cpu().detach().numpy() + 1j*initial_pos[:, 1000:].cpu().detach().numpy())
                initial_pos_pca = trained_pca.transform(initial_pos_mag)
                initial_pos_flag = False

            # Reconstruct the language output
            # for every element in the batch, we decode the 3 actions with 4 elements in them
            predicted_answer = answer_tokenizer.batch_decode(hunch_llm_final_guesses)

            questions_txt = question_tokenizer.decode(questions)
            answer_txt = answer_tokenizer.decode(answer)
            log_file.write(f"Question: {questions_txt}\n")
            log_file.write(f"Answer: {answer_txt}\n")
            log_file.write(f"HunchLLM Answer: {predicted_answer}\n")

            # Match the relation that are closest to positions we visit
            _, relation_indices = vector_rel_searcher.search(kge_action, 1)
            entity_emb, entity_indices = vector_entity_searcher.search(kge_cur_pos, 1)
            prev_emb, start_index = vector_entity_searcher.search(kge_prev_pos.detach().cpu(), 1)

            # combine index of start_index with the rest of entity_indices into pos_ids
            pos_ids = np.concatenate((start_index[0][:, None], entity_indices), axis=0)

            # matched_vectors, relation_indices = vector_searcher.search(
            #     possible_relation_embeddings.detach().numpy(),
            #     sampled_actions.detach().numpy(),
            # )

            # -----------------------------------
            'Context Tokens'
            relevant_entities_tokens = [id2entity[index] for index in relevant_entities]

            log_file.write(f"Relevant Entity Tokens: {relevant_entities_tokens}\n")

            if entity2title:
                entities_names = [entity2title[index] for index in relevant_entities_tokens]
                log_file.write(f"Relevant Entity Names: {entities_names}\n")
                wandb_steps.append(" , ".join(entities_names))

            relevant_relations_tokens = [id2relations[int(index)] for index in relevant_rels]
            log_file.write(f"Relevant Relations Tokens: {relevant_relations_tokens}\n")
            if relation2title: 
                relations_names = [relation2title[index] for index in relevant_relations_tokens]
                log_file.write(f"Relevant Relations Names: {relations_names}\n")
                wandb_steps.append(" -- ".join(relations_names))

            # -----------------------------------
            'Navigation Agent Tokens'

            relations_tokens = [id2relations[int(index)] for index in relation_indices.squeeze()]
            log_file.write(f"Relations Tokens: {relations_tokens}\n")

            if relation2title: 
                relations_names = [relation2title[index] for index in relations_tokens]
                log_file.write(f"Relations Names: {relations_names}\n")
                wandb_steps.append(" -- ".join(relations_names))

            action_distance = []
            for i0 in range(kge_action.shape[0] -1 ):
                action_distance.append(f"{torch.dist(kge_action.detach().cpu()[i0], torch.tensor(kge_action[i0+1]).cpu()).item():.2e}")
            
            log_file.write(f"Distance between KGE Actions: {action_distance} \n")

            entities_tokens = [id2entity[index] for index in pos_ids.squeeze()]
            log_file.write(f"Entity Tokens: {entities_tokens}\n")

            if entity2title:
                entities_names = [entity2title[index] for index in entities_tokens]
                log_file.write(f"Entity Names: {entities_names}\n")
                wandb_steps.append(" --> ".join(entities_names))

            position_distance = []
            for i0 in range(kge_cur_pos.shape[0]):
                position_distance.append(f"{torch.dist(kge_prev_pos[i0], kge_cur_pos[i0]).item():.2e}")
                logger.info(f"Eucledian Distance between kge_pre_pos[i0] and kge_cur_pos[i0]: {torch.dist(kge_prev_pos[i0], kge_cur_pos[i0]).item():.2e}")

                # Luis:
                # Divide the entities into imaginary and real parts
                re_prev, im_prev = torch.chunk(kge_prev_pos[i0], 2)
                re_cur, im_cur = torch.chunk(kge_cur_pos[i0], 2)

                # Calculate the distance between them 
                phase_prev = torch.angle(torch.complex(re_prev, im_prev))
                phase_cur = torch.angle(torch.complex(re_cur, im_cur))
                phase_diff = phase_cur - phase_prev

                # Optionally wrap within [-pi, pi] for correct interpretation
                # phase_diff = (phase_diff + torch.pi) % (2 * torch.pi) - torch.pi

                logger.info(f"Phase difference mean between KGE Positions: {phase_diff.mean()} \n")
                logger.info(f"Phase difference p2 between KGE Positions: {torch.norm(phase_diff, p=2).mean()} \n")


            log_file.write(f"Distance between KGE Positions: {position_distance} \n")

            closest_distance = []
            for i0 in range(kge_cur_pos.shape[0]):
                closest_distance.append(f"{torch.dist(kge_cur_pos[i0].cpu(), torch.tensor(entity_emb[i0]).cpu()).item():.2e}")

            log_file.write(f"Distance between KGE Current Positions & Closest Entity: {closest_distance} \n")

            start_distance = f"{torch.dist(kge_prev_pos[0].cpu(), torch.tensor(prev_emb[0]).cpu()).item():.2e}"

            log_file.write(f"Distance between KGE Start Position & Closest Entity: {start_distance} \n")


            # We will write predicted_answer into a wandb table:
            if wandb_on:
                wandb_questions.append(questions_txt)
                wandb_realAnswer.append(answer_txt)
                wandb_predAnswer.append("\n".join([f"{step} {answer}" for step,answer in zip(reasoning_steps_str,predicted_answer)]))
                
    ########################################
    # Report to Wandb
    ########################################
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


    # Save the emebeddings of the current position and the initial position of the agent
    if visualize:
        cur_pos = kge_cur_pos.cpu().numpy()
        cur_pos_mag = np.abs(cur_pos[:, :1000] + 1j*cur_pos[:, 1000:])
        cur_pos_pca = trained_pca.transform(cur_pos_mag)

        closest_entities_mag = np.abs(entity_emb[:, :1000] + 1j*entity_emb[:, 1000:])
        closest_entities_pca = trained_pca.transform(closest_entities_mag)


        write_2d_graph_displacement(data=[graph_pca, cur_pos_pca, initial_pos_pca, closest_entities_pca], 
                                    label=["Graph", "Current Pos", "Initial Pos", "Closest Entities"], 
                                    color=['b', 'g', 'r', 'y'], 
                                    alpha=[0.4, 0.4, 0.8, 0.8], 
                                    marker=['o', 's', '^', 'x'], 
                                    title=f"Visualization of Graph and Positions | Frame {frame_count}\nEvaluation for the last Question", 
                                    xlabel="PCA Component 1", 
                                    ylabel="PCA Component 2",
                                    annotation=graph_annotation,
                                    writer=writer, 
                                    frame_count=frame_count)

        visualization_flag = False

        initial_pos_flag = True
        frame_count += 1

    # TODO: Some other  metrics ?

def write_2d_graph_displacement(data: List[np.ndarray], label: List[str], color: List[str], alpha: List[float], marker: List[str],
                              title: str, xlabel: str, ylabel: str, writer: SummaryWriter, frame_count: int, annotation: List[str]):
        global ax, fig

        ax.clear()
        
        # Plot the various data points
        for i0 in range(len(data)):
            ax.scatter(data[i0][:,0], data[i0][:,1], c=color[i0], label=label[i0], alpha=alpha[i0], marker=marker[i0])

        if annotation:
            for i0, txt in enumerate(annotation):
                ax.annotate(txt, (data[0][i0, 0], data[0][i0, 1]))

        # Set the title and labels
        ax.set_title(f"{title}")
        ax.set_xlabel(f"{xlabel}")
        ax.set_ylabel(f"{ylabel}")

        # Show the grid
        ax.grid(True)

        ax.legend()

        # Save the plot to a BytesIO buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        # Convert the buffer content to an image and then to a NumPy array in HWC format
        image = Image.open(buf)
        image_np = np.array(image)

        # Add the image to TensorBoard
        writer.add_image("Visualization of Graph and Positions", image_np, frame_count, dataformats='HWC')

        # Close the buffer
        buf.close()

        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(0.1)