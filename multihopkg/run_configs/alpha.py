"""
Instead of using bach config filees, we will just separate them into different files we can just import

This is config `alpha.py`:
Just to try to get the first run running.
"""

import argparse

def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    'General Training Settings'
    ap.add_argument('--device', type=str, default="cuda:0", help="Device to run the model on (e.g., cuda:0 or cpu)")
    ap.add_argument("--seed", type=int, default=420, metavar="S", help="Random seed for reproducibility")
    ap.add_argument("--debug", "-d", action="store_true", help="Enable debug mode with extra logs") # TODO: Check if this is still needed.
    ap.add_argument('--verbose','-v',action="store_true", help="Print, log, and dump evaluation results")
    ap.add_argument('--visualize','-vv',action="store_true", help="Render 2D/3D visualizations of evaluation results in addition to printing and logging")
    ap.add_argument('--track_gradients', '-g', action='store_true', help='Track and log gradients during training')
    ap.add_argument('--preferred_config', type=str, default="configs/my_config.yaml", help="Path to YAML configuration file (default: configs/my_config.yaml). " \
                        "If not empty, overrides the respective command line arguments.")

    'Learning Hyperparameters'
    ap.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate for optimizer (default: 1e-5)')
    ap.add_argument('--beta', type=float, default=0.0, help='Entropy regularization coefficient (default: 0.0)') # TODO: Check if this is still used.
    ap.add_argument('--gamma', type=float, default=12, help='Margin or scaling factor used by the knowledge graph embedding model.' \
                            'Not actually used during flexible translation (default: 12)') # TODO: Force load this value from the state_dict instead. 
    ap.add_argument('--rl_gamma', type=float, default=0.9, help='Discount factor for the reinforcement learning agent (default: 0.9)')
    ap.add_argument('--baseline', type=str, default='n/a', help='Baseline strategy for policy gradient (default: n/a)') # TODO: Check if this is still used. Answer: Passed to NavigationAgent, but not used in the current implementation.
    
    ap.add_argument('--supervised_adapter_scalar', default=0.5, type=float, help='Scalar for the supervised adapter loss (default: 0.0)') 
    ap.add_argument('--supervised_sigma_scalar', default=0.1, type=float, help='Scalar for the supervised sigma loss (default: 0.1)')
    ap.add_argument('--supervised_expected_sigma', default=0.1, type=float, help='Scalar for the supervised expected sigma value (default: 0.1)') 
                    #TODO: Add the warmup parameters here


    'Dropout Scheduling'
    ap.add_argument('--action_dropout_rate', type=float, default=0.1, help='Dropout rate for randomly masking out knowledge graph edges (default: 0.1)') # TODO: Check if this is still used. Answer: Passed to NavigationAgent, but not used in the current implementation.
    ap.add_argument('--action_dropout_anneal_factor', type=float, default=0.95, help='Decrease the action dropout rate once the dev set results stopped increase (default: 0.95)') # TODO: Check if this is still used. Answer: Passed to NavigationAgent, but not used in the current implementation.
    ap.add_argument('--action_dropout_anneal_interval', type=int, default=1000, help='Number of epochs to wait before decreasing the action dropout rate (default: 1000. Action '
                         'dropout annealing is not used when the value is >= 1000.)') # TODO: Check if this is still used. Answer: Passed to NavigationAgent, but not used in the current implementation.
    
    'Training Duration & Rollout'
    ap.add_argument('--epochs',type=int,default=200, help='Total number of training epochs (default: 200)')
    ap.add_argument('--start_epoch', type=int, default=0, help='Epoch to start training instead of just observing (default: 0)') # TODO: Implement this.
    ap.add_argument('--num_rollouts', type=int, default=5, help='Number simultaneous trajectories to sample from during training for each sample question (default: 1)')
    ap.add_argument('--num_rollout_steps', type=int, default=8, help='Maximum number of steps per questions (default: 8)')

    'Batch Settings'
    ap.add_argument('--batch_size', type=int, default=256, help='Training mini-batch size (default: 256)')
    ap.add_argument('--batch_size_dev', type=int, default=64, help='Evaluation mini-batch size (default: 64)')
    ap.add_argument('--batches_b4_eval', type=int, default=100, help='Batches to train before first evaluation phase (default: 100)') #TODO: Remove if unused.
    ap.add_argument('--num_batches_till_eval', type=int, default=15, help='Batches to train between evaluations (default: 15)')

    'Datasets & File Paths'
    # QA Dataset
    ap.add_argument('--raw_QAData_path', type=str, default="./data/FB15k/freebaseqa_clean.csv", help="Path to the raw QA CSV dataset (default: FreebaseQA)")
    ap.add_argument('--cached_QAMetaData_path', type=str, default="./.cache/itl/freebaseqa_clean.json", help="Path to cached tokenized QA metadata JSON file")
    ap.add_argument('--force_data_prepro', '-f', action="store_true", help="Force re-processing of QA data, even if cache exists")
    ap.add_argument('--use_kge_question_embedding', '-kq', action="store_true", help="Use entity and relation embedding as the questions instead of textual question. Only valid for single-hop task. (default: False)")
    #TODO: Add the override_split option here


    # KG Dataset
    ap.add_argument('--data_dir', type=str, default="./data/FB15k", help='Root directory for KG triples and metadata (default: ./data/FB15k)')
    ap.add_argument('--node_data_path', type=str, default='./data/FB15k/node_data.csv', help='CSV path containing entity name mappings. Leave Empty if not applicable.')
    ap.add_argument('--node_data_key', type=str, default='MID', help='Entity key type (e.g., MID for FB15k, QID for Wikidata). Leave Empty if not applicable.')
    ap.add_argument('--relationship_data_path', type=str, default='./data/FB15k/relation_data.csv', help='CSV path containing relationship name mappings. Leave Empty if not applicable.')
    ap.add_argument('--relationship_data_key', type=str, default='Relation', help='Relation key type (e.g., Property for Wikidata, None for FB15k). Leave Empty if not applicable.')

    'Knowledge Graph Embedding Model'
    ap.add_argument('--model', type=str, default='pRotatE', help='Embedding model used for KG representation (default: pRotatE)')
    ap.add_argument('--trained_model_path', type=str, default="./models/protatE_FB15k/", help='Path to pre-trained embedding model directory')
    
    'Navigation Agent Settings'
    ap.add_argument('--nav_start_emb_type', type=str, default="centroid", help="Initial navigation point: 'centroid', 'random', or 'relevant'")
    ap.add_argument('--nav_epsilon_error', type=float, default=50.0, help="Allowable distance to consider the answer as 'reached' (default: 50.0)")
    ap.add_argument('--nav_epsilon_metric', type=str, default="l2", help="Distance metric for navigation: 'l1', 'l2', or 'deg' (default: l2)")
    ap.add_argument('-ts', '--add_transition_state', action='store_true', help="Include the past position, action, and current position into the state (default: False)")

    # RNN Settings
    ap.add_argument('--history_dim', type=int, default=768, metavar='H', help='Hidden size of action history LSTM encoder (default: 768)')
    ap.add_argument('--history_num_layers', type=int, default=3, metavar='L', help='Number of layers in action history LSTM encoder (default: 3)')
    ap.add_argument('--ff_dropout_rate', type=float, default=0.1, help='Dropout rate for feed-forward layers (default: 0.1)')
    ap.add_argument('--rnn_hidden',type=int,default=400, help='Hidden size of general-purpose RNN modules (default: 400)')

    'Textual Embedding (LLMs)'
	# TODO: (eventually) We might want to add option of locally trained models.
    # TODO: Replace the redundant models here (question_tokenizer_name vs question_embedding_model) and (answer_tokenizer_name vs pretrained_llm_for_hunch)
    
    # These are based on Halcyon/FoundationalLanguageModel
    
    # Question Embedding
    ap.add_argument("--question_tokenizer_name", type=str, default="bert-base-uncased", help="Tokenizer name for question embeddings")
    ap.add_argument('--question_embedding_model', type=str, default="bert-base-uncased", help="The Question embedding model to use (default: bert-base-uncased)")
    ap.add_argument('--question_embedding_module_trainable', type=bool, default=True, help="Whether to fine-tune the question embedding model (default: True)")
    ap.add_argument("--llm_model_dim", default=768, help="Dimensionality of the LLM embedding outputs (default: 768)")

    # Answer Embedding
    ap.add_argument("--answer_tokenizer_name", type=str, default="facebook/bart-base", help="Tokenizer name for answer embeddings")
    ap.add_argument('--further_train_hunchs_llm',  action="store_true", help="Enable further pretraining of the answer embedding LLM")
    ap.add_argument('--pretrained_llm_for_hunch', type=str, default="facebook/bart-base", help="Pretrained LLM used to embed answer 'hunches' (default: facebook/bart-base)")

    'Logging and Experiment Tracking'
    ap.add_argument("-w", "--wandb", action="store_true", help="Enable Weights & Biases experiment tracking")
    ap.add_argument("--wandb_project_name", type=str, help="wandb: Project name to group runs")
    ap.add_argument("--wr_name", type=str, help="wandb: Unique name for this run")
    ap.add_argument("--wr_notes", type=str, help="wandb: Additional notes for this run")


    ######################
    #### Legacy Parameters
    ######################
    """
    List of legacy parameters that are not used anymore but might be useful in the future. 
    Remove once confident that they are not needed anymore.
    Commented out for now.
    """

    # # For data processing
    # path_to_running_file = os.path.abspath(sys.argv[0])
    # default_cache_dir = os.path.join(os.path.dirname(path_to_running_file), ".cache")

    # ap.add_argument("--gpu", type=int, default=0)

    # ap.add_argument(
    #     "--QAtriplets_raw_dir",
    #     type=str,
    #     default=os.path.join(path_to_running_file, "data/itl/multihop_ds_datasets_FbWiki_TriviaQA.csv"),
    # )
    # ap.add_argument(
    #     "--QAtriplets_cache_dir",
    #     type=str,
    #     default=os.path.join(default_cache_dir, "qa_triplets.csv"),
    # )

    # ap.add_argument('--graph_embed_model_name', type=str,default="RotatE",
    #                 help='The name of the graph embedding model to use')

        # ap.add_argument('--freebaseqa_path', type=str, default="./data/freebaseqa", help="The path to the freebaseqa data")

    # New Paremters introduced by the new model
    # ap.add_argument("--pretrained_embedding_type",type=str,default="conve",help="The type of pretrained embedding to use")
    # ap.add_argument("--pretrained_embedding_weights_path",type=str,default="./models/itl/pretrained_embeddings.tar",help="Theh path to the pretrained embedding weights")
    # ap.add_argument("--emb_dropout_rate", type=float, default=0.3, help='Knowledge graph embedding dropout rate (default: 0.3)')
    # ap.add_argument("--relation_only",  action="store_true",  help='search with relation information only, ignoring entity representation (default: False)')
    # ap.add_argument('--xavier_initialization', type=bool, default=True,
    #                     help='Initialize all model parameters using xavier initialization (default: True)')

    # ap.add_argument('--exact_nn',  action="store_true", help="Whether to use exact nearest neighbor search or not (default: False)")
    # ap.add_argument('--num_cluster_for_ivf', type=int, default=100, help="Number of clusters for the IVF index if exact_computation is False (default: 100)")

    # ap.add_argument('--pretrained_llm_transformer_ckpnt_path', type=str, default="models/itl/pretrained_transformer_e1_s9176.ckpt", help="The path to the pretrained language model transformer weights (default: models/itl/pretrained_transformer_e1_s9176.ckpt)")
    # ap.add_argument('--pretrained_sun_model_loc', type=str, default="models/sun", help="The path to the pretrained language model transformer weights (default: models/itl/pretrained_transformer_e1_s9176.ckpt)")
    
    # ap.add_argument("--llm_num_heads", default=8)
    # ap.add_argument("--llm_num_layers", default=3)
    # ap.add_argument("--llm_ff_dim", default=3072)
    # ap.add_argument("--llm_ff_dropout_rate", default=0.1)
    # ap.add_argument("--llm_dropout_rate", default=0.1)
    # ap.add_argument("--max_seq_length", default=1024)

    # Might want to get rid of them as we see fit.
    # ap.add_argument('--relation_only_in_path', action='store_true',
    #                     help='include intermediate entities in path (default: False)')
        
    # ap.add_argument('--run_analysis', action='store_true',
    #                 help='run algorithm analysis and print intermediate results (default: False)')
    # ap.add_argument('--model_root_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'),
    #                 help='root directory where the model parameters are stored (default: None)')
    # ap.add_argument('--model_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'),
    #                 help='directory where the model parameters are stored (default: None)')

        # ap.add_argument('--use_action_space_bucketing', action='store_true',
    #                 help='bucket adjacency list by outgoing degree to avoid memory blow-up (default: False)')
    # ap.add_argument('--train_entire_graph', type=bool, default=False,
    #                 help='add all edges in the graph to extend training set (default: False)')
    # ap.add_argument('--num_epochs', type=int, default=200,
    #                 help='maximum number of pass over the entire training set (default: 20)')
    # ap.add_argument('--num_wait_epochs', type=int, default=5,
    #                 help='number of epochs to wait before stopping training if dev set performance drops')
    # ap.add_argument('--num_peek_epochs', type=int, default=2,
    #                 help='number of epochs to wait for next dev set result check (default: 2)')

    # ap.add_argument('--train_batch_size', type=int, default=256,
    #                 help='mini-batch size during training (default: 256)')

    # ap.add_argument('--learning_rate_decay', type=float, default=1.0,
    #                 help='learning rate decay factor for the Adam optimizer (default: 1)')
    # ap.add_argument('--adam_beta1', type=float, default=0.9,
    #                 help='Adam: decay rates for the first movement estimate (default: 0.9)')
    # ap.add_argument('--adam_beta2', type=float, default=0.999,
    #                 help='Adam: decay rates for the second raw movement estimate (default: 0.999)')
    # ap.add_argument('--grad_norm', type=float, default=10000,
    #                 help='norm threshold for gradient clipping (default 10000)')

    # ap.add_argument('--steps_in_episode', type=int, default=20,
    #                 help='number of steps in episode (default: 20)')

    # ap.add_argument('--gamma', type=float, default=1,
    #                 help='moving average weight (default: 1)')
    # ap.add_argument('--beam_size', type=int, default=100,
    #                 help='size of beam used in beam search inference (default: 100)')
    # ap.add_argument('--num_epochs_till_eval', type=int, default=100,
    #                 help='Number of epochs to run before running evaluation')
    return ap.parse_args()

