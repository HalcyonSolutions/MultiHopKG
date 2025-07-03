#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random
import wandb
import time

import numpy as np
import torch

from torch.utils.data import DataLoader

from multihopkg.exogenous.sun_models import KGEModel, save_model, update_best_model, clean_up_checkpoints, save_configs
from multihopkg.utils.data_splitting import read_triple

from multihopkg.utils.setup import set_seeds

from multihopkg.datasets import TrainDataset
from multihopkg.datasets import BidirectionalOneShotIterator, MultiTaskIterator, OneShotIterator, build_type_constraints

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--clean_up', action='store_true', help='Clean up checkpoints after training')
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--task', type=str, choices=['link_prediction', 'relation_prediction', 'all'], default='link_prediction',
                        help='Specify which task to train: link_prediction, relation-prediction, or all (multi-task)')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--reload_entities', action='store_true', help='Reload entity embeddings from checkpoint for transfer learning')
    parser.add_argument('--reload_relationship', action='store_true', help='Reload relation embeddings from checkpoint for transfer learning')
    parser.add_argument('--freeze_entities', action='store_true', help='Freeze entity embeddings (no gradient updates)')
    parser.add_argument('--freeze_relationship', action='store_true', help='Freeze relation embeddings (no gradient updates)')

    parser.add_argument('--autoencoder_flag', action='store_true', help='Toggle autoencoder')
    parser.add_argument('--autoencoder_hidden_dim', default=50, type=int, help='Autoencoder hidden dimension')
    parser.add_argument('--autoencoder_lambda', default=0.1, type=float, help='Autoencoder regularization')

    parser.add_argument('--wandb_project', type=str, default='', help='wandb project name')
    parser.add_argument('-track', action='store_true', help='track wandb')

    parser.add_argument('--saving_metric', default='', type=str, help='Metric used for the threshold required for saving model. If empty, no conditioning for saving model.')
    parser.add_argument('--saving_threshold', default=0.0, type=float, help='threshold required for saving model')

    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for the environment. If None, not used.")
    parser.add_argument("--timestamp", type=str, default=None, help="Timestamp for the run. If None, current time is used.")

    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

    # Log to wandb as well
    if wandb.run is not None:
        wandb.log({f"{mode}_{metric}": value for metric, value in metrics.items()}, step=step)    

def reload_embeddings_only(kge_model, init_checkpoint, reload_entities=False, reload_relationship=False):
    """
    Reload only entity or relation embeddings from a checkpoint directory.
    """
    checkpoint_path = os.path.join(init_checkpoint, 'checkpoint')
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']

    current_state = kge_model.state_dict()

    reload_keys = []
    if reload_entities:
        # All keys for entity embedding
        reload_keys.extend([k for k in state_dict.keys() if "entity_embedding" in k])
    if reload_relationship:
        reload_keys.extend([k for k in state_dict.keys() if "relation_embedding" in k])
        # If you have autoencoder weights for relations, add those here if you wish:
        # reload_keys.extend([k for k in state_dict.keys() if "relation_encoder" in k or "relation_decoder" in k])

    # Overwrite only requested keys
    for key in reload_keys:
        if key in current_state:
            current_state[key] = state_dict[key]
    kge_model.load_state_dict(current_state, strict=False)
    logging.info(f"Reloaded embeddings: {', '.join(reload_keys)} from {checkpoint_path}")

def create_dataloader(train_triples, nentity, nrelation, negative_sample_size, batch_size, cpu_num, mode):
    return DataLoader(
        TrainDataset(train_triples, nentity, nrelation, negative_sample_size, mode),
        batch_size=batch_size,
        shuffle=True,
        num_workers=max(1, cpu_num // 2),
        collate_fn=TrainDataset.collate_fn
    )

def main(args):
    # Initialize wandb

    if args.random_seed is not None:
        set_seeds(args.random_seed)

    if args.timestamp is None:
        local_time = time.localtime()
        args.timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)

    if args.track:
        if args.wandb_project == '':
            raise ValueError('wandb_project must be specified if tracking is enabled.')
        wandb.init(
            project=f"{args.wandb_project}",
            config=vars(args),
            name=f"{args.model}-{args.data_path.split('/')[1]}-{args.timestamp}"
        )
        args = argparse.Namespace(**wandb.config)  # <-- Make sure args is overwritten

    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    domain_constraints, range_constraints = build_type_constraints(all_true_triples)
    constraints = {
        'domain_constraints': domain_constraints,
        'range_constraints': range_constraints
    }

    # Logging before initializing the model
    if args.autoencoder_flag and not args.double_relation_embedding:
        logging.info('Autoencoder toggled ON')
        logging.info(f'Autoencoder hidden dim: {args.autoencoder_hidden_dim}')
        logging.info(f'Autoencoder lambda: {args.autoencoder_lambda}')
    else:
        logging.info('Autoencoder toggled OFF')
        args.autoencoder_flag = False # in case if double_relation_embedding is set to True

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        autoencoder_flag=args.autoencoder_flag,
        autoencoder_hidden_dim=args.autoencoder_hidden_dim,
        autoencoder_lambda=args.autoencoder_lambda,
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
    
    if args.do_train:
        # Set training dataloader iterator
        if args.task == 'all':
            modes = ['head-batch', 'tail-batch', 'relation-batch']
            dataloaders = [(mode, create_dataloader(train_triples, nentity, nrelation, args.negative_sample_size, args.batch_size, args.cpu_num, mode)) for mode in modes]
            train_iterator = MultiTaskIterator(dataloaders)

        elif args.task == 'link_prediction':
            train_dataloader_head = create_dataloader(train_triples, nentity, nrelation, args.negative_sample_size, args.batch_size, args.cpu_num, 'head-batch')
            train_dataloader_tail = create_dataloader(train_triples, nentity, nrelation, args.negative_sample_size, args.batch_size, args.cpu_num, 'tail-batch')
            train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        elif args.task == 'relation_prediction':
            train_dataloader_relation = create_dataloader(train_triples, nentity, nrelation, args.negative_sample_size, args.batch_size, args.cpu_num, 'relation-batch')
            train_iterator = OneShotIterator(train_dataloader_relation)
        else:
            raise ValueError(f"Unknown task: {args.task}. Supported tasks are 'link_prediction' and 'relation-prediction'.")
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

        if not(args.do_valid): args.saving_metric = '' # No validation, no saving condition
        if args.saving_metric not in ['', 'MRR', 'HITS@1', 'HITS@3', 'HITS@10']:
            logging.warning(f'Invalid saving metrics: {args.saving_metric}. Must be one of MRR, HITS@1, HITS@3, HITS@10 or empty. Setting to empty.')
            args.saving_metric = ''

    if args.init_checkpoint:
        if getattr(args, "reload_entities", False) or getattr(args, "reload_relationship", False):
            # Only reload selected embeddings (transfer learning mode)
            reload_embeddings_only(
                kge_model,
                args.init_checkpoint,
                reload_entities=getattr(args, "reload_entities", False),
                reload_relationship=getattr(args, "reload_relationship", False),
            )
            # Initialize optimizer as new!
            init_step = 0
        else:
            # Restore model from checkpoint directory
            logging.info('Loading checkpoint %s...' % args.init_checkpoint)
            checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
            init_step = checkpoint['step']
            kge_model.load_state_dict(checkpoint['model_state_dict'])
            if args.do_train:
                current_learning_rate = checkpoint['current_learning_rate']
                warm_up_steps = checkpoint['warm_up_steps']
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    # Freeze entity and relation embeddings if specified
    if getattr(args, "freeze_entities", False):
        kge_model.entity_embedding.requires_grad = False
        logging.info("Entity embeddings frozen (requires_grad=False)")

    if getattr(args, "freeze_relationship", False):
        kge_model.relation_embedding.requires_grad = False
        logging.info("Relation embeddings frozen (requires_grad=False)")
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training
    

    best_metric_value = None
    best_model_path = None
    if args.do_train:
        logging.info('learning_rate = %f' % current_learning_rate)

        training_logs = []
        
        #Training Loop
        for step in range(init_step, args.max_steps):
            
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            
            training_logs.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0 and args.saving_metric == '':
                # Normal saving without metric condition
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_configs(args)
                save_model(
                    kge_model,
                    optimizer,
                    save_variable_list,
                    args.save_path,
                    args.autoencoder_flag
                )
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args, constraints=constraints)
                log_metrics('Valid', step, metrics)

                # If the metric is present and above the threshold, save the model
                if args.saving_metric in metrics and metrics[args.saving_metric] > args.saving_threshold:
                    
                    save_variable_list = {
                        'step': step, 
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    save_configs(args)
                    save_dir = os.path.join(args.save_path, 'checkpoints', str(step))
                    os.makedirs(save_dir, exist_ok=True)
                    save_model(
                        kge_model,
                        optimizer, 
                        save_variable_list, 
                        save_dir, 
                        args.autoencoder_flag
                    )

                    # Track the best model (assuming higher is better for your metric; set maximize=False if lower is better)
                    if args.saving_metric in metrics:
                        best_metric_value, best_model_path = update_best_model(
                            kge_model, optimizer, save_variable_list, args.save_path,
                            args.saving_metric, metrics[args.saving_metric], 
                            best_metric_value, best_model_path,
                            autoencoder_flag=args.autoencoder_flag, maximize=True
                        )
        
        # Save the final model
        if args.saving_metric == '':
            logging.info('Final Evaluation on Valid Dataset...')
            save_variable_list = {
                'step': step, 
                'current_learning_rate': current_learning_rate,
                'warm_up_steps': warm_up_steps
            }
            save_configs(args)
            save_model(
                kge_model,
                optimizer,
                save_variable_list,
                args.save_path,
                args.autoencoder_flag
            )
        else:
            logging.info('Final Evaluation on Valid Dataset...')
            metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args, constraints=constraints)
            log_metrics('Valid', step, metrics)

            if args.saving_metric in metrics and metrics[args.saving_metric] > args.saving_threshold:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_configs(args)
                save_dir = os.path.join(args.save_path, 'checkpoints', str(step))
                os.makedirs(save_dir, exist_ok=True)
                save_model(
                    kge_model,
                    optimizer, 
                    save_variable_list, 
                    save_dir, 
                    args.autoencoder_flag
                )
                best_metric_value, best_model_path = update_best_model(
                    kge_model, optimizer, save_variable_list, args.save_path,
                    args.saving_metric, metrics[args.saving_metric], 
                    best_metric_value, best_model_path,
                    autoencoder_flag=args.autoencoder_flag, maximize=True
                )
            
            if getattr(args, 'clean_up', False):
                clean_up_checkpoints(args.save_path)
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args, constraints=constraints)
        log_metrics('Valid', step, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args, constraints=constraints)
        log_metrics('Test', step, metrics)
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args, constraints=constraints)
        log_metrics('Test', step, metrics)
        
if __name__ == '__main__':
    main(parse_args())
