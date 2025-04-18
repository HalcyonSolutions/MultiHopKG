#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from collections.abc import Mapping
from typing import Any, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader, Dataset

from multihopkg.utils.convenience import sample_random_entity
from multihopkg.emb.operations import normalize_angle_smooth, normalize_angle, angular_difference

from multihopkg.datasets import TestDataset

class KGEModel(nn.Module):

    def __init__(
        self,
        model_name: str,
        nentity: int,
        nrelation: int,
        hidden_dim: int,
        gamma: float,
        double_entity_embedding: bool = False,
        double_relation_embedding: bool = False,
    ):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
        # self.centroid = calculate_entity_centroid(self.entity_embedding)
        self.centroid = None
        self.embedding_range_max = None
        self.embedding_range_min = None

        # Initialize the model forward function dictionary once
        self.model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }
        
        # Initialize the flexible forward function dictionary once
        self.flexible_func = {
            "TransE": self.flexible_forward_transe,
            "RotatE": self.flexible_forward_rotate,
            "pRotatE": self.flexible_forward_protate,
        }

        # Initialize the absolute difference function dictionary once
        self.absolute_difference_func = {
            "TransE": self.absolute_difference_euclidean,
            "RotatE": self.absolute_difference_euclidean,
            "pRotatE": self.absolute_difference_phase,
        }

        # Initialize the denormalize and wrap functions for relations
        self.relation_denormalize_func = {
            "TransE": lambda x: x, # no operation is needed
            "RotatE": self.denormalize_embedding, # also a phase
            "pRotatE": self.denormalize_embedding,
        }

        self.relation_wrap_func = {
            "TransE": lambda x: x, # no operation is needed
            "RotatE": self.wrap_rotate_embedding, # also a phase
            "pRotatE": self.wrap_rotate_embedding,
        }

        # Initialize the denormalize and wrap functions for entities
        self.entity_denormalize_func = {
            "TransE": lambda x: x, # no operation is needed
            "RotatE": lambda x: x, # no operation is needed, complex number
            "pRotatE": self.denormalize_embedding,
        }

        self.entity_wrap_func = {
            "TransE": lambda x: x, # no operation is needed
            "RotatE": lambda x: x, # no operation is needed, complex number
            "pRotatE": self.wrap_rotate_embedding,
        }

    def load_embeddings(self, entity_embedding: np.ndarray, relation_embedding: np.ndarray):
        '''
        Load the entity and relation embeddings from the given paths.
        '''
        self.entity_embedding.data = torch.from_numpy(entity_embedding)
        self.relation_embedding.data = torch.from_numpy(relation_embedding)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        entity_embedding: np.ndarray,
        relation_embedding: np.ndarray,
        gamma: float,
        state_dict: Mapping[str, Any],
    ) -> 'KGEModel':
        """
        Create a KGEModel from pretrained embeddings.
        
        Args:
            model_name: Name of the model (TransE, DistMult, etc.)
            nentity: Number of entities
            nrelation: Number of relations
            entity_embedding: Pretrained entity embeddings as numpy array
            relation_embedding: Pretrained relation embeddings as numpy array
            gamma: Margin value for distance-based loss functions
            
        Returns:
            Initialized KGEModel with pretrained embeddings
        """
        # Derive dimensions from the embeddings
        entity_dim = entity_embedding.shape[1]
        relation_dim = relation_embedding.shape[1]
        
        # Determine if using double embeddings
        double_entity_embedding = (model_name in ['RotatE', 'ComplEx'])
        double_relation_embedding = (model_name == 'ComplEx')
        
        nentity = entity_embedding.shape[0]
        nrelation = relation_embedding.shape[0]

        # Calculate hidden dim based on entity dimension and embedding type
        if double_entity_embedding:
            hidden_dim = entity_dim // 2
        else:
            hidden_dim = entity_dim
            
        # Create model instance
        model = cls(
            model_name=model_name,
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=hidden_dim,
            gamma=gamma,
            double_entity_embedding=double_entity_embedding,
            double_relation_embedding=double_relation_embedding,
        )
        
        # Load pretrained embeddings
        model.load_embeddings(entity_embedding, relation_embedding)
        model.load_state_dict(state_dict)

        # Only makes sense in Euclidean space
        model.centroid = calculate_entity_centroid(model.entity_embedding)
        model.embedding_range_min, model.embedding_range_max = calculate_entity_range(model.entity_embedding)
        return model

    #-----------------------------------------------------------------------
    'Forward Function'
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)

        
        if self.model_name in self.model_func:
            score = self.model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    #-----------------------------------------------------------------------
    'Scoring Functions'

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/torch.pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head # Brodcasting is expected here. 
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/torch.pi)
        phase_relation = relation/(self.embedding_range.item()/torch.pi)
        phase_tail = tail/(self.embedding_range.item()/torch.pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    #-----------------------------------------------------------------------
    'Training and Evaluation'

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics

    #-----------------------------------------------------------------------
    'Translation in Embedding Space'

    def flexible_forward_rotate(
        self, cur_states: torch.Tensor, cur_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies a phase translation to the head entity given the relation. 
        This is meant to work on the original RotatE model.
        """

        tail = self.RotatE_Eval(cur_states, cur_actions)

        return tail


    def RotatE_Eval(self, head, relation):
        """
        Calculates the phase translation of the head entity given the relation.
        Assumes that the entity is a complex number in a flatten vector form,
        where the first half is the real and the latter half the imaginary.
        The relation is a phase translation in radians.
        
        NOTE: The translation does not change the magnitude of the head entity, despite
        the fact that the entities all have different magnitudes. This function only offers
        a phase translation, not a magnitude translation.

        args:
            head: torch.Tensor. Head embedding (flattened complex number)
            relation: torch.Tensor. Relation embedding (phase in radians)

        returns:
            torch.Tensor. Translated head embedding (flattened complex number)
        """

        re_head, im_head = torch.chunk(head, 2, dim=1)

        phase_relation = self.denormalize_embedding(relation)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_est_tail = re_head * re_relation - im_head * im_relation
        im_est_tail = re_head * im_relation + im_head * re_relation

        return torch.cat([re_est_tail, im_est_tail], dim=-1)


    def flexible_forward_protate(
        self, cur_states: torch.Tensor, cur_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies a phase rotation to the head entity given the relation. 
        This is meant to work on the original pRotatE model.
        """
        head_rad = self.denormalize_embedding(cur_states)
        
        rotation_rad = self.pRotatE_Eval(head_rad, cur_actions)        # apply the phase rotation, result in rads   

        # TODO: VERY IMPORTANT: Verify if this doesn't break the learning process
        # ! Observation: If there is no normalization, there is NaN somewhere in the log_prob or rewards
        rotation_rad = normalize_angle_smooth(rotation_rad) # smooth normalization to [-pi, pi] since it is cyclic, using trigonometric functions

        return self.normalize_embedding(rotation_rad)


    def pRotatE_Eval(self, phase_head, relation):
        """
        Calculates the phase rotation of the head entity given the relation. 
        Assumes that the entity is represented as a phase in radians (not a complex number).
        
        args:
            phase_head: torch.Tensor. Phase of the head embedding (in radians)
            relation: torch.Tensor

        returns:
            torch.Tensor. Phase of the translated head embedding (in radians). Not limited to [-pi, pi]
        """
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]
        phase_relation = normalize_angle_smooth(self.denormalize_embedding(relation))

        phase_translation = phase_head + phase_relation

        return phase_translation
    

    def flexible_forward_transe(
        self, cur_states: torch.Tensor, cur_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies a euclidean translation to the head entity given the relation. 
        This is meant to work on the original TransE model.
        """

        tail = self.TransE_Eval(cur_states, cur_actions)

        # TODO: Improve upon this, this is a temporatory solution
        # Extract the max and min and use them instead
        tail = torch.clamp(tail, self.embedding_range_min, self.embedding_range_max)

        return tail


    def TransE_Eval(self, head, relation):
        """
        Calculates the euclidean translation of the head entity given the relation.
        Assumes that the entity is represented as a vector (not a phase nor complex number).

        args:
            head: torch.Tensor. Head embedding
            relation: torch.Tensor. Relation embedding
        
        returns:
            torch.Tensor. Translated head embedding
        """
        return head + relation


    def flexible_forward(self, cur_states: torch.Tensor, cur_actions: torch.Tensor) -> torch.Tensor:
        """
        Execute the flexible forward pass for the model.
        This function is used for the flexible action space.
        """
        
        if self.model_name in self.flexible_func:
            return self.flexible_func[self.model_name](cur_states, cur_actions)
        else:
            raise ValueError(f"Model {self.model_name} does not support flexible forward pass.")

    #-----------------------------------------------------------------------
    'Distance Calculations'

    def absolute_difference_euclidean(self, head: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """
        Calculate the absolute difference between two vectors in Euclidean space.
        Also used by RotatE, but not pRotatE.

        args:
            head: torch.Tensor. Head embedding
            tail: torch.Tensor. Tail embedding
        
        returns:
            torch.Tensor. Absolute difference between head and tail embeddings
        """
        return torch.abs(head - tail)
    

    def absolute_difference_phase(self, head: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """
        Calculate the absolute difference between two vectors in phase space.
        Does not denormalize the result (stays in radians).
        
        args:
            head: torch.Tensor. Head embedding (in radians)
            tail: torch.Tensor. Tail embedding (in radians)
        
        returns:
            torch.Tensor. Absolute difference between head and tail embeddings
        """
        head = self.denormalize_embedding(head)
        tail = self.denormalize_embedding(tail)
        return angular_difference(head, tail, smooth=torch.is_grad_enabled()) # if gradient is required, use smooth normalization
        
    
    def absolute_difference(self, head: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """
        Calculate the absolute difference between two vectors.
        Works for both Euclidean and Rotational space.
        Compatible with gradient calculation and none gradient calculation.
        Does not denormalize the result.
        
        args:
            head: torch.Tensor. Head embedding
            tail: torch.Tensor. Tail embedding
        
        returns:
            torch.Tensor. Absolute difference between head and tail embeddings
        """
        
        if self.model_name in self.absolute_difference_func:
            return self.absolute_difference_func[self.model_name](head, tail)
        else:
            raise ValueError(f"Model {self.model_name} does not support absolute difference calculation.")

    #-----------------------------------------------------------------------
    'Normalization, Denormalization, and Wrapping'

    def denormalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Process the embedding tensor to ensure it is in the correct format.
        
        args:
            embedding: torch.Tensor. Action tensor
        
        returns:
            torch.Tensor. Processed embedding tensor
        """
        return embedding/(self.embedding_range.item()/torch.pi)


    def normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Process the embedding tensor to ensure it is in the correct format.
        
        args:
            embedding: torch.Tensor. Action tensor
        
        returns:
            torch.Tensor. Processed embedding tensor
        """
        return embedding * (self.embedding_range.item()/torch.pi)
    
    def wrap_rotate_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Wrap the embedding tensor to ensure it is in the correct format.
        
        args:
            embedding: torch.Tensor. Action tensor
        
        returns:
            torch.Tensor. Wrapped embedding tensor
        """
        if torch.is_grad_enabled():
            return normalize_angle_smooth(embedding)
        else:
            return normalize_angle(embedding)

    def denormalize_relation(self, relation: torch.Tensor) -> torch.Tensor:
        if self.model_name in self.relation_denormalize_func:
            return self.relation_denormalize_func[self.model_name](relation)
        else:
            raise ValueError(f"Model {self.model_name} does not support relation denormalization.")
    
    def wrap_relation(self, relation: torch.Tensor) -> torch.Tensor:
        if self.model_name in self.relation_wrap_func:
            return self.relation_wrap_func[self.model_name](relation)
        else:
            raise ValueError(f"Model {self.model_name} does not support relation wrapping.")
    
    def denormalize_entity(self, entity: torch.Tensor) -> torch.Tensor:
        if self.model_name in self.entity_denormalize_func:
            return self.entity_denormalize_func[self.model_name](entity)
        else:
            raise ValueError(f"Model {self.model_name} does not support entity denormalization.")
    
    def wrap_entity(self, entity: torch.Tensor) -> torch.Tensor:
        if self.model_name in self.entity_wrap_func:
            return self.entity_wrap_func[self.model_name](entity)
        else:
            raise ValueError(f"Model {self.model_name} does not support entity wrapping.")

    #-----------------------------------------------------------------------
    'Getters'

    def get_centroid(self) -> torch.Tensor:
        return self.centroid

    def get_entity_dim(self):
        return self.entity_dim

    def get_relation_dim(self):
        return self.relation_dim

    def get_all_entity_embeddings_wo_dropout(self) -> torch.Tensor:
        assert isinstance(self.entity_embedding, nn.Parameter) or isinstance(
            self.entity_embedding, nn.Embedding
        ), "The entity embedding must be either a nn.Parameter or nn.Embedding"
        return self.entity_embedding.data

    def get_all_relations_embeddings_wo_dropout(self) -> torch.Tensor:
        assert isinstance(self.relation_embedding, nn.Parameter) or isinstance(
            self.relation_embedding, nn.Embedding
        ), "The relation embedding must be either a nn.Parameter or nn.Embedding"
        return self.relation_embedding

    def get_starting_embedding(self, startType: str = 'centroid', ent_id: torch.Tensor = None)   -> torch.Tensor:
        """
        Returns the starting point for the navigation.
            
            :param startType: The type of starting point to use. Options are 'centroid', 'random', 'relevant'
            :param ent_id: The entity id to use as the starting point if 'relevant' is chosen.
            :return: The starting point for the navigation.
        """
        if startType == 'centroid':
            return self.get_centroid()
        elif startType == 'random':
            return sample_random_entity(self.entity_embedding)
        elif startType == 'relevant' and not (isinstance(ent_id, type(None))):
            return get_embeddings_from_indices(self.entity_embedding, ent_id)
        else:
            raise Warning("Invalid navigation starting type/point. Using centroid instead.")
            return self.centroid

def get_embeddings_from_indices(embeddings: Union[nn.Embedding, nn.Parameter], indices: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor of indices, returns the embeddings of the corresponding rows.

    Args:
        embeddings (Union[nn.Embedding, nn.Parameter]): The embedding matrix.
        indices (torch.Tensor): A tensor of indices.

    Returns:
        torch.Tensor: The embeddings corresponding to the given indices.
    """

    if isinstance(embeddings, nn.Parameter):
        return embeddings.data[indices]
    elif isinstance(embeddings, nn.Embedding):
        return embeddings.weight.data[indices]
    else:
        raise TypeError("Embeddings must be either nn.Parameter or nn.Embedding")

def calculate_entity_centroid(embeddings: Union[nn.Embedding, nn.Parameter]):
    if isinstance(embeddings, nn.Parameter):
        entity_centroid = torch.mean(embeddings.data, dim=0)
    elif isinstance(embeddings, nn.Embedding):
        entity_centroid = torch.mean(embeddings.weight.data, dim=0)
    return entity_centroid

def calculate_entity_range(embeddings: Union[nn.Embedding, nn.Parameter]):
    if isinstance(embeddings, nn.Parameter):
        max_range = torch.max(embeddings.data).item()
        min_range = torch.min(embeddings.data).item()
    elif isinstance(embeddings, nn.Embedding):
        max_range = torch.max(embeddings.weight.data).item()
        min_range = torch.min(embeddings.weight.data).item()
    return min_range, max_range

class LegacyKGEModel(nn.Module):
    def __init__(
        self,
        model_name,
        nentity,
        nrelation,
        hidden_dim,
        gamma,
        double_entity_embedding=False,
        double_relation_embedding=False,
    ):
        super(LegacyKGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False,
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        ########################################
        # Embedddingsss
        ########################################
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )

        self.relation_embedding = nn.Parameter(
            torch.zeros(nrelation, self.relation_dim)
        )
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )
        ########################################
        # Embedddingsss Done
        ########################################

        if model_name == "pRotatE":
            self.modulus = nn.Parameter(
                torch.Tensor([[0.5 * self.embedding_range.item()]])
            )

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ["TransE", "DistMult", "ComplEx", "RotatE", "pRotatE"]:
            raise ValueError("model %s not supported" % model_name)

        if model_name == "RotatE" and (
            not double_entity_embedding or double_relation_embedding
        ):
            raise ValueError("RotatE should use --double_entity_embedding")

        if model_name == "ComplEx" and (
            not double_entity_embedding or not double_relation_embedding
        ):
            raise ValueError(
                "ComplEx should use --double_entity_embedding and --double_relation_embedding"
            )

    def flexible_forward_rotate(
        self, cur_states: torch.Tensor, cur_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Will be a way of executing continuous actions here
        """
        batch_size = cur_states.shape[0]

        head = cur_states
        relation = cur_actions
        tail = self.RotatE_Eval(head, relation)

        return tail

    def forward(self, sample, mode="single"):
        """
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        """

        if mode == "single":
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding, dim=0, index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding, dim=0, index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding, dim=0, index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == "head-batch":
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding, dim=0, index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding, dim=0, index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding, dim=0, index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == "tail-batch":
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding, dim=0, index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding, dim=0, index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding, dim=0, index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError("mode %s not supported" % mode)

        model_func = {
            "TransE": self.TransE,
            "DistMult": self.DistMult,
            "ComplEx": self.ComplEx,
            "RotatE": self.RotatE,
            "pRotatE": self.pRotatE,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError("model %s not supported" % self.model_name)

        return score, head, relation, tail

    def TransE(self, head, relation, tail, mode):
        if mode == "head-batch":
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == "head-batch":
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == "head-batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE_Eval(self, head, relation):

        re_head, im_head = torch.chunk(head, 2, dim=1)

        phase_relation = relation / (self.embedding_range.item() / torch.pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_est_tail = re_head * re_relation - im_head * im_relation
        im_est_tail = re_head * im_relation + im_head * re_relation

        return torch.cat([re_est_tail, im_est_tail], dim=-1)

    def RotatE(self, head, relation, tail, mode):

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / torch.pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == "head-batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / torch.pi)
        phase_relation = relation / (self.embedding_range.item() / torch.pi)
        phase_tail = tail / (self.embedding_range.item() / torch.pi)

        if mode == "head-batch":
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        """
        A single train step. Apply back-propation and return the loss
        """

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(
            train_iterator
        )

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (
                F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                * F.logsigmoid(-negative_score)
            ).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = -positive_score.mean()
            negative_sample_loss = -negative_score.mean()
        else:
            positive_sample_loss = (
                -(subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            )
            negative_sample_loss = (
                -(subsampling_weight * negative_score).sum() / subsampling_weight.sum()
            )

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p=3) ** 3
                + model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {"regularization": regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            "positive_sample_loss": positive_sample_loss.item(),
            "negative_sample_loss": negative_sample_loss.item(),
            "loss": loss.item(),
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, 
                  nentity, nrelation, test_batch_size, cpu_num = 1, cuda = False):
        """
        Evaluate the model on test or valid datasets
        """
        if cuda: model.cuda()
        model.eval()
            
        #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        #Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                nentity, 
                nrelation, 
                'head-batch'
            ), 
            batch_size=test_batch_size,
            num_workers=max(1, cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                nentity, 
                nrelation, 
                'tail-batch'
            ), 
            batch_size=test_batch_size,
            num_workers=max(1, cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )
        
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        
        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    if cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score, _, _, _ = model((positive_sample, negative_sample), mode)
                    score += filter_bias

                    #Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim = 1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        #Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        #ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0/ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    step += 1
                    print(f"Step: {step}/{total_steps}")

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
