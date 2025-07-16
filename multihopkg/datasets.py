from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

from typing import List, Tuple

class TestDataset(Dataset):
    __test__ = False # To avoid pytest confusion

    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.domain_set = set((h, r) for h, r, _ in all_true_triples)
        self.range_set = set((r, t) for _, r, t in all_true_triples)
        self.neighbor_set = set((h, t) for h, _, t in all_true_triples)
        self.triples = triples
        self.nentity = nentity # do not include the wildcard entities
        self.nrelation = nrelation # do not include the wildcard relation
        self.mode = mode

        # Filter triples based on the mode
        if self.mode in ['domain-batch', 'nbr-head-batch']:
            self.triples = self.filter_wildcard_triples(triples, 'tail', self.nentity + 1)
            self.len = len(self.triples)
        elif self.mode in ['range-batch', 'nbr-tail-batch']:
            self.triples = self.filter_wildcard_triples(triples, 'head', self.nentity)
            self.len = len(self.triples)
        elif self.mode in ['nbe-head-batch', 'nbe-tail-batch']:
            self.triples = self.filter_wildcard_triples(triples, 'relation', self.nrelation)
            self.len = len(self.triples)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        # TODO: Inspect if wildcard id need to be replaced here
        if self.mode == 'head-batch':
            tmp = [(0, candidate_head) if (candidate_head, relation, tail) not in self.triple_set
                   else (-1, head) for candidate_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, candidate_tail) if (head, relation, candidate_tail) not in self.triple_set
                   else (-1, tail) for candidate_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        elif self.mode == 'relation-batch':
            tmp = [(0, candidate_rel) if (head, candidate_rel, tail) not in self.triple_set
                else (-1, relation) for candidate_rel in range(self.nrelation)]
            tmp[relation] = (0, relation)
        elif self.mode == 'domain-batch':
            tmp = [(0, candidate_head) if (candidate_head, relation) not in self.domain_set
                   else (-1, head) for candidate_head in range(self.nentity)]
            tmp[head] = (0, head)
            # tail = self.nentity + 1  # Use a wildcard tail entity
        elif self.mode == 'range-batch':
            tmp = [(0, candidate_tail) if (relation, candidate_tail) not in self.range_set
                   else (-1, tail) for candidate_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
            # head = self.nentity  # Use a wildcard head entity
        elif self.mode == 'nbe-head-batch':
            tmp = [(0, candidate_head) if (candidate_head, tail) not in self.neighbor_set
                   else (-1, head) for candidate_head in range(self.nentity)]
            tmp[head] = (0, head)
            # relation = self.nrelation  # Use a wildcard relation
        elif self.mode == 'nbe-tail-batch':
            tmp = [(0, candidate_tail) if (head, candidate_tail) not in self.neighbor_set
                   else (-1, tail) for candidate_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
            # relation = self.nrelation # Use a wildcard relation
        elif self.mode == 'nbr-head-batch':
            tmp = [(0, candidate_rel) if (head, candidate_rel) not in self.domain_set
                   else (-1, relation) for candidate_rel in range(self.nrelation)]
            tmp[relation] = (0, relation)
            # tail = self.nentity + 1  # Use a wildcard tail entity
        elif self.mode == 'nbr-tail-batch':
            tmp = [(0, candidate_rel) if (candidate_rel, tail) not in self.range_set
                   else (-1, relation) for candidate_rel in range(self.nrelation)]
            tmp[relation] = (0, relation)
            # head = self.nentity  # Use a wildcard head entity
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
        
        tmp = torch.LongTensor(tmp)         
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode

    @staticmethod
    def filter_wildcard_triples(triples: List[Tuple[str]], wildcard_position:str , wildcard_value: int):
        """
        Filter triples based on a wildcard position and value.
        Prevents duplicate triples with wildcard values.
        """
        if wildcard_position == 'head':
            return list(set([(wildcard_value, relation, tail) for (_, relation, tail) in triples]))
        elif wildcard_position == 'relation':
            return list(set([(head, wildcard_value, tail) for (head, _, tail) in triples]))
        elif wildcard_position == 'tail':
            return list(set([(head, relation, wildcard_value) for (head, relation, _) in triples]))
        else:
            raise ValueError("Invalid wildcard position specified.")

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity # do not include the wildcard entities
        self.nrelation = nrelation # do not include the wildcard relation
        self.negative_sample_size = negative_sample_size
        self.mode = mode

        # Filter triples based on the mode
        if self.mode in ['domain-batch', 'nbr-head-batch']:
            self.triples = self.filter_wildcard_triples(self.triples, 'tail', self.nentity + 1)
            self.triple_set = set(self.triples)
            self.len = len(self.triples)
        elif self.mode in ['range-batch', 'nbr-tail-batch']:
            self.triples = self.filter_wildcard_triples(self.triples, 'head', self.nentity)
            self.triple_set = set(self.triples)
            self.len = len(self.triples)
        elif self.mode in ['nbe-head-batch', 'nbe-tail-batch']:
            self.triples = self.filter_wildcard_triples(self.triples, 'relation', self.nrelation)
            self.triple_set = set(self.triples)
            self.len = len(self.triples)

        # Count frequency of partial triples for subsampling
        self.count, self.rel_count = self.count_frequency(self.triples)

        # Prepare true triples for negative sampling
        if self.mode in ['head-batch', 'relation-batch', 'tail-batch']:
            self.true_head, self.true_tail, self.true_rels = self.get_true_head_and_tail(self.triples)
        elif self.mode in ['domain-batch', 'range-batch']:
            self.true_domain, self.true_range = self.get_true_domain_and_range(self.triples)
        elif self.mode in ['nbe-head-batch', 'nbe-tail-batch']:
            self.true_neighbor_head, self.true_neighbor_tail = self.get_true_entity_neighbors(self.triples)
        elif self.mode in ['nbr-head-batch', 'nbr-tail-batch']:
            self.true_relation_head, self.true_relation_tail = self.get_true_relation_neighbors(self.triples)
        else:
            raise ValueError('Training batch mode %s not supported' % self.mode)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample
        
        negative_sample_list = []
        negative_sample_size = 0

        if self.mode in ['head-batch', 'tail-batch', 'domain-batch', 'range-batch', 'nbr-head-batch', 'nbr-tail-batch']:
            subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        elif self.mode in ['relation-batch', 'nbe-head-batch', 'nbe-tail-batch']:
            subsampling_weight = self.rel_count[(head, tail)]
        else:
            raise ValueError('Training batch mode %s not supported' % self.mode)
        
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        max_int = 0
        true_tokens = None
        if self.mode == 'head-batch':
            true_tokens = self.true_head[(relation, tail)]  # guessing the head
            max_int = self.nentity
        elif self.mode == 'tail-batch':
            true_tokens = self.true_tail[(head, relation)]  # guessing the tail
            max_int = self.nentity
        elif self.mode == 'relation-batch':
            true_tokens = self.true_rels[(head, tail)]      # guessing the relation
            max_int = self.nrelation
        elif self.mode == 'domain-batch':                   # Do not consider the wildcard entities in negative sampling
            true_tokens = self.true_domain[relation]        # guessing the head
            max_int = self.nentity
        elif self.mode == 'range-batch':                    # Do not consider the wildcard entities in negative sampling
            true_tokens = self.true_range[relation]         # guessing the tail
            max_int = self.nentity
        elif self.mode == 'nbe-head-batch':                 # Do not consider the wildcard entities in negative sampling
            true_tokens = self.true_neighbor_head[tail]     # guessing the head
            max_int = self.nentity
        elif self.mode == 'nbe-tail-batch':                 # Do not consider the wildcard entities in negative sampling
            true_tokens = self.true_neighbor_tail[head]     # guessing the tail
            max_int = self.nentity
        elif self.mode == 'nbr-head-batch':                  # Do not consider the wildcard entities in negative sampling
            true_tokens = self.true_relation_head[head]      # guessing the relation
            max_int = self.nrelation
        elif self.mode == 'nbr-tail-batch':                  # Do not consider the wildcard entities in negative sampling
            true_tokens = self.true_relation_tail[tail]
            max_int = self.nrelation
        else:
            raise ValueError('Training batch mode %s not supported' % self.mode)

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(max_int, size=self.negative_sample_size*2)
            mask = np.in1d(
                negative_sample, 
                true_tokens,  # guessing the head, tail, or relation
                assume_unique=True, 
                invert=True
            )
            negative_sample = negative_sample[mask] # list of fake entities or relations, not triplets
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
            
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        rel_count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1

            # Relation-batch count (count how often (head, tail) appears with any relation)
            if (head, tail) not in rel_count:
                rel_count[(head, tail)] = start
            else:
                rel_count[(head, tail)] += 1
        return count, rel_count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = defaultdict(set)
        true_tail = defaultdict(set)
        true_rels = defaultdict(set)

        for head, relation, tail in triples:
            true_tail[(head, relation)].add(tail)
            true_head[(relation, tail)].add(head)
            true_rels[(head, tail)].add(relation)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(true_head[(relation, tail)]))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(true_tail[(head, relation)]))
        for head, tail in true_rels:
            true_rels[(head, tail)] = np.array(list(true_rels[(head, tail)]))                 

        return true_head, true_tail, true_rels
    
    @staticmethod
    def get_true_domain_and_range(triples):
        """
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        """
        true_domain = defaultdict(set)
        true_range = defaultdict(set)

        for head, relation, tail in triples:
            true_domain[relation].add(head)
            true_range[relation].add(tail)

        for relation in true_domain:
            true_domain[relation] = np.array(list(true_domain[relation]))
        for relation in true_range:
            true_range[relation] = np.array(list(true_range[relation]))

        return true_domain, true_range

    @staticmethod
    def get_true_entity_neighbors(triples):
        """
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        """
        true_neighbor_head = defaultdict(set)
        true_neighbor_tail = defaultdict(set)

        for head, _, tail in triples:
            true_neighbor_head[tail].add(head)
            true_neighbor_tail[head].add(tail)

        for tail in true_neighbor_head:
            true_neighbor_head[tail] = np.array(list(true_neighbor_head[tail]))
        for head in true_neighbor_tail:
            true_neighbor_tail[head] = np.array(list(true_neighbor_tail[head]))

        return true_neighbor_head, true_neighbor_tail

    @staticmethod
    def get_true_relation_neighbors(triples):
        """
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        """
        true_relation_head = defaultdict(set)
        true_relation_tail = defaultdict(set)

        for head, relation, tail in triples:
            true_relation_head[head].add(relation)
            true_relation_tail[tail].add(relation)

        for head in true_relation_head:
            true_relation_head[head] = np.array(list(true_relation_head[head]))
        for tail in true_relation_tail:
            true_relation_tail[tail] = np.array(list(true_relation_tail[tail]))

        return true_relation_head, true_relation_tail
    
    @staticmethod
    def filter_wildcard_triples(triples: List[Tuple[str]], wildcard_position:str , wildcard_value: int):
        """
        Filter triples based on a wildcard position and value.
        Prevents duplicate triples with wildcard values.
        """
        if wildcard_position == 'head':
            return list(set([(wildcard_value, relation, tail) for (_, relation, tail) in triples]))
        elif wildcard_position == 'relation':
            return list(set([(head, wildcard_value, tail) for (head, _, tail) in triples]))
        elif wildcard_position == 'tail':
            return list(set([(head, relation, wildcard_value) for (head, relation, _) in triples]))
        else:
            raise ValueError("Invalid wildcard position specified.")

class OneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data

class MultiTaskIterator(object):
    def __init__(self, dataloaders):
        self.iterators = {mode: self.one_shot_iterator(dataloader) for mode, dataloader in dataloaders}
        self.step = 0
        self.modes = list(self.iterators.keys())
        
    def __next__(self):
        self.step += 1
        mode = self.modes[self.step % len(self.modes)]
        data = next(self.iterators[mode])
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data

#----------------------------------------------

def build_type_constraints(triples):
    """
    Given triples, build implicit domain and range constraints for each relation.
    Returns:
        domain_constraints: relation → set of valid head entities
        range_constraints: relation → set of valid tail entities
    """
    domain_constraints = defaultdict(set)
    range_constraints = defaultdict(set)

    for h, r, t in triples:
        domain_constraints[r].add(h)
        range_constraints[r].add(t)

    # Convert to normal dicts for serialization safety
    return dict(domain_constraints), dict(range_constraints)

def build_neighbor_constraints(triples):
    """
    Given triples, build implicit neighbor constraints for each entity.
    Returns:
        neighbor_constraints: entity → set of valid neighboring entities
    """
    head_neighbor_constraints = defaultdict(set)
    tail_neighbor_constraints = defaultdict(set)

    for h, _, t in triples:
        head_neighbor_constraints[h].add(t)
        tail_neighbor_constraints[t].add(h) # assumes kg is undirected

    # Convert to normal dict for serialization safety
    return dict(head_neighbor_constraints), dict(tail_neighbor_constraints)

def build_neighbor_rel_constraints(triples):
    """ 
    Given triples, build implicit neighbor relation constraints for each entity.
    Returns:
        neighbor_rel_constraints: entity → set of valid neighboring relations
    """
    head_neighbor_rel_constraints = defaultdict(set)
    tail_neighbor_rel_constraints = defaultdict(set)

    for h, r, t in triples:
        head_neighbor_rel_constraints[h].add(r)
        tail_neighbor_rel_constraints[t].add(r)

    # Convert to normal dict for serialization safety
    return dict(head_neighbor_rel_constraints), dict(tail_neighbor_rel_constraints)