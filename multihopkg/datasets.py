from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

class TestDataset(Dataset):
    __test__ = False # To avoid pytest confusion

    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

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

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count, self.rel_count = self.count_frequency(triples)
        self.true_head, self.true_tail, self.true_rels = self.get_true_head_and_tail(self.triples)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample
        
        negative_sample_list = []
        negative_sample_size = 0

        if self.mode == 'head-batch' or self.mode == 'tail-batch':
            subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        else:
            subsampling_weight = self.rel_count[(head, tail)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        while negative_sample_size < self.negative_sample_size:
            if self.mode == 'head-batch':
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'tail-batch':
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'relation-batch':
                negative_sample = np.random.randint(self.nrelation, size=self.negative_sample_size * 2)
                mask = np.in1d(
                    negative_sample, 
                    self.true_rels[(head, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)

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
        
        true_head = {}
        true_tail = {}
        true_rels = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)
            if (head, tail) not in true_rels:
                true_rels[(head, tail)] = []
            true_rels[(head, tail)].append(relation)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))
        for head, tail in true_rels:
            true_rels[(head, tail)] = np.array(list(set(true_rels[(head, tail)])))                 

        return true_head, true_tail, true_rels

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