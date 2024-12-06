"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Fact scoring networks.
 Code adapted from https://github.com/TimDettmers/ConvE/blob/master/model.py
"""

import copy
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor
from src.knowledge_graph import KnowledgeGraph
from src.utils.logs import create_logger
import pdb


class TripleE(nn.Module):
    def __init__(self, args, num_entities):
        super(TripleE, self).__init__()
        conve_args = copy.deepcopy(args)
        conve_args.model = "conve"
        self.conve_nn = ConvE(conve_args, num_entities)
        conve_state_dict = torch.load(args.conve_state_dict_path)
        conve_nn_state_dict = get_conve_nn_state_dict(conve_state_dict)
        self.conve_nn.load_state_dict(conve_nn_state_dict)

        complex_args = copy.deepcopy(args)
        complex_args.model = "complex"
        self.complex_nn = ComplEx(complex_args)

        distmult_args = copy.deepcopy(args)
        distmult_args.model = "distmult"
        self.distmult_nn = DistMult(distmult_args)

    def forward(self, e1, r, conve_kg, secondary_kgs):
        complex_kg = secondary_kgs[0]
        distmult_kg = secondary_kgs[1]
        return (
            self.conve_nn.forward(e1, r, conve_kg)
            + self.complex_nn.forward(e1, r, complex_kg)
            + self.distmult_nn.forward(e1, r, distmult_kg)
        ) / 3

    def forward_fact(self, e1, r, conve_kg, secondary_kgs):
        complex_kg = secondary_kgs[0]
        distmult_kg = secondary_kgs[1]
        return (
            self.conve_nn.forward_fact(e1, r, conve_kg)
            + self.complex_nn.forward_fact(e1, r, complex_kg)
            + self.distmult_nn.forward_fact(e1, r, distmult_kg)
        ) / 3


class HyperE(nn.Module):
    def __init__(self, args, num_entities):
        super(HyperE, self).__init__()
        self.conve_nn = ConvE(args, num_entities)
        conve_state_dict = torch.load(args.conve_state_dict_path)
        conve_nn_state_dict = get_conve_nn_state_dict(conve_state_dict)
        self.conve_nn.load_state_dict(conve_nn_state_dict)

        complex_args = copy.deepcopy(args)
        complex_args.model = "complex"
        self.complex_nn = ComplEx(complex_args)

    def forward(self, e1, r, conve_kg, secondary_kgs):
        complex_kg = secondary_kgs[0]
        return (
            self.conve_nn.forward(e1, r, conve_kg)
            + self.complex_nn.forward(e1, r, complex_kg)
        ) / 2

    def forward_fact(self, e1, r, e2, conve_kg, secondary_kgs):
        complex_kg = secondary_kgs[0]
        return (
            self.conve_nn.forward_fact(e1, r, e2, conve_kg)
            + self.complex_nn.forward_fact(e1, r, e2, complex_kg)
        ) / 2


# ------------------------------------------------------------------------------
"Original Models with Modifications"


class ComplEx(nn.Module):
    def __init__(self, args):
        super(ComplEx, self).__init__()

    def forward(self, e1: Tensor, r: Tensor, kg: KnowledgeGraph) -> [Tensor, Tensor]:
        # Compute the displacement from E1 using relation R
        E1_real = kg.get_entity_embeddings(e1)
        R_real = kg.get_relation_embeddings(r)
        E1_img = kg.get_entity_img_embeddings(e1)
        R_img = kg.get_relation_img_embeddings(r)

        # Compute the approximate tail entity (displacement) for both real and imaginary parts
        E2_real_approx, E2_img_approx = self.forward_displacement(
            E1_real, R_real, E1_img, R_img
        )

        return E2_real_approx, E2_img_approx

    def forward_displacement(
        self, E1_real: Tensor, R_real: Tensor, E1_img: Tensor, R_img: Tensor
    ) -> [Tensor, Tensor]:
        """
        Compute the displacement of the head entity along the relation vector.
        .. math::
            \mathbf{e}_t \approx \mathbf{e}_h \circ \mathbf{e}_r

        Parameters:
        E1_real (torch.Tensor): Real part of the head entity embedding (batch_size, embedding_dim).
        R_real (torch.Tensor): Real part of the relation embedding (batch_size, embedding_dim).
        E1_img (torch.Tensor): Imaginary part of the head entity embedding (batch_size, embedding_dim).
        R_img (torch.Tensor): Imaginary part of the relation embedding (batch_size, embedding_dim).

        Returns:
        torch.Tensor: Real and imaginary parts of the approximate tail entity embedding.
        """
        # Compute the approximate real part of the tail entity
        E2_real_approx = E1_real * R_real - E1_img * R_img

        # Compute the approximate imaginary part of the tail entity
        E2_img_approx = E1_real * R_img + E1_img * R_real

        return E2_real_approx, E2_img_approx

    def dist_mult_func(E1: Tensor, R: Tensor, E2: Tensor) -> Tensor:
        return torch.mm(E1 * R, E2.transpose(1, 0))

    def forward_original(self, e1, r, kg):
        E1_real = kg.get_entity_embeddings(e1)
        R_real = kg.get_relation_embeddings(r)
        E2_real = kg.get_all_entity_embeddings()
        E1_img = kg.get_entity_img_embeddings(e1)
        R_img = kg.get_relation_img_embeddings(r)
        E2_img = kg.get_all_entity_img_embeddings()

        rrr = self.dist_mult_func(R_real, E1_real, E2_real)
        rii = self.dist_mult_func(R_real, E1_img, E2_img)
        iri = self.dist_mult_func(R_img, E1_real, E2_img)
        iir = self.dist_mult_func(R_img, E1_img, E2_real)
        S = rrr + rii + iri - iir
        S = F.sigmoid(S)
        return S

    def forward_fact(self, e1, r, e2, kg):
        def dist_mult_fact(E1, R, E2):
            return torch.sum(E1 * R * E2, dim=1, keepdim=True)

        E1_real = kg.get_entity_embeddings(e1)
        R_real = kg.get_relation_embeddings(r)
        E2_real = kg.get_entity_embeddings(e2)
        E1_img = kg.get_entity_img_embeddings(e1)
        R_img = kg.get_relation_img_embeddings(r)
        E2_img = kg.get_entity_img_embeddings(e2)

        rrr = dist_mult_fact(R_real, E1_real, E2_real)
        rii = dist_mult_fact(R_real, E1_img, E2_img)
        iri = dist_mult_fact(R_img, E1_real, E2_img)
        iir = dist_mult_fact(R_img, E1_img, E2_real)
        S = rrr + rii + iri - iir
        S = F.sigmoid(S)
        return S


class ConvE(nn.Module):
    def __init__(self, args, num_entities):
        super(ConvE, self).__init__()
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        assert args.emb_2D_d1 * args.emb_2D_d2 == args.entity_dim
        assert args.emb_2D_d1 * args.emb_2D_d2 == args.relation_dim
        self.emb_2D_d1 = args.emb_2D_d1
        self.emb_2D_d2 = args.emb_2D_d2
        self.num_out_channels = args.num_out_channels
        self.w_d = args.kernel_size
        self.HiddenDropout = nn.Dropout(args.hidden_dropout_rate)
        self.FeatureDropout = nn.Dropout(args.feat_dropout_rate)

        # stride = 1, padding = 0, dilation = 1, groups = 1
        self.conv1 = nn.Conv2d(1, self.num_out_channels, (self.w_d, self.w_d), 1, 0)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(self.num_out_channels)
        self.bn2 = nn.BatchNorm1d(self.entity_dim)
        self.register_parameter("b", nn.Parameter(torch.zeros(num_entities)))
        h_out = 2 * self.emb_2D_d1 - self.w_d + 1
        w_out = self.emb_2D_d2 - self.w_d + 1
        self.feat_dim = self.num_out_channels * h_out * w_out
        self.fc = nn.Linear(self.feat_dim, self.entity_dim)

    # TOREM: Probably not used
    def forward_geometric_maybe(
        self, e1: Tensor, r: Tensor, kg: KnowledgeGraph
    ) -> Tensor:
        # Compute the displacement from E1 using relation R
        E1 = kg.get_entity_embeddings(e1).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        R = kg.get_relation_embeddings(r).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)

        # Compute the approximate tail entity embedding
        E2_approx = self.forward_displacement(E1, R)

        return E2_approx

    def forward_displacement(self, E1: Tensor, R: Tensor) -> Tensor:
        """
        Compute the displacement of the head entity along the relation vector.

        Parameters:
        E1 (torch.Tensor): Head entity embedding (batch_size, 1, emb_2D_d1, emb_2D_d2).
        R (torch.Tensor): Relation embedding (batch_size, 1, emb_2D_d1, emb_2D_d2).

        Returns:
        torch.Tensor: Approximate tail entity embedding.
        """
        # Concatenate head and relation embeddings along the second dimension
        stacked_inputs = torch.cat([E1, R], 2)
        stacked_inputs = self.bn0(stacked_inputs)

        # Apply convolution, activation, dropout, and fully connected layer
        X = self.conv1(stacked_inputs)
        X = F.relu(X)
        X = self.FeatureDropout(X)
        X = X.view(-1, self.feat_dim)
        X = self.fc(X)
        X = self.HiddenDropout(X)
        X = self.bn2(X)
        X = F.relu(X)

        """
        NOTE: Not sure with this one
        """
        # The result is the approximate tail entity embedding
        return X

    def forward(self, e1, r, kg: KnowledgeGraph):
        E1 = kg.get_entity_embeddings(e1).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        R = kg.get_relation_embeddings(r).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        E2 = kg.get_all_entity_embeddings()

        stacked_inputs = torch.cat([E1, R], 2)
        stacked_inputs = self.bn0(stacked_inputs)

        X = self.conv1(stacked_inputs)
        # X = self.bn1(X)
        X = F.relu(X)
        X = self.FeatureDropout(X)
        X = X.view(-1, self.feat_dim)
        X = self.fc(X)
        X = self.HiddenDropout(X)
        X = self.bn2(X)
        X = F.relu(X)
        X = torch.mm(X, E2.transpose(1, 0))
        X += self.b.expand_as(X)

        S = F.sigmoid(X)
        return S

    def forward_fact(self, e1, r, e2, kg):
        """
        Compute network scores of the given facts.
        :param e1: [batch_size]
        :param r:  [batch_size]
        :param e2: [batch_size]
        :param kg:
        """
        # print(e1.size(), r.size(), e2.size())
        # print(e1.is_contiguous(), r.is_contiguous(), e2.is_contiguous())
        # print(e1.min(), r.min(), e2.min())
        # print(e1.max(), r.max(), e2.max())
        E1 = kg.get_entity_embeddings(e1).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        R = kg.get_relation_embeddings(r).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        E2 = kg.get_entity_embeddings(e2)

        stacked_inputs = torch.cat([E1, R], 2)
        stacked_inputs = self.bn0(stacked_inputs)

        X = self.conv1(stacked_inputs)
        # X = self.bn1(X)
        X = F.relu(X)
        X = self.FeatureDropout(X)
        X = X.view(-1, self.feat_dim)
        X = self.fc(X)
        X = self.HiddenDropout(X)
        X = self.bn2(X)
        X = F.relu(X)
        X = torch.matmul(X.unsqueeze(1), E2.unsqueeze(2)).squeeze(2)
        X += self.b[e2].unsqueeze(1)

        S = F.sigmoid(X)
        return S


class DistMult(nn.Module):
    def __init__(self, args):
        super(DistMult, self).__init__()

    def forward(self, e1: Tensor, r: Tensor, kg: KnowledgeGraph) -> Tensor:
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        return self.forward_displacement(E1, R)

    def forward_displacement(self, E1: Tensor, R: Tensor) -> Tensor:
        """
        Compute the displacement of the head entity along the relation vector.
        Displacement: \mathbf{e}_t  \approx \mathbf{e}_h \circ \mathbf{e}_r

        Parameters:
        E1 (torch.Tensor): Embedding of the head entity (batch_size, embedding_dim).
        R (torch.Tensor): Embedding of the relation (batch_size, embedding_dim).

        Returns:
        torch.Tensor: Displacement embedding, representing the expected tail entity.
        """
        # Compute the approximate tail entity as the displacement from E1 using R
        E2_approx = E1 * R  # Element-wise product for DisMult logic
        return E2_approx

    def forward_original(self, e1, r, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_all_entity_embeddings()
        S = torch.mm(E1 * R, E2.transpose(1, 0))
        # Predicts Entities
        S = F.sigmoid(S)
        return S

    def forward_fact(self, e1, r, e2, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_entity_embeddings(e2)
        S = torch.sum(E1 * R * E2, dim=1, keepdim=True)
        S = F.sigmoid(S)
        return S


# ------------------------------------------------------------------------------
"Additional Models"


class TransE(nn.Module):
    """
    Based on : https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/kge/transe.html#TransE
    """

    def __init__(
        self,
        margin: float = 1.0,
        p_norm: float = 1.0,
    ):
        super(TransE, self).__init__()
        self.p_norm = p_norm
        self.margin = margin

    def forward(
        self, head: LongTensor, tail: LongTensor, r: LongTensor, kg: KnowledgeGraph
    ) -> Tensor:
        head_embeddings = kg.get_entity_embeddings(head)
        rel_embeddings = kg.get_relation_embeddings(r)
        tail_embeddings = kg.get_entity_embeddings(tail)

        head_embeddings = F.normalize(head_embeddings, p=self.p_norm, dim=-1)
        tail_embeddings = F.normalize(tail_embeddings, p=self.p_norm, dim=-1)

        # NOTE: Not sure why the rel is not doing the same.

        return -((head_embeddings + rel_embeddings) - tail_embeddings).norm(
            p=self.p_norm, dim=-1
        )

    def forward_for_score(
        self, heads: LongTensor, rs: LongTensor, kg: KnowledgeGraph
    ) -> Tensor:
        """
        above but for every head given it will multiply against every entity to see which ones are the best ones.
        """
        # Ensure nothing much is being collected:
        with torch.no_grad():
            all_possible_tails = kg.get_all_entity_embeddings().unsqueeze(
                0
            )  # For all the bataches
            head_embeddings = kg.get_entity_embeddings(heads)
            # head_embeddings[SHAPE] = (batch_size, num_entities, entity_dim)
            mult_head_embeddings = head_embeddings.unsqueeze(1)
            # mult_head_embeddings[SHAPE] = (batch_size, all_possible_tails.size(0), entity_dim)

            rel_embeddings = kg.get_relation_embeddings(rs).unsqueeze(1)
            # Then at this point we can use broadcasting to do the same operation as in `forwad` but across all these dimensions
            calculated_score = -(
                (mult_head_embeddings + rel_embeddings) - all_possible_tails
            ).norm(p=self.p_norm, dim=-1)

        return calculated_score

    # Note: Not really used since parameters are set in KG, but for reference.
    def reset_parameters(self):
        bound = 6.0 / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        F.normalize(
            self.rel_emb.weight.data,
            p=self.p_norm,
            dim=-1,
            out=self.rel_emb.weight.data,
        )

    def forward_displacement(self, E1: Tensor, R: Tensor) -> Tensor:
        """
        Compute the displacement of the head entity along the relation vector.
        .. math::
            \mathbf{e}_t \approx \mathbf{e}_h + \mathbf{e}_r

        Parameters:
        E1 (torch.Tensor): Embedding of the head entity (batch_size, embedding_dim).
        R (torch.Tensor): Embedding of the relation (batch_size, embedding_dim).

        Returns:
        torch.Tensor: Displacement embedding, representing the expected tail entity.
        """
        return E1 + R


class RotatE(nn.Module):
    def __init__(self, margin: float = 1, p_norm: float = 1):
        super(RotatE, self).__init__()
        self.margin = margin
        self.p_norm = p_norm
        self.logger = create_logger(self.__class__.__name__)

    def forward(
        self, head: Tensor, tails: Tensor, relation: Tensor, kg: KnowledgeGraph
    ) -> torch.Tensor:
        # Compute the displacement from E1 using relation R
        head_real = kg.get_entity_embeddings(head)
        head_img = kg.get_entity_img_embeddings(head)
        tail_real = kg.get_entity_embeddings(tails)
        tail_img = kg.get_entity_img_embeddings(tails)

        rel_theta = kg.get_relation_embeddings(relation)
        rel_real, rel_img = torch.cos(rel_theta), torch.sin(rel_theta)

        # And fromm this we continue let me see something really quick

        # Compute the approximate tail entity (displacement) for both real and imaginary parts
        tail_approx_real, tail_approx_img = self.forward_displacement(
            head_real, rel_real, head_img, rel_img
        )

        score = torch.norm(
            torch.stack([tail_approx_real - tail_real, tail_approx_img - tail_img], dim=-1), dim=-1
        )

        # score = torch.hypot(
        #     (tail_approx_real - tail_real),
        #     (tail_approx_img - tail_img)
        # )

        if torch.any(torch.isnan(score)): 
            pdb.set_trace() 
            pass

        return score

    def forward_displacement(
        self, head_real: Tensor, rel_real: Tensor, head_img: Tensor, rel_img: Tensor
    ):
        """
        compute the displacement of the head entity along the relation vector.
        .. math::
            \mathbf{e}_t \approx \mathbf{e}_h \circ \mathbf{e}_r,

        parameters:
        e1_real (torch.tensor): real part of the head entity embedding (batch_size, embedding_dim).
        r_real (torch.tensor): real part of the relation embedding (batch_size, embedding_dim).
        e1_img (torch.tensor): imaginary part of the head entity embedding (batch_size, embedding_dim).
        r_img (torch.tensor): imaginary part of the relation embedding (batch_size, embedding_dim).

        returns:
        torch.tensor: real and imaginary parts of the approximate tail entity embedding.
        """
        # compute the approximate real part of the tail entity
        e2_real_approx = head_real * rel_real - head_img * rel_img

        # compute the approximate imaginary part of the tail entity
        e2_img_approx = head_real * rel_img + head_img * rel_real

        return e2_real_approx, e2_img_approx

    def forward_for_score(
        self, heads: LongTensor, rs: LongTensor, kg: KnowledgeGraph
    ) -> Tensor:
        """
        above but for every head given it will multiply against every entity to see which ones are the best ones.
        """
        # Ensure nothing much is being collected:
        with torch.no_grad():
            all_possible_tails = kg.get_all_entity_embeddings().unsqueeze(
                0
            )  # For all the bataches
            calculated_score = self(
                heads, all_possible_tails, rs, kg
            )

        return calculated_score




def get_conve_nn_state_dict(state_dict):
    conve_nn_state_dict = {}
    for param_name in [
        "mdl.b",
        "mdl.conv1.weight",
        "mdl.conv1.bias",
        "mdl.bn0.weight",
        "mdl.bn0.bias",
        "mdl.bn0.running_mean",
        "mdl.bn0.running_var",
        "mdl.bn1.weight",
        "mdl.bn1.bias",
        "mdl.bn1.running_mean",
        "mdl.bn1.running_var",
        "mdl.bn2.weight",
        "mdl.bn2.bias",
        "mdl.bn2.running_mean",
        "mdl.bn2.running_var",
        "mdl.fc.weight",
        "mdl.fc.bias",
    ]:
        conve_nn_state_dict[param_name.split(".", 1)[1]] = state_dict["state_dict"][
            param_name
        ]
    return conve_nn_state_dict


def get_conve_kg_state_dict(state_dict):
    kg_state_dict = dict()
    for param_name in ["kg.entity_embeddings.weight", "kg.relation_embeddings.weight"]:
        kg_state_dict[param_name.split(".", 1)[1]] = state_dict["state_dict"][
            param_name
        ]
    return kg_state_dict


def get_complex_kg_state_dict(state_dict):
    kg_state_dict = dict()
    for param_name in [
        "kg.entity_embeddings.weight",
        "kg.relation_embeddings.weight",
        "kg.entity_img_embeddings.weight",
        "kg.relation_img_embeddings.weight",
    ]:
        kg_state_dict[param_name.split(".", 1)[1]] = state_dict["state_dict"][
            param_name
        ]
    return kg_state_dict


def get_distmult_kg_state_dict(state_dict):
    kg_state_dict = dict()
    for param_name in ["kg.entity_embeddings.weight", "kg.relation_embeddings.weight"]:
        kg_state_dict[param_name.split(".", 1)[1]] = state_dict["state_dict"][
            param_name
        ]
    return kg_state_dict
