# --------------------------------------------------------------------------------------------------------------------------
# Description: This script is about deep graph infomax
# Majoritiy of the code adapted from the implementation given at the pytorch geometric code base
# --------------------------------------------------------------------------------------------------------------------------
#

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ..models import Discriminator_bn

import trainer
from trainer import Model
EPS = 1e-15

@trainer.Model.register("bracs_disc")
class DeepGraphInfomax_disc(trainer.Model):
    r"""
    Modified version of Deep graph infomax but at the core same principles
    Tries for good graph level representation
    Everything is same as bracs, however here i am using different discriminator function
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_channels = kwargs["out_channel"]
        model_name = kwargs.pop("model_name")
        
        self.encoder = Model.create(subclass_name=model_name,**kwargs)
        self.kwargs = kwargs

        self.disc_local = Discriminator_bn(self.hidden_channels)
        self.disc_global = Discriminator_bn(self.hidden_channels)

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch):
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""

        graph_embeddings , all_embeddings, batch = self.encoder(x = x, edge_index = edge_index, batch=batch)

        #get encoded values for discriminator
        global_emb = self.disc_global(graph_embeddings)
        local_emb = self.disc_local(all_embeddings)

        return local_emb, global_emb, batch

    def discriminate(self, z, summary, sigmoid=True):
        """Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.
        Args:
            z (Tensor): The latent space.
            summary (Tensor): The summary vector.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, summary)
        # return torch.sigmoid(value) if sigmoid else value
        return value

    def loss(self, node_emb, summary, batch):
        """Computes the mutual information maximization objective."""
        num_graphs = summary.shape[0]
        num_nodes = node_emb.shape[0]

        pos_mask = torch.zeros((num_nodes, num_graphs)).to(node_emb.device)
        neg_mask = torch.ones((num_nodes, num_graphs)).to(node_emb.device)
        for nodeidx, graphidx in enumerate(batch):
            pos_mask[nodeidx][graphidx] = 1.
            neg_mask[nodeidx][graphidx] = 0.

        res = torch.mm(node_emb, summary.t())
            
        E_pos = -nn.functional.softplus(-res * pos_mask).mean()
        neg_value = res * neg_mask
        E_neg = (nn.functional.softplus(-res * neg_mask) + neg_value).mean()

        return E_neg - E_pos

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.hidden_channels})'

    def reset(self, value: Any):
        if hasattr(value, 'reset_parameters'):
            value.reset_parameters()
        else:
            for child in value.children() if hasattr(value, 'children') else []:
                self.reset(child)

    def uniform(self, size: int, value: Any):
        if isinstance(value, Tensor):
            bound = 1.0 / math.sqrt(size)
            value.data.uniform_(-bound, bound)
        else:
            for v in value.parameters() if hasattr(value, 'parameters') else []:
                self.uniform(size, v)
            for v in value.buffers() if hasattr(value, 'buffers') else []:
                self.uniform(size, v)

    @property
    def get_encoder(self):
        return self.encoder
