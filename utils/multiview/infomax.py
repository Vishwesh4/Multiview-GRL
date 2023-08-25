#
# --------------------------------------------------------------------------------------------------------------------------
# Description: This script is about multiview deep graph infomax incorporating two graph views
# Code adapted from pytorch geometric code base
# --------------------------------------------------------------------------------------------------------------------------
#
from typing import Any

import torch
import torch.nn as nn
import numpy as np
import trainer
from trainer import Model

from ..models import Discriminator_bn

@trainer.Model.register("multiview_contrast")
class Multiview_Infomax_contrast(Model):
    def __init__(self, **kwargs) -> None:
        super(Model, self).__init__()
        cellgraph_params = kwargs["cellgraph_params"]
        patchgraph_params = kwargs["patchgraph_params"]
        self.out_channel = kwargs["out_channel"]
        cellgraph_name = cellgraph_params.pop("cellgraph_name")
        patchgraph_name = patchgraph_params.pop("patchgraph_name")
        cell_out = cellgraph_params["out_channel"]
        patch_out = patchgraph_params["out_channel"]

        self.cell_encoder = Model.create(subclass_name=cellgraph_name,**cellgraph_params)
        self.patch_encoder = Model.create(subclass_name=patchgraph_name,**patchgraph_params)

        #Defining descriminators, assuming all have same embedding dimensions
        self.disc_local_cell = Discriminator_bn(cell_out)
        self.disc_local_patch = Discriminator_bn(patch_out)
        self.disc_global_cell = Discriminator_bn(self.out_channel)
        self.disc_global_patch = Discriminator_bn(self.out_channel)

        self.init_emb()
    
    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data_cell, data_patch, device):
        #For cell graph
        x, edge_index, batch_cell = data_cell.x.to(device), data_cell.edge_index.to(device), data_cell.batch.to(device)
        graph_embeddings_cell , all_embeddings, batch_cell = self.cell_encoder(x = x, edge_index = edge_index, batch=batch_cell)
        #get encoded values for discriminator
        global_emb_cell = self.disc_global_cell(graph_embeddings_cell)
        local_emb_cell = self.disc_local_cell(all_embeddings)

        #For patch graph
        x, edge_index, batch_patch = data_patch.x.to(device), data_patch.edge_index.to(device), data_patch.batch.to(device)
        graph_embeddings_patch , all_embeddings, batch_patch = self.patch_encoder(x = x, edge_index = edge_index, batch=batch_patch)
        #get encoded values for discriminator
        global_emb_patch = self.disc_global_patch(graph_embeddings_patch)
        local_emb_patch = self.disc_local_patch(all_embeddings)

        return (local_emb_cell,local_emb_patch), (global_emb_cell,global_emb_patch), (batch_cell,batch_patch)

    def loss(self, node_emb_tup, summary_tup, batch_tup):
        """Computes the mutual information maximization objective."""
        node_emb_cell, node_emb_patch = node_emb_tup
        batch_cell, batch_patch = batch_tup
        summary_cell, summary_patch = summary_tup
        num_graphs = summary_cell.shape[0]
        
        #Cell graph
        num_nodes = node_emb_cell.shape[0]
        pos_mask = torch.zeros((num_nodes, num_graphs)).to(node_emb_cell.device)
        neg_mask = torch.ones((num_nodes, num_graphs)).to(node_emb_cell.device)
        for nodeidx, graphidx in enumerate(batch_cell):
            pos_mask[nodeidx][graphidx] = 1.
            neg_mask[nodeidx][graphidx] = 0.

        ### CELL GRAPH
        #cell vs cell comparison
        res = torch.mm(node_emb_cell, summary_cell.t())            
        E_pos_cell = -nn.functional.softplus(-res * pos_mask).mean()
        neg_value = res * neg_mask
        E_neg_cell = (nn.functional.softplus(-res * neg_mask) + neg_value).mean()
        E_cc = E_neg_cell - E_pos_cell
        #patch vs cell comparison
        res = torch.mm(node_emb_cell, summary_patch.t())
        E_pos_cell = -nn.functional.softplus(-res * pos_mask).mean()
        neg_value = res * neg_mask
        E_neg_cell = (nn.functional.softplus(-res * neg_mask) + neg_value).mean()
        E_cp = E_neg_cell - E_pos_cell

        ### PATCH GRAPH
        num_nodes = node_emb_patch.shape[0]
        pos_mask = torch.zeros((num_nodes, num_graphs)).to(node_emb_patch.device)
        neg_mask = torch.ones((num_nodes, num_graphs)).to(node_emb_patch.device)
        for nodeidx, graphidx in enumerate(batch_patch):
            pos_mask[nodeidx][graphidx] = 1.
            neg_mask[nodeidx][graphidx] = 0.

        #patch vs patch comparison
        res = torch.mm(node_emb_patch, summary_patch.t())
        E_pos_patch = -nn.functional.softplus(-res * pos_mask).mean()
        neg_value = res * neg_mask
        E_neg_patch = (nn.functional.softplus(-res * neg_mask) + neg_value).mean()        
        E_pp = E_neg_patch - E_pos_patch
        #cell vs patch comparison
        res = torch.mm(node_emb_patch, summary_cell.t())
        E_pos_patch = -nn.functional.softplus(-res * pos_mask).mean()
        neg_value = res * neg_mask
        E_neg_patch = (nn.functional.softplus(-res * neg_mask) + neg_value).mean()        
        E_pc = E_neg_patch - E_pos_patch
        
        return E_cc+E_cp+E_pc+E_pp

    def get_representation(self, data_cell, data_patch, device,type="patch"):
        #Just return patch graph
        if type=="patch":
            x, edge_index, batch_patch = data_patch.x.to(device), data_patch.edge_index.to(device), data_patch.batch.to(device)
            graph_embeddings, _, _ = self.patch_encoder(x = x, edge_index = edge_index, batch=batch_patch)
        else:
            x, edge_index, batch = data_cell.x.to(device), data_cell.edge_index.to(device), data_cell.batch.to(device)
            graph_embeddings, _, _ = self.cell_encoder(x = x, edge_index = edge_index, batch=batch)      
        return graph_embeddings

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.hidden_channels})'

    def reset(self, value: Any):
        if hasattr(value, 'reset_parameters'):
            value.reset_parameters()
        else:
            for child in value.children() if hasattr(value, 'children') else []:
                self.reset(child)


@trainer.Model.register("multiview_contrast_sample_v2")
class Multiview_Infomax_contrast_sampler_v2(Multiview_Infomax_contrast, Model):
    def __init__(self,num_samples=500,proc_percent=0.2,sample_proc="top", **kwargs) -> None:
        """
        Choose sample procedure from top/bottom
        """
        super(Model, self).__init__()
        cellgraph_params = kwargs["cellgraph_params"]
        patchgraph_params = kwargs["patchgraph_params"]
        self.out_channel = kwargs["out_channel"]
        cellgraph_name = cellgraph_params.pop("cellgraph_name")
        patchgraph_name = patchgraph_params.pop("patchgraph_name")
        cell_out = cellgraph_params["out_channel"]
        patch_out = patchgraph_params["out_channel"]

        self.cell_encoder = Model.create(subclass_name=cellgraph_name,**cellgraph_params)
        self.patch_encoder = Model.create(subclass_name=patchgraph_name,**patchgraph_params)
        self.num_samples = num_samples
        self.sample_proc = sample_proc
        self.proc_percent = proc_percent

        #Defining descriminators, assuming all have same embedding dimensions
        self.disc_local_cell = Discriminator_bn(cell_out)
        self.disc_local_patch = Discriminator_bn(patch_out)
        self.disc_global_cell = Discriminator_bn(self.out_channel)
        self.disc_global_patch = Discriminator_bn(self.out_channel)

        self.init_emb()

    def loss(self, node_emb_tup, summary_tup, batch_tup):
        """Computes the mutual information maximization objective."""
        node_emb_cell, node_emb_patch = node_emb_tup
        batch_cell, batch_patch = batch_tup
        summary_cell, summary_patch = summary_tup
        num_graphs = summary_cell.shape[0]
        
        #Cell graph
        num_nodes = node_emb_cell.shape[0]
        pos_mask = torch.zeros((num_nodes, num_graphs)).to(node_emb_cell.device)
        neg_mask = torch.ones((num_nodes, num_graphs)).to(node_emb_cell.device)
        for nodeidx, graphidx in enumerate(batch_cell):
            pos_mask[nodeidx][graphidx] = 1.
            neg_mask[nodeidx][graphidx] = 0.

        ### CELL GRAPH
        #cell vs cell comparison
        res = torch.mm(node_emb_cell, summary_cell.t())            
        E_pos_cell = -nn.functional.softplus(-res * pos_mask).mean()
        neg_value = res * neg_mask
        E_neg_cell = (nn.functional.softplus(-res * neg_mask) + neg_value).mean()
        E_cc = E_neg_cell - E_pos_cell
        
        res = torch.mm(node_emb_cell, summary_patch.t())
        node_emb_cell_sample, pos_mask, neg_mask = self.sample_nodes(res,node_emb_cell,batch_cell,num_graphs)
        #patch vs cell comparison
        res = torch.mm(node_emb_cell_sample, summary_patch.t())
        E_pos_cell = -nn.functional.softplus(-res * pos_mask).mean()
        neg_value = res * neg_mask
        E_neg_cell = (nn.functional.softplus(-res * neg_mask) + neg_value).mean()
        E_cp = E_neg_cell - E_pos_cell

        ### PATCH GRAPH
        num_nodes = node_emb_patch.shape[0]
        pos_mask = torch.zeros((num_nodes, num_graphs)).to(node_emb_patch.device)
        neg_mask = torch.ones((num_nodes, num_graphs)).to(node_emb_patch.device)
        for nodeidx, graphidx in enumerate(batch_patch):
            pos_mask[nodeidx][graphidx] = 1.
            neg_mask[nodeidx][graphidx] = 0.

        #patch vs patch comparison
        res = torch.mm(node_emb_patch, summary_patch.t())
        E_pos_patch = -nn.functional.softplus(-res * pos_mask).mean()
        neg_value = res * neg_mask
        E_neg_patch = (nn.functional.softplus(-res * neg_mask) + neg_value).mean()        
        E_pp = E_neg_patch - E_pos_patch
        #cell vs patch comparison
        res = torch.mm(node_emb_patch, summary_cell.t())
        E_pos_patch = -nn.functional.softplus(-res * pos_mask).mean()
        neg_value = res * neg_mask
        E_neg_patch = (nn.functional.softplus(-res * neg_mask) + neg_value).mean()        
        E_pc = E_neg_patch - E_pos_patch
        
        return E_cc+E_cp+E_pc+E_pp

    def sample_nodes(self,res,node_embs,node_batch,num_graphs):
        node_emb_cell_sample = []
        batch_cell_sample = []
        for g in range(num_graphs):
            graph_idx = torch.where(node_batch==g)[0]
            loc_graph = node_embs[graph_idx,:]
            #Get the rankings
            if self.sample_proc=="top":
                _, idx = torch.sort(-nn.functional.softplus(-res[graph_idx,g]),dim=0,descending=True)
            elif self.sample_proc=="bottom":
                _, idx = torch.sort(-nn.functional.softplus(-res[graph_idx,g]),dim=0,descending=False)
            else:
                raise ValueError(f"Incorrect sample procedure given {self.sample_proc}, choose from top/bottom")
            n = int(self.proc_percent*self.num_samples)
            idx_sample_proc = idx[:n]
            idx_sample_random = idx[n + torch.randperm(np.max((len(idx)-n,0)))[:int((1-self.proc_percent)*self.num_samples)]]
            idx_sample = torch.concat((idx_sample_proc,idx_sample_random))
            #Select top n samples
            node_emb_cell_sample.append(loc_graph[idx_sample])
            batch_cell_sample.append(torch.ones(len(idx_sample),device=node_embs.device,dtype=torch.int64)*g)
        #Get the final matrix
        node_emb_cell_sample = torch.concat(node_emb_cell_sample,dim=0)
        batch_cell_sample = torch.concat(batch_cell_sample,dim=0)
        num_nodes = node_emb_cell_sample.shape[0]
        pos_mask = torch.zeros((num_nodes, num_graphs)).to(node_emb_cell_sample.device)
        neg_mask = torch.ones((num_nodes, num_graphs)).to(node_emb_cell_sample.device)
        for nodeidx, graphidx in enumerate(batch_cell_sample):
            pos_mask[nodeidx][graphidx] = 1.
            neg_mask[nodeidx][graphidx] = 0.
        return node_emb_cell_sample, pos_mask, neg_mask
