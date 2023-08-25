import torch
import torch_geometric
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv

from trainer import Model

@Model.register("GNN_ASAP_bn")
class GNN_ASAP_bn(Model):
    """
    Implements ASAP pooling
    """
    def __init__(self,
                  in_channel:int, 
                  hidden_channel:int, 
                  out_channel:int, 
                  num_gc_layers:int, 
                  batch_norm:bool=False, 
                  pooling:bool=False, 
                  **kwargs):
        """
        Parameters
        in_channel: Number of input features
        hidden_channel: Number of hidden features for MLP in GIN layers
        out_channel: Number of output features
        num_gc_layers: Number of graph convolutions layers in GNN encoder
        batch_norm: Set True to perform batch norm after each layer
        pooling: Set True to perform ASAP pooling
        """
        super(GNN_ASAP_bn, self).__init__()

        self.num_gc_layers = num_gc_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.pool = torch.nn.ModuleList()
        self.is_batch_norm = batch_norm
        self.is_pooling = pooling

        if self.is_pooling:
            hidden_channel = out_channel
        
        for i in range(num_gc_layers):

            if i==0:
                nn = Sequential(Linear(in_channel, hidden_channel), ReLU(), Linear(hidden_channel, hidden_channel))
                bn = torch.nn.BatchNorm1d(hidden_channel)
                layerpool = torch_geometric.nn.ASAPooling(in_channels=hidden_channel, ratio = 0.5, dropout=0.1)
            elif i==(num_gc_layers-1):
                nn = Sequential(Linear(hidden_channel, hidden_channel), ReLU(), Linear(hidden_channel, out_channel))
                bn = torch.nn.BatchNorm1d(out_channel)
                layerpool = torch_geometric.nn.ASAPooling(in_channels=out_channel, ratio = 0.5, dropout=0.1)
            else:
                nn = Sequential(Linear(hidden_channel, hidden_channel), ReLU(), Linear(hidden_channel, hidden_channel))
                bn = torch.nn.BatchNorm1d(hidden_channel)
                layerpool = torch_geometric.nn.ASAPooling(in_channels=hidden_channel, ratio = 0.5, dropout=0.1)


            conv = GINConv(nn, train_eps=True)

            self.convs.append(conv)
            self.bns.append(bn)
            self.pool.append(layerpool)
        #Final layer applying network
        self.embedder = Linear(out_channel+hidden_channel*(num_gc_layers-1),out_channel)

    def forward(self, x, edge_index, batch):
        xs = []
        local_emb = []
        xpool_list = []
        batch_mod = []
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(edge_index.device)
            x = x.to(edge_index.device)

        for i in range(self.num_gc_layers):

            x = self.convs[i](x, edge_index)
            if self.is_pooling:
                x, edge_index, edge_weight, batch, perm = self.pool[i](x,edge_index,batch=batch)
            if self.is_batch_norm:
                x = self.bns[i](x)
            x = F.relu(x)
            if self.is_pooling:
                local_emb.append(x)
                batch_mod.append(batch)
            else:
                xs.append(x)
            xpool_list.append(torch_geometric.nn.global_mean_pool(x, batch))

        x = self.embedder(torch.cat(xpool_list, 1))
        
        if self.is_pooling:
            local_emb = torch.cat(local_emb,0)
            batch_mod = torch.cat(batch_mod,0)
            return x, local_emb, batch_mod
        else:
            return x, self.embedder(torch.cat(xs, 1)), batch