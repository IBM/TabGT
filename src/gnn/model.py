import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, PNAConv
import torch.nn.functional as F
import torch

from .config import GNNConfig


class GINe(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_hidden = config.n_hidden
        self.num_gnn_layers = config.n_gnn_layers
        self.final_dropout = config.final_dropout
        self.n_classes = config.n_classes

        self.node_emb = nn.Linear(config.n_node_feats, config.n_hidden)
        self.edge_emb = nn.Linear(config.n_edge_feats, config.n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = GINEConv(nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden), 
                nn.ReLU(), 
                nn.Linear(self.n_hidden, self.n_hidden)
                ), edge_dim=self.n_hidden)
            
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.n_hidden))

        self.readout = LinkPredHead(config)

    def forward(self, x, edge_index, edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        neg_edge_attr = self.edge_emb(neg_edge_attr)
        pos_edge_attr = self.edge_emb(pos_edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
        
        out = self.readout(x, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
        
        return out, x
    
    def loss_fn(self, input1, input2):
        # input 1 is pos_preds and input_2 is neg_preds
        return -torch.log(input1 + 1e-12).mean() - torch.log(1 - input2 + 1e-12).mean()
    
    
class PNA(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_hidden = config.n_hidden
        self.n_hidden = int((self.n_hidden // 5) * 5)
        self.num_gnn_layers = config.n_gnn_layers
        self.final_dropout = config.final_dropout
        self.n_classes = config.n_classes
        self.deg = config.deg

        self.node_emb = nn.Linear(config.n_node_feats, config.n_hidden)
        self.edge_emb = nn.Linear(config.n_edge_feats, config.n_hidden)


        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = PNAConv(in_channels=self.n_hidden, out_channels=self.n_hidden,
                           aggregators=aggregators, scalers=scalers, deg=self.deg,
                           edge_dim=self.n_hidden, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.n_hidden))

        self.mlp = nn.Sequential(Linear(self.n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                              Linear(25, self.n_classes))

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2

        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        out = x
        return self.mlp(out)
    
    def loss_fn(self, input1, input2):
        # input 1 is pos_preds and input_2 is neg_preds
        return -torch.log(input1 + 1e-15).mean() - torch.log(1 - input2 + 1e-15).mean()

class LinkPredHead(torch.nn.Module):
    def __init__(self, config: GNNConfig) -> None:
        super().__init__()
        self.n_hidden = config.n_hidden
        self.n_classes = config.n_classes
        self.final_dropout = config.final_dropout

        self.mlp = nn.Sequential(Linear(self.n_hidden*3, self.n_hidden), nn.ReLU(), nn.Dropout(self.final_dropout), Linear(self.n_hidden, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                            Linear(25, self.n_classes))
            
    def forward(self, x, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr):
        # print(f'{pos_edge_index=}')
        # print(f'{neg_edge_index=}')
        #reshape s.t. each row in x corresponds to the concatenated src and dst node features for each edge
        x_pos = x[pos_edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x_neg = x[neg_edge_index.T].reshape(-1, 2 * self.n_hidden).relu()

        #concatenate the node feature vector with the corresponding edge features
        x_pos = torch.cat((x_pos, pos_edge_attr.view(-1, pos_edge_attr.shape[1])), 1)
        x_neg = torch.cat((x_neg, neg_edge_attr.view(-1, neg_edge_attr.shape[1])), 1)

        return (torch.sigmoid(self.mlp(x_pos)), torch.sigmoid(self.mlp(x_neg)))