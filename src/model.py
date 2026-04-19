import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from layers import NMV_SAGE_Layer, SAGEConv_2

class NMV_Graph_Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NMV_Graph_Model, self).__init__()
        self.sage_block1 = NMV_SAGE_Layer(in_channels, hidden_channels)
        self.S = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        nn.init.xavier_uniform_(self.S)
        self.sage_block2 = SAGEConv_2(hidden_channels, out_channels)

    def forward(self, x, edge_index, nmv_values):
        h1 = self.sage_block1(x, edge_index, nmv_values)
        m = torch.matmul(h1, self.S)
        E_logits = torch.matmul(m, h1.t())
        E = torch.softmax(F.relu(E_logits), dim=1)
        logits = self.sage_block2(h1, edge_index, edge_weight=nmv_values)
        return logits, h1, E