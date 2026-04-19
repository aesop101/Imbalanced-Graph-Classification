import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from utils import GumbelActivation


class NMV_SAGE_Layer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(NMV_SAGE_Layer, self).__init__(aggr='mean')
        self.W = nn.Linear(2 * in_channels, out_channels)
        self.activation = GumbelActivation()

    def forward(self, x, edge_index, nmv_values):
        return self.propagate(edge_index, x=x, nmv=nmv_values)

    def message(self, x_j, nmv):
        return nmv.view(-1, 1) * x_j

    def update(self, aggr_out, x):
        combined = torch.cat([x, aggr_out], dim=1)
        out = self.W(combined)
        return self.activation(out)
    

class SAGEConv_2(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(2 * in_channels, out_channels)
        self.activation = GumbelActivation()

    def forward(self, x, edge_index, edge_weight=None):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out, x):
        combined = torch.cat([x, aggr_out], dim=1)
        return self.activation(self.lin(combined))