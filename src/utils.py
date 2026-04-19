import torch
from torch_geometric.utils import homophily
import numpy as np
import torch.nn as nn

def print_graph_stats(dataset):
    data = dataset[0]

    num_nodes = data.num_nodes
    num_edges = data.num_edges
    num_classes = dataset.num_classes

    h_score = homophily(data.edge_index, data.y, method='edge')

    print(f"--- Statistics for {dataset.name} ---")
    print(f"Nodes:      {num_nodes}")
    print(f"Edges:      {num_edges}")
    print(f"Classes:    {num_classes}")
    print(f"Homophily:  {h_score:.4f}")
    print("-" * 30)



def get_asymmetric_nmv_matrix(data, minority_classes, p, sparse=True):
    edge_index = data.edge_index
    labels = data.y
    num_nodes = data.num_nodes
    u_idx, v_idx = edge_index

    class_counts = torch.bincount(labels)
    max_count = class_counts.max().item()
    rho_per_class = max_count / (class_counts.float() + 1e-6)

    is_minority = torch.zeros(num_nodes, dtype=torch.float, device=labels.device)
    is_minority[torch.tensor(minority_classes, device=labels.device)] = 1.0

    rho_u = rho_per_class[labels[u_idx]]
    indicator_u = is_minority[u_idx]
    base_weight = (rho_u - 1) * indicator_u + 1

    same_class = (labels[u_idx] == labels[v_idx]).float()
    diff_class_mask = (labels[u_idx] != labels[v_idx]).float()

    nmv_values = (base_weight * same_class) + (base_weight * p * diff_class_mask)

    if sparse:
        return torch.sparse_coo_tensor(edge_index, nmv_values, (num_nodes, num_nodes))
    else:
        mat = torch.zeros((num_nodes, num_nodes), device=labels.device)
        mat[u_idx, v_idx] = nmv_values
        return mat



class GumbelActivation(nn.Module):
    def __init__(self):
        super(GumbelActivation, self).__init__()

    def forward(self, x):
        x = torch.clamp(x, min=-10, max=10)
        return torch.exp(-torch.exp(-x))
    


def get_imbalance_ratios(labels):
    class_counts = torch.bincount(labels)
    max_count = class_counts.max().float()
    ir_dict = max_count / (class_counts.float() + 1e-6)
    return ir_dict



def get_asymmetric_weight_matrix(labels, minority_classes):
    num_nodes = labels.size(0)
    device = labels.device

    ir_lookup = get_imbalance_ratios(labels)
    node_ir = ir_lookup[labels]

    is_minority = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    is_minority[torch.tensor(minority_classes, device=device)] = True

    W = torch.ones((num_nodes, num_nodes), device=device)
    effective_ir = torch.where(is_minority, node_ir, torch.ones_like(node_ir))
    W = torch.max(effective_ir.view(-1, 1), effective_ir.view(1, -1))
    return W



def compute_edge_loss(h1, S, edge_index, W_e):
    m = torch.matmul(h1, S)
    E_logits = torch.matmul(m, h1.t())
    E = torch.softmax(F.relu(E_logits), dim=1)

    A = torch.zeros_like(E)
    A[edge_index[0], edge_index[1]] = 1.0

    loss_matrix = W_e * torch.pow(E - A, 2)
    return torch.sum(loss_matrix)


