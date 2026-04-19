import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from imblearn.metrics import geometric_mean_score

from dotenv import load_dotenv
import os
load_dotenv()

HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE"))
LR = float(os.getenv("LR"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY"))
P_VAL = float(os.getenv("P_VAL"))
EPOCHS = float(os.getenv("EPOCHS"))
STEP = float(os.getenv("STEP"))

class Graph_SMOTE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Graph_SMOTE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, 1),
            nn.Sigmoid()
        )
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.classifier(z)

    def generate_synthetic_nodes(self, z, y, train_mask, minority_classes, oversample_ratio=1.0):
        new_z, new_y = [], []
        for c in minority_classes:
            c_indices = ((y == c) & train_mask).nonzero(as_tuple=True)[0]
            if len(c_indices) < 2: continue

            num_to_generate = int(len(c_indices) * oversample_ratio)
            for _ in range(num_to_generate):
                idx1, idx2 = c_indices[torch.randint(0, len(c_indices), (2,))]
                alpha = torch.rand(1).to(z.device)
                z_new = z[idx1] + alpha * (z[idx2] - z[idx1])

                new_z.append(z_new)
                new_y.append(c)

        if len(new_z) > 0:
            return torch.stack(new_z), torch.tensor(new_y).to(z.device)
        return None, None
    

def train_graph_smote(model, data, train_mask, minority_classes, optimizer):
    model.train()
    optimizer.zero_grad()

    z = model.encode(data.x, data.edge_index)
    syn_z, syn_y = model.generate_synthetic_nodes(z, data.y, train_mask, minority_classes)

    combined_z = torch.cat([z[train_mask], syn_z], dim=0) if syn_z is not None else z[train_mask]
    combined_y = torch.cat([data.y[train_mask], syn_y], dim=0) if syn_y is not None else data.y[train_mask]

    logits = model.classifier(combined_z)
    loss_cls = F.cross_entropy(logits, combined_y)

    loss_cls.backward()
    optimizer.step()
    return loss_cls.item()

@torch.no_grad()
def test_graph_smote(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    probs = F.softmax(logits[mask], dim=1)
    preds = logits[mask].argmax(dim=1)
    y_true = data.y[mask].cpu().numpy()

    acc = (preds == data.y[mask]).sum().item() / mask.sum().item()
    f1 = f1_score(y_true, preds.cpu().numpy(), average='macro')
    gmean = geometric_mean_score(y_true, preds.cpu().numpy(), average='macro')
    auc = roc_auc_score(y_true, probs.cpu().numpy(), multi_class='ovr', average='macro')

    return acc, f1, gmean, auc


# experiments = [
#     ("Cora", cora_data, cora_dataset, train_mask_1, val_mask_1, test_mask_1), # Assuming you named masks uniquely
#     ("CiteSeer", citeseer_data, citeseer_dataset, train_mask_2, val_mask_2, test_mask_2),
#     ("PubMed", pubmed_data, pubmed_dataset, train_mask_3, val_mask_3, test_mask_3)
# ]

def run_baseline_gnn(experiments, hidden_dim=HIDDEN_SIZE, epochs=EPOCHS, step=STEP):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for name, data, dataset, train_mask, val_mask, test_mask, minority_classes in experiments:
        data = data.to(device)
        train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

        model = Graph_SMOTE(dataset.num_node_features, hidden_dim, dataset.num_classes).to(device)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        for epoch in range(1, epochs + 1):
            loss = train_graph_smote(model, data, train_mask, minority_classes, optimizer)
            if epoch % step == 0:
                acc, f1, gmean, auc = test_graph_smote(model, data, val_mask)
                print(f"{name} Epoch {epoch} Loss: {loss:.4f}, Val Acc: {acc:.4f}, Val F1: {f1:.4f}, Val G-Mean: {gmean:.4f}, Val AUC: {auc:.4f}")

        acc, f1, gmean, auc = test_graph_smote(model, data, test_mask)
        print(f"\n{name} Results: Acc: {acc:.4f}, F1: {f1:.4f}, G-Mean: {gmean:.4f}, AUC: {auc:.4f}")