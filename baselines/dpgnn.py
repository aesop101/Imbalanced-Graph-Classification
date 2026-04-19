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

class DPGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DPGNN, self).__init__()

        self.local_conv1 = SAGEConv(in_channels, hidden_channels)
        self.local_conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.global_conv1 = SAGEConv(in_channels, hidden_channels)
        self.global_conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        l = F.relu(self.local_conv1(x, edge_index))
        l = self.local_conv2(l, edge_index)

        g = F.relu(self.global_conv1(x, edge_index))
        g = self.global_conv2(g, edge_index)

        combined = torch.cat([l, g], dim=-1)
        return self.classifier(combined)
    
def train_dpgnn(model, data, train_mask, optimizer):
    model.train()
    optimizer.zero_grad()

    labels = data.y[train_mask]
    class_counts = torch.bincount(labels)
    weights = 1.0 / class_counts.float()
    weights = weights / weights.sum() # Normalize

    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask], weight=weights)

    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test_dpgnn(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index)

    probs = F.softmax(logits[mask], dim=1)
    preds = logits[mask].argmax(dim=1)
    y_true = data.y[mask].cpu().numpy()
    y_pred = preds.cpu().numpy()
    y_probs = probs.cpu().numpy()

    acc = (preds == data.y[mask]).sum().item() / mask.sum().item()
    f1 = f1_score(y_true, y_pred, average='macro')
    gmean = geometric_mean_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    return acc, f1, gmean, auc


# experiments = [
#     ("Cora", cora_data, cora_dataset, train_mask_1, val_mask_1, test_mask_1), # Assuming you named masks uniquely
#     ("CiteSeer", citeseer_data, citeseer_dataset, train_mask_2, val_mask_2, test_mask_2),
#     ("PubMed", pubmed_data, pubmed_dataset, train_mask_3, val_mask_3, test_mask_3)
# ]

def run_baseline_gnn(experiments, hidden_dim=HIDDEN_SIZE, epochs=EPOCHS, step=STEP):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for name, data, dataset, train_mask, val_mask, test_mask, _ in experiments:
        data = data.to(device)
        train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

        model = DPGNN(dataset.num_node_features, hidden_dim, dataset.num_classes).to(device)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        print(f"\nTraining DPGNN on {name}...")
        for epoch in range(1, epochs + 1):
            loss = train_dpgnn(model, data, train_mask, optimizer)
            if epoch % step == 0:
                acc, f1, gmean, auc = test_dpgnn(model, data, val_mask)
                print(f"{name} Epoch {epoch} Loss: {loss:.4f}, Val Acc: {acc:.4f}, Val F1: {f1:.4f}, Val G-Mean: {gmean:.4f}, Val AUC: {auc:.4f}")

        acc, f1, gmean, auc = test_dpgnn(model, data, test_mask)
        print(f"\n{name} Results: Acc: {acc:.4f}, F1: {f1:.4f}, G-Mean: {gmean:.4f}, AUC: {auc:.4f}")