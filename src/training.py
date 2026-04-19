from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from utils import get_asymmetric_nmv_matrix, get_imbalance_ratios, get_asymmetric_weight_matrix, compute_edge_loss
import torch.nn.functional as F
import torch

def train_full_pipeline(model, data, minority_classes, p_val, optimizer):
    model.train()
    optimizer.zero_grad()

    nmv_sparse = get_asymmetric_nmv_matrix(data, minority_classes, p=p_val, sparse=True)
    nmv_sparse = nmv_sparse.coalesce()

    indices = nmv_sparse.indices()
    values = nmv_sparse.values()

    logits, h1, E = model(data.x, indices, values)

    ir_weights = get_imbalance_ratios(data.y).to(data.x.device)
    loss_cls = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask], weight=ir_weights)

    W_e = get_asymmetric_weight_matrix(data.y, minority_classes)
    loss_edge = compute_edge_loss(h1, model.S, indices, W_e)

    total_loss = loss_cls + 0.000001 * loss_edge

    total_loss.backward()
    optimizer.step()

    return total_loss.item()

@torch.no_grad()
def evaluate_model(model, data, minority_classes, p_val, mask):
    model.eval()

    nmv_sparse = get_asymmetric_nmv_matrix(data, minority_classes, p=p_val, sparse=True).coalesce()
    logits, _, _ = model(data.x, nmv_sparse.indices(), nmv_sparse.values())

    y_true = data.y[mask].cpu().numpy()
    y_probs = F.softmax(logits[mask], dim=1).cpu().numpy()
    y_pred = logits[mask].argmax(dim=1).cpu().numpy()

    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average='macro'),
        "gmean": geometric_mean_score(y_true, y_pred, average='macro'),
        "auc": roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    }