import torch
from src.data_loader import get_dataset, get_imbalanced_split
from src.model import NMV_Graph_Model
from src.training import train_full_pipeline, evaluate_model
from dotenv import load_dotenv
import os
load_dotenv()

HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE"))
LR = float(os.getenv("LR"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY"))
P_VAL = float(os.getenv("P_VAL"))
EPOCHS = float(os.getenv("EPOCHS"))
STEP = float(os.getenv("STEP"))

def main():
    cora_data, cora_dataset = get_dataset("Cora")
    train_mask_1, val_mask_1, test_mask_1, minority_classes_1, max_class_count_1 = get_imbalanced_split(cora_data, train_ratio=0.1, val_ratio=0.2, num_minority_classes=3)

    citeseer_data, citeseer_dataset = get_dataset("CiteSeer")
    train_mask_2, val_mask_2, test_mask_2, minority_classes_2, max_class_count_2 = get_imbalanced_split(citeseer_data, train_ratio=0.1, val_ratio=0.2, num_minority_classes=3)

    pubmed_data, pubmed_dataset = get_dataset("PubMed")
    train_mask_3, val_mask_3, test_mask_3, minority_classes_3, max_class_count_3 = get_imbalanced_split(pubmed_data, train_ratio=0.1, val_ratio=0.2, num_minority_classes=3)

    datasets = [
        ("Cora", cora_data, minority_classes_1, train_mask_1, val_mask_1, test_mask_1),
        ("CiteSeer", citeseer_data, minority_classes_2, train_mask_2, val_mask_2, test_mask_2),
        ("PubMed", pubmed_data, minority_classes_3, train_mask_3, val_mask_3, test_mask_3)
    ]

    for name, data, min_classes, t_mask, v_mask, te_mask in datasets:
        print(f"\n=== Training on {name} ===")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        data.train_mask, data.val_mask, data.test_mask = t_mask, v_mask, te_mask

        model = NMV_Graph_Model(data.num_features, HIDDEN_SIZE, int(data.y.max() + 1)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        for epoch in range(1, EPOCHS+1):
            loss = train_full_pipeline(model, data, min_classes, P_VAL, optimizer)

            if epoch % STEP == 0:
                metrics = evaluate_model(model, data, min_classes, P_VAL, data.val_mask)
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val F1: {metrics['f1']:.4f}")

        test_metrics = evaluate_model(model, data, min_classes, P_VAL, data.test_mask)
        print(f"FINAL TEST RESULTS FOR {name}:")
        for k, v in test_metrics.items():
            print(f" - {k.upper()}: {v:.4f}")