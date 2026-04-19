import torch
from torch_geometric.datasets import Planetoid, WikiCS, Twitch
from torch_geometric.transforms import NormalizeFeatures
import torch_geometric.transforms as T
import os

def get_dataset(name, root='data/'):
    os.makedirs(root, exist_ok=True)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=root, name=name, transform=NormalizeFeatures())
    elif name == 'Wiki':
        dataset = WikiCS(root=root, transform=NormalizeFeatures())
    elif name == 'Twitch':
        dataset = Twitch(root=root, name='EN', transform=NormalizeFeatures()) # Using EN as example
    else:
        raise ValueError(f"Dataset {name} not supported.")

    return dataset[0], dataset

def get_imbalanced_split(data, train_ratio=0.1, val_ratio=0.2, num_minority_classes=3):
    labels = data.y
    num_classes = labels.max().item() + 1

    class_counts = torch.bincount(labels)
    sorted_classes = torch.argsort(class_counts)

    minority_classes = sorted_classes[:num_minority_classes].tolist()
    majority_classes = sorted_classes[num_minority_classes:].tolist()

    max_class_count = class_counts.max().item()

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    for c in range(num_classes):
        indices = (labels == c).nonzero(as_tuple=True)[0]
        indices = indices[torch.randperm(len(indices))]

        num_train = int(len(indices) * train_ratio)
        num_val = int(len(indices) * val_ratio)

        train_mask[indices[:num_train]] = True
        val_mask[indices[num_train:num_train+num_val]] = True
        test_mask[indices[num_train+num_val:]] = True

    return train_mask, val_mask, test_mask, minority_classes, max_class_count
