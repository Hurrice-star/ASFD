from utils.config import cfg  # isort: split

import csv
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils.eval import get_val_cfg, validate
from utils.datasets import create_dataloader
from utils.utils import get_network

cfg = get_val_cfg(cfg, split="test", copy=False)

assert cfg.ckpt_path, "Please specify the path to the model checkpoint"
model_name = os.path.basename(cfg.ckpt_path).replace(".pth", "")
dataset_root = os.path.join(cfg.root_dir, "data", "test_new")
# cfg.datasets_test = [name for name in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, name))]
rows = []
print(f"'{cfg.exp_name}:{model_name}' model testing on...")

all_features = []
all_labels = []

print(f"'{cfg.exp_name}:{model_name}' model testing on...")

for i, dataset in enumerate(cfg.datasets_test):
    cfg.dataset_root = os.path.join(dataset_root, dataset)
    cfg.datasets = [""]

    model = get_network(cfg.arch)
    state_dict = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict["model"])
    model.cuda()
    model.eval()

    data_loader = create_dataloader(cfg)

    features = []
    labels = []

    with torch.no_grad():
        for data in data_loader:
                inputs, x2, x3, targets = data[0], data[1], data[2], data[3]  
                inputs, x2, x3, targets = inputs.cuda(), x2.cuda(), x3.cuda(), targets.cuda()
                outputs = model(inputs, x2, x3)  
            
                features.append(outputs.cpu().numpy())
                labels.append(targets.cpu().numpy().flatten())


    all_features.extend(features)
    all_labels.extend(labels)

    test_results = validate(model, cfg)

    print(f"Results for {dataset}:")
    for k, v in test_results.items():
        if isinstance(v, list):
            v = ', '.join(f"{x:.5f}" for x in v)
            print(f"{k}: [{v}]")
        else:
            print(f"{k}: {v:.5f}")
    print("*" * 50)

    if i == 0:
        rows.append(["TestSet"] + list(test_results.keys()))
    rows.append([dataset] + list(test_results.values()))

# Save results to CSV
results_dir = os.path.join(cfg.root_dir, "data", "results")
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, f"{cfg.exp_name}-{model_name}.csv"), "w") as f:
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerows(rows)





