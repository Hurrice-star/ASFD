import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import CONFIGCLASS
from utils.utils import to_cuda


def get_val_cfg(cfg: CONFIGCLASS, split="val", copy=True):
    if copy:
        from copy import deepcopy
        val_cfg = deepcopy(cfg)
    else:
        val_cfg = cfg
    val_cfg.dataset_root = os.path.join(val_cfg.dataset_root, split)
    val_cfg.datasets = cfg.datasets_test
    val_cfg.isTrain = False
    val_cfg.aug_flip = False
    val_cfg.serial_batches = True
    val_cfg.jpg_method = ["pil"]
    if len(val_cfg.blur_sig) == 2:
        b_sig = val_cfg.blur_sig
        val_cfg.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_cfg.jpg_qual) != 1:
        j_qual = val_cfg.jpg_qual
        val_cfg.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    return val_cfg


def validate(model: nn.Module, cfg: CONFIGCLASS):
    from sklearn.metrics import accuracy_score, average_precision_score
    from sklearn.preprocessing import label_binarize
    from utils.datasets import create_dataloader

    data_loader = create_dataloader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3  # 三分类任务

    y_true, y_pred = [], []
    model.eval()

    with torch.no_grad():
        for data in data_loader:
            # 假设 data[0] 包含三张图像 [img1, img2, img3] 和标签
            img1, img2, img3, label, meta = data if len(data) == 5 else (*data, None)
            in_tens1 = to_cuda(img1, device)
            in_tens2 = to_cuda(img2, device)
            in_tens3 = to_cuda(img3, device)
            meta = to_cuda(meta, device)

            # 使用模型进行三图像输入的前向传播
            predict = model(in_tens1, in_tens2, in_tens3).softmax(dim=1)

            y_pred.extend(predict.cpu().numpy())
            y_true.extend(label.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true_one_hot = label_binarize(y_true, classes=[0, 1, 2])

    # 计算每个类别的平均精度 (AP) 和准确率 (ACC)
    aps = [average_precision_score(y_true_one_hot[:, i], y_pred[:, i]) for i in range(num_classes)]
    ap_mean = np.mean(aps)  # 计算所有类别的 AP 平均值

    # 计算整体准确率
    y_pred_class = np.argmax(y_pred, axis=1)  # 获取每个样本的预测类别
    acc = accuracy_score(y_true, y_pred_class)

    # 计算每个类别的 ACC
    accs = []
    for i in range(num_classes):
        # 获取当前类别的样本索引
        class_mask = y_true == i
        acc_class = accuracy_score(y_true[class_mask], y_pred_class[class_mask])
        accs.append(acc_class)

    # 输出每个类别的 ACC 和 AP
    for i in range(num_classes):
        print(f"Class {i}: ACC = {accs[i]:.4f}, AP = {aps[i]:.4f}")

    # 存储结果
    results = {
        "ACC": acc,
        "AP": ap_mean,  # 将平均 AP 返回为 'AP'
        "Class_ACC": accs,  # 存储每个类别的 ACC
        "Class_AP": aps,  # 存储每个类别的 AP
    }

    return results


