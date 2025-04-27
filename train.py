from utils.config import cfg  # isort: split

import os
import time

from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils.datasets import create_dataloader
from utils.earlystop import EarlyStopping
from utils.eval import get_val_cfg, validate
from utils.trainer import Trainer
from utils.utils import Logger
from sklearn.metrics import accuracy_score
import torch
import numpy as np

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_cfg = get_val_cfg(cfg, split="val", copy=True)
    cfg.dataset_root = os.path.join(cfg.dataset_root, "train")
    data_loader = create_dataloader(cfg)
    dataset_size = len(data_loader)

    log = Logger()
    log.open(cfg.logs_path, mode="a")
    log.write("Num of training images = %d\n" % (dataset_size * cfg.batch_size))
    log.write("Config:\n" + str(cfg.to_dict()) + "\n")

    train_writer = SummaryWriter(os.path.join(cfg.exp_dir, "train"))
    val_writer = SummaryWriter(os.path.join(cfg.exp_dir, "val"))

    # train.py

    if __name__ == "__main__":

        trainer = Trainer(cfg)
        early_stopping = EarlyStopping(patience=cfg.earlystop_epoch, delta=-0.001, verbose=True)

        for epoch in range(cfg.nepoch):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            # Initialize lists to track true labels and predictions for accuracy
            y_true_train = []
            y_pred_train = []

            for data in tqdm(data_loader, dynamic_ncols=True):
                trainer.total_steps += 1
                epoch_iter += cfg.batch_size

                # 正确解包数据
                inputs1, inputs2, inputs3, labels = data  # 修改为三输入
                inputs1 = inputs1.to(device)
                inputs2 = inputs2.to(device)
                inputs3 = inputs3.to(device)  # 添加第三个输入
                labels = labels.to(device)

                # 将三张图像的输入传递给模型
                trainer.set_input((inputs1, inputs2, inputs3, labels))  # 修改为三输入
                trainer.optimize_parameters()

                # 进行预测
                with torch.no_grad():
                    outputs = trainer.model(inputs1, inputs2, inputs3)  # 修改为三输入
                    _, preds = torch.max(outputs, 1)

                y_true_train.extend(labels.cpu().numpy())
                y_pred_train.extend(preds.cpu().numpy())

                # Log training loss
                train_writer.add_scalar("loss", trainer.loss, trainer.total_steps)

                if trainer.total_steps % cfg.save_latest_freq == 0:
                    log.write(
                        "saving the latest model %s (epoch %d, model.total_steps %d)\n"
                        % (cfg.exp_name, epoch, trainer.total_steps)
                    )
                    trainer.save_networks("latest")

            # Calculate training accuracy
            train_acc = accuracy_score(np.array(y_true_train), np.array(y_pred_train))
            train_writer.add_scalar("ACC", train_acc, trainer.total_steps)
            log.write(f"(Train @ epoch {epoch}) ACC: {train_acc}\n")

            if epoch % cfg.save_epoch_freq == 0:
                log.write("saving the model at the end of epoch %d, iters %d\n" % (epoch, trainer.total_steps))
                trainer.save_networks("latest")
                trainer.save_networks(epoch)

            # Validation
            trainer.eval()

            val_results = validate(trainer.model, val_cfg)
            val_writer.add_scalar("AP", val_results["AP"], trainer.total_steps)
            val_writer.add_scalar("ACC", val_results["ACC"], trainer.total_steps)
            log.write(f"(Val @ epoch {epoch}) AP: {val_results['AP']}; ACC: {val_results['ACC']}\n")

            # Early stopping logic
            if cfg.earlystop:
                early_stopping(val_results["ACC"], trainer)
                if early_stopping.early_stop:
                    if trainer.adjust_learning_rate():
                        log.write("Learning rate dropped by 10, continue training...\n")
                        early_stopping = EarlyStopping(patience=cfg.earlystop_epoch, delta=-0.002, verbose=True)
                    else:
                        log.write("Early stopping.\n")
                        break

            # Scheduler step if warmup is enabled
            if cfg.warmup:
                trainer.scheduler.step()

            # Return to training mode
            trainer.train()
