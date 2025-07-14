import os
import shutil
import sys
import random
import threading
import datetime, time
from time import sleep

import pandas as pd
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.serialization import safe_globals

# 支持在任意目录下直接运行
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import Config
from core.model import MultiTaskModel

# 全局训练状态，用于线程间通信
train_progress = {
    "epoch": 0,
    "batch": 0,
    "best_val_loss": float('inf'),
    "patience_counter": 0,
    "stop": False,
}


def set_seed():
    torch.manual_seed(Config.seed)
    random.seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.seed)
    import numpy as np
    np.random.seed(Config.seed)


def format_seconds(seconds):
    """将秒转换为 H:M:S 格式"""
    return str(datetime.timedelta(seconds=int(seconds)))


class DeepDressDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(Config.normalize_mean, Config.normalize_std)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(Config.base_root, "datasets", "wiki", row["image"])
        try:
            image = Image.open(image_path)
            image = self.transforms(image)
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"图像读取失败: {image_path} | 错误: {str(e)}")
        kind = int(row["kind"])
        gender = int(row["gender"])
        age = float(row["age"])
        return image, kind, gender, age


def validate(model, val_loader, criterion_kind, criterion_gender, criterion_age, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, kind, gender, age in val_loader:
            images = images.to(device)
            kind = kind.to(device)
            gender = gender.to(device)
            age = age.to(device, dtype=torch.float32)
            pred_kind, pred_gender, pred_age = model(images)
            loss_kind = criterion_kind(pred_kind, kind)
            loss_gender = criterion_gender(pred_gender, gender)
            loss_age = criterion_age(pred_age.squeeze(), age)
            loss = loss_kind + loss_gender + loss_age
            val_loss += loss.item()
    avg_loss = val_loss / len(val_loader)
    model.train()
    return avg_loss


def monitor_thread_func(model, optimizer, scaler):
    last_epoch, last_batch, start_time = train_progress["epoch"], train_progress["batch"], time.time()
    total_epochs = Config.epochs
    dataset_len = len(train_progress["train_dataset"])
    last_pos = last_epoch * dataset_len + last_batch
    total_batches = (dataset_len + Config.batch_size - 1) // Config.batch_size
    pbar = tqdm(total=total_epochs * total_batches, desc="Training Progress", dynamic_ncols=True)
    while train_progress["stop"] == False or last_batch < train_progress['batch']:
        if train_progress["epoch"] > last_epoch or train_progress["batch"] > last_batch:
            if train_progress["patience_counter"] == Config.early_stop_patience:
                break
            elapsed = time.time() - start_time
            steps_done = train_progress["epoch"] * total_batches + train_progress["batch"]
            steps_total = total_epochs * total_batches
            avg_step_time = elapsed / (steps_done - last_pos)
            eta = (steps_total - steps_done) * avg_step_time
            datetime.datetime.now() + datetime.timedelta(seconds=eta)
            pbar.n = steps_done
            pbar.set_postfix({
                "Epoch": f"{train_progress['epoch'] + 1}/{total_epochs}",
                "Batch": f"{train_progress['batch'] + 1}/{total_batches}",
                "Elapsed": format_seconds(elapsed),
                "ETA": format_seconds(eta),
            })
            pbar.update()
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            save_path = Config.model_save_path
            checkpoint = {
                "epoch": train_progress["epoch"],
                "batch": train_progress["batch"],
                "model_state": state_dict,
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": train_progress["best_val_loss"],
                "patience_counter": train_progress["patience_counter"],
                "train_dataset": train_progress["train_dataset"],
                "val_dataset": train_progress["val_dataset"],
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            }
            if scaler is not None:
                checkpoint["scaler_state"] = scaler.state_dict()
            torch.save(checkpoint, save_path)
            if train_progress["epoch"] > last_epoch:
                if train_progress["patience_counter"] == 0:
                    shutil.copy(save_path, Config.model_save_path.replace(".pt", "_best.pt"))
            last_epoch, last_batch = train_progress["epoch"], train_progress["batch"]
        else:
            sleep(1)
    pbar.close()


def main():
    set_seed()

    model = MultiTaskModel()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(Config.device)

    optimizer = optim.Adam(model.parameters(), lr=getattr(Config, "lr", 1e-4))
    criterion_kind = nn.CrossEntropyLoss()
    criterion_gender = nn.CrossEntropyLoss()
    criterion_age = nn.MSELoss()

    use_amp = getattr(Config, "use_amp", False)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # 恢复断点
    start_epoch = 0
    start_batch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    train_dataset = None
    val_dataset = None
    if os.path.exists(Config.model_save_path):
        with safe_globals([Dataset]):
            checkpoint = torch.load(Config.model_save_path, map_location=Config.device, weights_only=False)
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scaler and "scaler_state" in checkpoint and checkpoint["scaler_state"]:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint.get("epoch", 0)
        start_batch = checkpoint.get("batch", 0)
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        patience_counter = checkpoint.get("patience_counter", 0)
        train_dataset = checkpoint.get("train_dataset", None)
        val_dataset = checkpoint.get("val_dataset", None)
    if not train_dataset or not val_dataset:
        full_dataset = DeepDressDataset(Config.csv_path)
        val_ratio = getattr(Config, "val_ratio", 0.1)
        val_size = int(len(full_dataset) * val_ratio)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=2)
    # 更新全局状态
    train_progress["train_dataset"] = train_dataset
    train_progress["val_dataset"] = val_dataset
    train_progress["epoch"] = start_epoch
    train_progress["batch"] = start_batch
    train_progress["best_val_loss"] = best_val_loss
    train_progress["patience_counter"] = patience_counter

    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor_thread_func, args=(model, optimizer, scaler))
    monitor_thread.start()

    for epoch in range(start_epoch, Config.epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, kind, gender, age) in enumerate(train_loader):
            # 跳过已训练批次
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            images = images.to(Config.device)
            kind = kind.to(Config.device)
            gender = gender.to(Config.device)
            age = age.to(Config.device, dtype=torch.float32)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    pred_kind, pred_gender, pred_age = model(images)
                    loss_kind = criterion_kind(pred_kind, kind)
                    loss_gender = criterion_gender(pred_gender, gender)
                    loss_age = criterion_age(pred_age.squeeze(), age)
                    loss = loss_kind + loss_gender + loss_age
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_kind, pred_gender, pred_age = model(images)
                loss_kind = criterion_kind(pred_kind, kind)
                loss_gender = criterion_gender(pred_gender, gender)
                loss_age = criterion_age(pred_age.squeeze(), age)
                loss = loss_kind + loss_gender + loss_age
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # 更新进度状态（无任何print）
            train_progress["epoch"] = epoch
            train_progress["batch"] = batch_idx

        val_loss = validate(model, val_loader, criterion_kind, criterion_gender, criterion_age, Config.device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            train_progress["best_val_loss"] = best_val_loss
            train_progress["patience_counter"] = patience_counter
        else:
            patience_counter += 1
            train_progress["patience_counter"] = patience_counter
            if patience_counter >= getattr(Config, "early_stop_patience", 8):
                print("🛑 达到early stopping条件，训练终止。")
                break

    # 通知监控线程结束并等待
    train_progress["stop"] = True
    monitor_thread.join()
    print("✅训练完成，最优模型。")


if __name__ == "__main__":
    main()
