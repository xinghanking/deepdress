import datetime
import json
import os
import random
import shutil
import sys
import threading
import time
from time import sleep

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 支持在任意目录下直接运行
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import Config
from core.model import MultiTaskModel

datasets_indices_path = os.path.join(Config.checkpoint_dir, "indices")

# 全局训练状态，用于线程间通信
train_progress = {
    "epoch": 0,
    "batch": 0,
    "best_val_loss": float('inf'),
    "patience_counter": 0,
    "stop": False,
    "full_indices": {}
}

#torch.manual_seed(Config.seed)
#    random.seed(Config.seed)
#    if torch.cuda.is_available():
#        torch.cuda.manual_seed_all(Config.seed)
#    import numpy as np
#    np.random.seed(Config.seed)


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
        image_path = os.path.join(Config.base_root, "datasets", row["image"])
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
    dataset_len = len(train_progress["full_indices"]["train"])
    total_batches = (dataset_len + Config.batch_size - 1) // Config.batch_size
    last_pos = last_epoch * total_batches + last_batch
    pbar = tqdm(total=total_epochs * total_batches, desc="Training Progress", dynamic_ncols=True)
    while train_progress["stop"] == False or last_batch < train_progress['batch']:
        if train_progress["epoch"] > last_epoch or train_progress["batch"] > last_batch:
            if train_progress["patience_counter"] == Config.early_stop_patience:
                break
            if train_progress["epoch"] > last_epoch:
                json.dump(train_progress["full_indices"], open(datasets_indices_path, "w"))
            elapsed = time.time() - start_time
            steps_done = train_progress["epoch"] * total_batches + train_progress["batch"]
            steps_total = total_epochs * total_batches
            avg_step_time = elapsed / (steps_done - last_pos)
            eta = (steps_total - steps_done) * avg_step_time
            pbar.n = steps_done
            pbar.set_postfix({
                "Epoch": f"{train_progress['epoch'] + 1}/{total_epochs}",
                "Batch": f"{train_progress['batch'] + 1}/{total_batches}",
                "Elapsed": format_seconds(elapsed),
                "ETA": format_seconds(eta),
            })
            pbar.refresh()
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            save_path = Config.model_save_path
            checkpoint = {
                "model_state": state_dict,
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": train_progress["best_val_loss"],
                "patience_counter": train_progress["patience_counter"],
            }
            if train_progress["epoch"] < total_epochs - 1 or train_progress["batch"] < total_batches - 1:
                checkpoint["epoch"] = train_progress["epoch"]
                checkpoint["batch"] = train_progress["batch"]
                checkpoint["patience_counter"] = train_progress["patience_counter"]
                checkpoint["rng_state"] = torch.get_rng_state()
                checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            if scaler is not None:
                checkpoint["scaler_state"] = scaler.state_dict()
            try:
                if os.path.exists(save_path):
                    bak_path = save_path + ".bak"
                    if os.path.exists(bak_path):
                        os.remove(bak_path)
                    shutil.move(save_path, bak_path)
                torch.save(checkpoint, save_path)
            except:
                raise RuntimeError("保存训练模型失败")
            if train_progress["epoch"] > last_epoch:
                if train_progress["patience_counter"] == 0:
                    shutil.copy(save_path, Config.model_save_path.replace(".pt", "_best.pt"))
            last_epoch, last_batch = train_progress["epoch"], train_progress["batch"]
        else:
            sleep(10)
    pbar.close()

#set_seed()

model = MultiTaskModel()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(Config.device)

optimizer = optim.Adam(model.parameters(), lr=getattr(Config, "lr", 1e-4))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.1
)
criterion_kind = nn.CrossEntropyLoss()
criterion_gender = nn.CrossEntropyLoss()
criterion_age = nn.MSELoss()

use_amp = Config.use_amp
scaler = torch.amp.GradScaler("cuda") if use_amp else None

# 恢复断点
start_epoch = 0
start_batch = 0
best_val_loss = float('inf')
patience_counter = 0

full_dataset = DeepDressDataset(Config.csv_path)
dataset_size = len(full_dataset)
if os.path.exists(datasets_indices_path):
    full_indices = json.load(open(datasets_indices_path))
else:
    val_size = int(dataset_size * Config.val_ratio)
    train_size = dataset_size - val_size
    full_indices = list(range(dataset_size))
    random.shuffle(full_indices)
    train_indices, val_indices = full_indices[:train_size], full_indices[train_size:]
    full_indices = {"train": train_indices, "val": val_indices}
    json.dump(full_indices, open(datasets_indices_path, "w"))
val_indices = full_indices["val"]
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
checkpoint = None
if os.path.exists(Config.model_save_path):
    try:
        checkpoint = torch.load(Config.model_save_path, map_location=Config.device, weights_only=False)
    except RuntimeError:
        checkpoint = None
if not checkpoint:
    bak_path = Config.model_save_path + ".bak"
    if os.path.exists(bak_path):
        try:
            checkpoint = torch.load(bak_path, map_location=Config.device, weights_only=False)
            os.remove(Config.model_save_path)
        except RuntimeError:
            checkpoint = None
if checkpoint:
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

val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=2)
# 更新全局状态
train_progress["epoch"] = start_epoch
train_progress["batch"] = start_batch
train_progress["best_val_loss"] = best_val_loss
train_progress["patience_counter"] = patience_counter
train_progress["full_indices"] = full_indices

# 启动监控线程
monitor_thread = threading.Thread(target=monitor_thread_func, args=(model, optimizer, scaler))
monitor_thread.start()

for epoch in range(start_epoch, Config.epochs):
    train_dataset = torch.utils.data.Subset(full_dataset, full_indices["train"])
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=2)
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
    random.shuffle(full_indices["train"])
    train_progress["full_indices"] = full_indices
    scheduler.step(val_loss)
# 通知监控线程结束并等待
train_progress["stop"] = True
monitor_thread.join()
print("✅训练完成，最优模型。")
