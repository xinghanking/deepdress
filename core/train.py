import datetime
import json
import os
import random
import shutil
import sys
import threading
import time
from os import unlink
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
bak_save_path = Config.model_save_path.replace(".pt", "_bak.pt")
# 全局训练状态，用于线程间通信
train_progress = {
    "current": {
        "epoch": 0,
        "batch": 0,
    },
    "full_indices": {},
    "last_epoch": {},
    "stop": False,
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


def monitor_thread_func():
    last_epoch, last_batch, start_time = train_progress["current"]["epoch"], train_progress["current"]["batch"], time.time()
    total_epochs = Config.epochs
    dataset_len = len(train_progress["full_indices"]["train"])
    total_batches = (dataset_len + Config.batch_size - 1) // Config.batch_size
    last_pos = last_epoch * total_batches
    pbar = tqdm(total=total_epochs * total_batches, desc="Training Progress", dynamic_ncols=True)
    best_path = Config.model_save_path.replace(".pt", "_best.pt")
    if os.path.exists(best_path):
        try:
            torch.load(best_path, map_location=lambda storage, loc: storage)
        except:
            unlink(best_path)
    while not train_progress["stop"]:
        if train_progress["current"]["epoch"] > last_epoch or train_progress["current"]["batch"] > last_batch:
            if train_progress["current"]["patience_counter"] == Config.early_stop_patience:
                print("stop training")
                train_progress["stop"] = True
                break
            elapsed = time.time() - start_time
            steps_done = train_progress["current"]["epoch"] * total_batches + train_progress["current"]["batch"]
            steps_total = total_epochs * total_batches
            avg_step_time = elapsed / (steps_done - last_pos)
            eta = (steps_total - steps_done) * avg_step_time
            pbar.n = steps_done
            pbar.set_postfix({
                "Epoch": f"{train_progress['current']['epoch'] + 1}/{total_epochs}",
                "Batch": f"{train_progress['current']['batch'] + 1}/{total_batches}",
                "Elapsed": format_seconds(elapsed),
                "ETA": format_seconds(eta),
            })
            pbar.refresh()
            save_path = Config.model_save_path
            if "last_epoch" in train_progress:
                json.dump(train_progress["full_indices"], open(datasets_indices_path, "w"))
                try:
                    if os.path.exists(best_path):
                        shutil.copy(best_path, best_path.replace(".pt", "_bak.pt"))
                    torch.save(train_progress["last_epoch"], best_path)
                    del train_progress["last_epoch"]
                except:
                    raise RuntimeError("保存训练模型失败")
            else:
                try:
                    if os.path.exists(save_path):
                        shutil.copy(save_path, bak_save_path)
                    torch.save(train_progress["current"], save_path)
                    sleep(60)
                except:
                    raise RuntimeError("保存训练模型失败")
            last_epoch, last_batch = train_progress["current"]["epoch"], train_progress["current"]["batch"]
    if train_progress["last_epoch"]:
        torch.save(train_progress["last_epoch"], best_path)
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
    if os.path.exists(bak_save_path):
        try:
            checkpoint = torch.load(bak_save_path, map_location=Config.device, weights_only=False)
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
else:
    best_val_loss = float('inf')
    patience_counter = 0
train_progress["current"] = checkpoint
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=2)
# 更新全局状态
train_progress["current"]["epoch"] = start_epoch
train_progress["current"]["batch"] = start_batch
train_progress["current"]["best_val_loss"] = best_val_loss
train_progress["current"]["patience_counter"] = patience_counter
train_progress["full_indices"] = full_indices

# 启动监控线程
monitor_thread = threading.Thread(target=monitor_thread_func)
monitor_thread.start()

for epoch in range(start_epoch, Config.epochs):
    train_dataset = torch.utils.data.Subset(full_dataset, full_indices["train"])
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=2)
    model.train()
    #total_loss = 0.0

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

        #total_loss += loss.item()

        # 更新进度状态（无任何print）
        train_progress["current"]["epoch"] = epoch
        train_progress["current"]["batch"] = batch_idx
        train_progress["current"]["model_state"] = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        train_progress["current"]["optimizer_state"] = optimizer.state_dict()
        train_progress["current"]["rng_state"] = torch.get_rng_state()
        train_progress["current"]["cuda_rng_state"] = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        if scaler is not None:
            train_progress["current"]["scaler_state"] = scaler.state_dict()
    val_loss = validate(model, val_loader, criterion_kind, criterion_gender, criterion_age, Config.device)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        train_progress["current"]["patience_counter"] = 0
        train_progress["current"]["best_val_loss"] = best_val_loss
        train_progress["last_epoch"] = {
            "model_state": train_progress["current"]["model_state"],
            "optimizer_state": train_progress["current"]["optimizer_state"],
            "best_val_loss": best_val_loss
        }
    else:
        patience_counter += 1
        train_progress["current"]["patience_counter"] = patience_counter
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