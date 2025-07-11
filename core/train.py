import os
import sys
import random
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# 支持在任意目录下直接运行
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import Config
from core.model import MultiTaskModel

def set_seed():
    torch.manual_seed(Config.seed)
    random.seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.seed)
    import numpy as np
    np.random.seed(Config.seed)

class DeepDressDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5, 0.0], [0.5, 0.5, 0.5, 1.0])
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

def save_checkpoint(model, optimizer, epoch, path, best=False):
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    save_path = path if not best else path.replace(".pt", "_best.pt")
    torch.save({
        "epoch": epoch,
        "model_state": state_dict,
        "optimizer_state": optimizer.state_dict()
    }, save_path)

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

def main():
    set_seed()

    # 数据集加载和划分
    full_dataset = DeepDressDataset(Config.csv_path)
    val_ratio = getattr(Config, "val_ratio", 0.1)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=2)

    # 模型&优化器
    model = MultiTaskModel()
    if torch.cuda.device_count() > 1:
        print(f"⚙️ 使用 {torch.cuda.device_count()} 个 GPU")
        model = nn.DataParallel(model)
    model = model.to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=getattr(Config, "lr", 1e-4))
    criterion_kind = nn.CrossEntropyLoss()
    criterion_gender = nn.CrossEntropyLoss()
    criterion_age = nn.MSELoss()

    # 混合精度
    use_amp = getattr(Config, "use_amp", False)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(Config.model_save_path):
        print("🔄 检测到已保存模型，尝试加载...")
        checkpoint = torch.load(Config.model_save_path, map_location=Config.device)
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"✅ 恢复训练自第 {start_epoch + 1} 轮")

    patience = getattr(Config, "early_stop_patience", 8)
    patience_counter = 0

    with open(Config.train_progress_path, "a") as log_file:
        for epoch in range(start_epoch, Config.epochs):
            model.train()
            total_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.epochs}")
            for images, kind, gender, age in pbar:
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
                pbar.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(train_loader)
            val_loss = validate(model, val_loader, criterion_kind, criterion_gender, criterion_age, Config.device)
            log_msg = f"Epoch {epoch + 1}/{Config.epochs} - Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}\n"
            log_file.write(log_msg)
            print(log_msg.strip())

            # Early stopping & best model保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, Config.model_save_path, best=True)
                patience_counter = 0
                print("🌟 最优模型已保存。")
            else:
                patience_counter += 1
                print(f"⏳ 验证集无提升，early stopping计数：{patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("🛑 达到early stopping条件，训练终止。")
                    break

            save_checkpoint(model, optimizer, epoch, Config.model_save_path)

    print(f"✅ 最优模型已保存到 {Config.model_save_path.replace('.pt', '_best.pt')}")

if __name__ == "__main__":
    main()
