import torch
import os


class Config:
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径
    base_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))  # 项目根目录
    data_root = os.path.join(base_root, "datasets", "processed")  # 预处理数据集目录
    csv_path = os.path.join(base_root, "datasets", "train.csv")  # 训练 CSV 文件
    checkpoint_dir = os.path.join(base_root, "core", "checkpoints")  # 模型检查点目录
    model_save_path = os.path.join(checkpoint_dir, "deepdress.pt")  # 模型检查点路径
    train_progress_path = os.path.join(checkpoint_dir, "train_progress.txt")  # 训练进度文件

    modnet_path = os.path.join(base_root, "lib", "modnet")
    modnet_ckpt = os.path.join(modnet_path, "modnet_photographic_portrait_matting.ckpt")

    # 模型参数
    clip_model = "ViT-B-32-quickgelu"  # CLIP 模型类型
    clip_pretrain = "openai"  # CLIP 预训练权重

    # 训练参数
    batch_size = 16  # 批次大小
    lr = 1e-4  # 学习率
    epochs = 20  # 训练轮数
    val_ratio = 0.2  # 验证集比例
    early_stop_patience = 8
    use_amp = torch.cuda.is_available()
    seed = 42

    # 预处理参数# 填充颜色（resize_with_padding）
    image_size = (224, 224)  # 图像目标尺寸
    bg_color = (0, 0, 0, 0)
    face_min_size = 48
    normalize_mean = [0.48145466, 0.4578275, 0.40821073, 0.0]  # 归一化均值
    normalize_std = [0.26862954, 0.26130258, 0.27577711, 1.0]  # 归一化标准差