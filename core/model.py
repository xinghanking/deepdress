import numpy as np
import torch
import torch.nn as nn
import open_clip
from core.config import Config

class MultiTaskModel(nn.Module):
    def __init__(
        self,
        backbone=Config.clip_model,
        pretrained=Config.clip_pretrain,
    ):
        super().__init__()
        self.device = Config.device
        self.in_chans = 4  # 固定4通道输入

        # 创建模型和预处理（默认是3通道预训练模型）
        self.backbone, _, _ = open_clip.create_model_and_transforms(
            model_name=backbone,
            pretrained=pretrained
        )

        # 强制修改ViT的patch_embed以支持4通道输入
        proj = self.backbone.visual.conv1
        old_weight = proj.weight.data
        # 新权重：前三通道用原权重，第4通道初始化为0（也可以随机）
        new_weight = torch.zeros(
            (old_weight.shape[0], 4, old_weight.shape[2], old_weight.shape[3]),
            dtype=old_weight.dtype,
            device=old_weight.device
        )
        new_weight[:, :3, :, :] = old_weight
        proj.weight = nn.Parameter(new_weight)
        proj.in_channels = 4

        self.dropout = nn.Dropout(0.2)
        out_dim = self.backbone.visual.output_dim
        self.kind_head = nn.Linear(out_dim, 3)
        self.gender_head = nn.Linear(out_dim, 2)
        self.age_head = nn.Linear(out_dim, 1)

    def preprocess(self, image):
        rgba = np.array(image).astype(np.float32) / 255.0
        rgba = torch.tensor(rgba, dtype=torch.float32).permute(2, 0, 1)
        mean = torch.tensor(Config.normalize_mean)
        std = torch.tensor(Config.normalize_std)
        rgba = (rgba - mean[:, None, None]) / std[:, None, None]
        return rgba

    def forward(self, images):
        # images: (B, 4, H, W) 固定4通道输入
        features = self.backbone.encode_image(images)
        features = self.dropout(features)
        kind_logits = self.kind_head(features)
        gender_logits = self.gender_head(features)
        age_output = self.age_head(features)
        return kind_logits, gender_logits, age_output