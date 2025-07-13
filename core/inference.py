import torch

from core.config import Config
from core.model import MultiTaskModel
from core.utils import get_face_bbox, get_face_img, get_head_img, resize_with_padding

# 初始化模型并加载权重
model = None


def predict(image_rgba):
    global model
    if model is None:
        model = MultiTaskModel()
        ckpt = torch.load(Config.model_save_path, map_location=Config.device)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
        model = model.to(Config.device)
        model.eval()
    input_tensor = model.preprocess(image_rgba).unsqueeze(0).to(Config.device)
    with torch.no_grad():
        kind_logits, gender_logits, age_output = model(input_tensor)
        kind = kind_logits.argmax(dim=1).item()
        gender = gender_logits.argmax(dim=1).item()
        age = round(age_output.squeeze().item())
    return kind, gender, age

def get_dress_info(image_rgb):
    kind, gender, age = predict(image_rgb)
    if kind == 0:
        return {
            "face": {"gender": gender, "age": age},
            "head": {"gender": gender, "age": age},
            "full": {"gender": gender, "age": age},
        }
    face_bbox = get_face_bbox(image_rgb)
    if face_bbox is None:
        return {
            "face": {"gender": gender, "age": age},
            "head": {"gender": gender, "age": age},
            "full": {"gender": gender, "age": age},
        }
    face_img = get_face_img(image_rgb, face_bbox)
    _, face_gender, face_age = predict(face_img)
    if kind == 1:
        return {
            "face": {"gender": face_gender, "age": face_age},
            "head": {"gender": gender, "age": age},
            "full": {"gender": gender, "age": age},
        }
    head_img = get_head_img(image_rgb, face_bbox)
    _, head_gender, head_age = predict(head_img)
    return {
        "face": {"gender": face_gender, "age": face_age},
        "head": {"gender": head_gender, "age": head_age},
        "full": {"gender": gender, "age": age},
    }