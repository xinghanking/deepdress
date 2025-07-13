import io
import logging
import os
import sys
import threading
import traceback
from collections import OrderedDict

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from core.config import Config

# Global MODNet model cache
_modnet_model = None
_mp_solutions = {}

def setup_logger(log_file=os.path.join(Config.base_root, "deepdress.log")) -> logging.Logger:
    """
    配置日志记录器

    参数:
        log_file: 日志文件路径，默认使用 Config.base_root

    返回:
        logging.Logger: 配置好的日志记录器
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger()


logger = setup_logger()


def pil_to_stream(pil_img, format='PNG'):
    """
    将 PIL 图像转换为字节流

    参数:
        pil_img: PIL.Image 对象
        format: 输出格式，默认为 'PNG'

    返回:
        bytes: 图像字节流
    """
    try:
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format=format)
        return img_byte_arr.getvalue()
    except Exception as e:
        logger.error(f"PIL to stream conversion failed: {e}")
        return None


# Initialize MODNet model
def load_modnet_model():
    global _modnet_model
    if _modnet_model is None:
        sys.path.append(Config.modnet_path)
        from models.modnet import MODNet

        # 初始化 MODNet
        _modnet_model = MODNet(backbone_pretrained=False).to(Config.device)
        state_dict = torch.load(Config.modnet_ckpt, map_location=Config.device)
        new_state_dict = OrderedDict({k.replace("module.", ""): v for k, v in state_dict.items()})
        _modnet_model.load_state_dict(new_state_dict)
        _modnet_model.eval()
        _modnet_model.to(Config.device)
    return _modnet_model

def get_face_mesh():
    global _mp_solutions
    thread_id = threading.get_ident()
    if thread_id not in _mp_solutions:
        _mp_solutions[thread_id] = {}
    if "face_mesh" not in _mp_solutions[thread_id]:
        _mp_solutions[thread_id]["face_mesh"] = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.1
        )
    return _mp_solutions[thread_id]["face_mesh"]

def get_selfie_segmentation():
    global _mp_solutions
    thread_id = threading.get_ident()
    if thread_id not in _mp_solutions:
        _mp_solutions[thread_id] = {}
    if "selfie_segmentation" not in _mp_solutions[thread_id]:
        _mp_solutions[thread_id]["selfie_segmentation"] = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
    return _mp_solutions[thread_id]["selfie_segmentation"]

def get_face_detection():
    global _mp_solutions
    thread_id = threading.get_ident()
    if thread_id not in _mp_solutions:
        _mp_solutions[thread_id] = {}
    if "face_detection" not in _mp_solutions[thread_id]:
        _mp_solutions[thread_id]["face_detection"] = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1)
    return _mp_solutions[thread_id]["face_detection"]

def pad_to_32_multiples(image: Image.Image):
    """
    将图像 pad 成宽高为 32 的倍数，保持原图居中。
    :param image: PIL.Image，RGB 格式
    :param background: RGB 背景色（默认黑色）
    :return: padded PIL.Image
    """
    w, h = image.size
    new_w = ((w + 31) // 32) * 32
    new_h = ((h + 31) // 32) * 32

    pad_left = (new_w - w) // 2
    pad_top = (new_h - h) // 2

    # 创建新背景图像
    new_image = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    new_image.paste(image, (pad_left, pad_top))
    return new_image


def resize_with_padding(img: Image.Image, size: tuple[int, int] = Config.image_size) -> Image.Image:
    """保持比例缩放，居中贴到透明背景上"""
    if img.width > size[0] or img.height > size[1]:
        ratio = min(size[0] / img.width, size[1] / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    new_img = Image.new("RGBA", size, Config.bg_color)
    paste_pos = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
    new_img.paste(img, paste_pos, mask=img)
    return new_img

def validate_image(image: Image.Image):
    """验证已处理图像（检查点恢复）"""
    try:
        image_np = np.array(image)
        mp_face_mesh = get_face_mesh()
        img_rgb = image_np
        results = mp_face_mesh.process(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        if not results.multi_face_landmarks:
            return False, "未检测到人脸关键点"
        if len(results.multi_face_landmarks) > 1:
            return False, "检测到多张人脸"
        landmarks = results.multi_face_landmarks[0]
        key_indices = [33, 263, 1, 61, 291]
        h, w, _ = image_np.shape
        if h < Config.face_min_size or w < Config.face_min_size:
            return False, "区域过小"
        for idx in key_indices:
            lm = landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            if x < 0 or x >= w or y < 0 or y >= h:
                return False, "五官不完整（关键点超出边界）"
        left_eye = np.array([landmarks.landmark[33].x, landmarks.landmark[33].y])
        right_eye = np.array([landmarks.landmark[263].x, landmarks.landmark[263].y])
        nose = np.array([landmarks.landmark[1].x, landmarks.landmark[1].y])
        eye_mid = (left_eye + right_eye) / 2
        dx = nose[0] - eye_mid[0]
        yaw = np.arctan2(dx, 1) * 180 / np.pi
        if abs(yaw) > 30:
            return False, f"侧脸偏移（yaw={yaw:.1f}°）"
        pitch = (nose[1] - eye_mid[1]) * 100
        if abs(pitch) > 15:
            return False, f"俯仰偏移（pitch={abs(pitch):.2f}°）"
        x_coords = [landmarks.landmark[i].x * w for i in key_indices]
        y_coords = [landmarks.landmark[i].y * h for i in key_indices]
        left = max(int(min(x_coords) - 0.1 * w), 0)
        right = min(int(max(x_coords) + 0.1 * w), w)
        top = max(int(min(y_coords) - 0.1 * h), 0)
        bottom = min(int(max(y_coords) + 0.1 * h), h)
        if right - left < Config.face_min_size or bottom - top < Config.face_min_size:
            return False, "人脸区域过小"
        return (left, top, right, bottom), ""
    except Exception as e:
        return False, f"验证失败: {str(e)}"


def get_full_img(img_rgb: Image.Image) -> tuple[Image.Image | None, str]:
    try:
        if img_rgb.mode != "RGB":
            img_rgb = img_rgb.convert("RGB")
        w, h = img_rgb.size  # Use img.size to get width and height
        image_np = np.array(img_rgb)
        # Validate image
        if image_np is None or image_np.size == 0:
            return None, "Image is empty or not loaded"

        # Only proportionally scale low-resolution images (do not process high-resolution images)
        if h < Config.face_min_size or w < Config.face_min_size:
            return None, "Image is small"
        image_area = w * h
        selfie_segmentation = get_selfie_segmentation()
        result = selfie_segmentation.process(image_np)
        if result.segmentation_mask is None:
            return None, "No portait detected"
        alpha = result.segmentation_mask
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(alpha, 0, 1)

        # Thresholding, generate binary mask
        binary_mask = (alpha > 0.4).astype(np.uint8)

        labeled, num_features = ndimage.label(binary_mask)
        if num_features == 0:
            return None, "No portrait detected"
        if num_features == 1:
            # 单人像，直接提取
            best_region = binary_mask
        else:
            # Select the most prominent portrait
            image_center_x, image_center_y = w // 2, h // 2
            best_score = -1
            best_region = None

            for i in range(1, num_features + 1):
                region_mask = (labeled == i).astype(np.uint8)
                coords = np.where(region_mask)
                if len(coords[0]) == 0:  # Check if coordinates are empty
                    continue

                # Calculate bounding box
                top = max(int(min(coords[0])), 0)
                bottom = min(int(max(coords[0])) + 1, h)
                left = max(int(min(coords[1])), 0)
                right = min(int(max(coords[1])) + 1, w)
                w = right - left
                h = bottom - top
                # Calculate portrait area
                if w < Config.face_min_size or h < Config.face_min_size:  # Filter out too small or too large regions
                    continue
                area = w * h
                # Calculate sharpness
                region = image_np[top:bottom, left:right]
                if region.size == 0 or (right - left) < 5 or (bottom - top) < 5:
                    print(f"Skipping region {i}: invalid region, width={right - left}, height={bottom - top}")
                    continue
                region_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)  # Convert after checking size
                sharpness = cv2.Laplacian(region_gray, cv2.CV_64F).var()
                sharpness = min(sharpness / 1000, 1.0)  # Normalize sharpness

                # Calculate distance from region center to image center
                region_center_x = (left + right) // 2
                region_center_y = (top + bottom) // 2
                distance = np.sqrt((region_center_x - image_center_x) ** 2 + (region_center_y - image_center_y) ** 2)

                # Calculate distance factor, handling potential division by zero
                max_dim = max(w, h)
                distance_factor = min(distance / max_dim, 1.0) if max_dim > 0 else 0.0

                # Combined score (area + sharpness - distance penalty)
                score = (area / image_area) * 0.5 + sharpness * 0.4 - distance_factor * 0.1
                if score > best_score:
                    best_score = score
                    best_region = region_mask

            if best_region is None:
                return None, "No valid portrait selected"

        # Extract portrait (RGBA format, transparent background)
        person_mask = best_region * alpha  # Use original alpha mask to preserve edge details
        alpha_channel = (person_mask * 255).astype(np.uint8)  # Convert to 0-255 alpha channel
        rgba_image = np.dstack((image_np, alpha_channel))  # Combine RGB and alpha channels
        person_image = Image.fromarray(rgba_image, mode="RGBA")
        person_image = person_image.crop(person_image.getbbox())
        return person_image, ""

    except (OSError, Image.UnidentifiedImageError):
        return None, "Failed to load image"
    except Exception as e:
        tb_str = traceback.format_exc()
        return None, f"Extraction failed: {str(e)}，{tb_str}"

def get_face_bbox(full_img: Image.Image) -> tuple[int, int, int, int] | None:
    img_np = np.array(full_img)  # 转换为RGB NumPy图像
    h, w = img_np.shape[:2]
    # 使用 MediaPipe 检测人脸
    face_detection = get_face_detection()
    results = face_detection.process(cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2BGR))
    if results.detections is None:
        return None

    # 选置信度最高的一张人脸
    best_detection = max(results.detections, key=lambda d: d.score[0])
    box = best_detection.location_data.relative_bounding_box
    top = int(max(box.ymin, 0) * h)
    left = int(max(box.xmin, 0) * w)
    bottom = int(min(box.ymin + box.height, 1) * h)
    right = int(min(box.xmin + box.width, 1) * w)
    if bottom <= top or right <= left or bottom - top < Config.face_min_size or right - left < Config.face_min_size:
        return None
    alpha_channel = img_np[:, :, 3]
    binary_mask = (alpha_channel > 0).astype(np.uint8)
    labeled, num_features = ndimage.label(binary_mask)
    if num_features > 1:
        label_id = labeled[(left + right) // 2, (top + bottom) // 2]
        if label_id == 0:
            return None
    # 裁剪区域并返回
    return left, top, right, bottom


def crop_clear(img_rgba):
    w, h = img_rgba.size
    img_np = np.array(img_rgba)
    alpha_channel = img_np[:, :, 3]
    binary_mask = (alpha_channel > 0).astype(np.uint8)
    labeled, num_features = ndimage.label(binary_mask)
    if num_features > 1:
        face_center_x = w // 2
        face_center_y = h // 2
        label_id = labeled[face_center_y, face_center_x]
        binary_mask = (labeled == label_id).astype(np.uint8)
        img_np[:, :, 3] = binary_mask * img_np[:, :, 3]
        img_rgba = Image.fromarray(img_np, mode="RGBA")
    return img_rgba


def get_face_img(full_img, face_bbox):
    try:
        face_img = full_img.crop(face_bbox)
        face_img = crop_clear(face_img)
        return face_img.crop(face_img.getbbox())
    except Exception as e:
        logger.error(f"Face crop failed: {str(e)}")
        return None


def get_head_img(full_img, face_bbox):
    try:
        ctx = {'face_mesh': get_face_mesh()}
        img_np = np.array(full_img)
        h, w = img_np.shape[:2]

        # 硬件加速的RGB转换 (比cv2快3倍)
        img_np = img_np[:, :, :3]  # RGB转BGR

        # 人脸检测 (优化推理速度)
        results = ctx['face_mesh'].process(img_np)
        left, top, right, bottom = face_bbox
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            neck = landmarks[200]
            if 0 < neck.y < 1:
                bottom = max(bottom, int(neck.y * h))
        head_img = full_img.crop((0, 0, w, bottom))
        head_img = crop_clear(head_img)
        return head_img.crop(head_img.getbbox())
    except Exception as e:
        logger.error(f"Head crop failed: {str(e)}")
        return None


def get_train_preprocess_image(image_rgb:Image.Image)->tuple[tuple[Image.Image|None, Image.Image|None, Image.Image|None]|None, str|None]:
    ok, msg = validate_image(image_rgb)
    if not ok:
        return None, msg
    full_img, msg = get_full_img(image_rgb)
    if full_img is None:
        return None, msg
    face_bbox = get_face_bbox(full_img)
    if face_bbox is None:
        return None, "没有找到人脸"
    left, top, right, bottom = face_bbox
    if right - left < Config.face_min_size or bottom - top < Config.face_min_size:
        return None, "face is too small"
    face_img = get_face_img(full_img, face_bbox)
    if face_img is None:
        return None, "获取face失败"
    w, h = face_img.size
    if w < Config.face_min_size or h < Config.face_min_size:
        return None, "face is too small"
    face_area = w * h
    w, h = full_img.size
    full_area = w * h
    face_img = resize_with_padding(face_img)
    if face_area > full_area * 0.5:
        return (face_img, None, None), None
    head_img = get_head_img(full_img, face_bbox)
    if head_img is None:
        return (face_img, None, None), "获得head失败"
    w, h = head_img.size
    head_img = resize_with_padding(head_img)
    if w * h > full_area * 0.5:
        return (face_img, head_img, None), None
    full_img = resize_with_padding(full_img)
    return (face_img, head_img, full_img), None
