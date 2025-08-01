from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import base64
from io import BytesIO
from PIL import Image
import requests
import os
import sys,traceback

# 解决跨模块导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.inference import get_dress_info
from core.utils import get_full_img, resize_with_padding

router = APIRouter()

class ImageInput(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    image_upload: Optional[UploadFile] = None

@router.post("/get_age")
async def get_info(input: ImageInput):
    try:
        # 1. 加载图片
        if input.image_url:
            try:
                response = requests.get(input.image_url, timeout=5)
                response.raise_for_status()
                if not response.headers["content-type"].startswith("image/"):
                    raise HTTPException(status_code=400, detail=f"图片格式有误")
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"图片下载失败: {str(e)}")
        elif input.image_base64:
            try:
                image_data = base64.b64decode(input.image_base64)
                if image_data is None:
                    raise HTTPException(status_code=400, detail=f"图片格式有误")
                image = Image.open(image_data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Base64解码失败: {str(e)}")
        elif input.image_upload:
            try:
                if not input.image_upload.content_type.startswith("image/"):
                    raise HTTPException(status_code=400, detail=f"上传图片有误")
                image_data = await input.image_upload.read()
                if image_data is None:
                    raise HTTPException(status_code=400, detail=f"上传图片有误")
                image = Image.open(BytesIO(image_data))
                await input.image_upload.close()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"获取失败: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="必须提供 image_url 或 image_base64 或 上传图片")
        if image.mode != "RGB":
            image = image.convert("RGB")
        # 2. 提取人像区域
        image, msg = get_full_img(image)
        if image is None:
            return {"code": 1, "msg": msg}

        # 4. 模型推理
        result = get_dress_info(image)

        # 5. 返回结果
        return {"code": 0, "data": result}

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail=f"服务内部错误: {str(e)}")
