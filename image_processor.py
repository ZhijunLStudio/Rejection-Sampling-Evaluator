import base64
import logging
from io import BytesIO
from typing import Dict
from PIL import Image

class ImageProcessor:
    def __init__(self):
        # 图像缓存
        self._image_cache = {}
        
    def encode_image(self, image_path: str) -> str:
        """将图像编码为Base64字符串，带缓存优化"""
        # 检查缓存
        if image_path in self._image_cache:
            return self._image_cache[image_path]
            
        try:
            with Image.open(image_path) as image:
                # 将图像转换为RGB模式（处理RGBA等情况）
                image = image.convert("RGB")
                
                # 保存到内存缓冲区
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format="PNG")
                
                # 编码为base64
                base64_data = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                result = f"data:image/png;base64,{base64_data}"
                
                # 更新缓存
                self._image_cache[image_path] = result
                
                return result
        except Exception as e:
            logging.error(f"图像处理错误: {str(e)}")
            raise Exception(f"图像处理错误: {str(e)}")
    
    def clear_cache(self):
        """清除图像缓存"""
        self._image_cache.clear()
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self._image_cache)
