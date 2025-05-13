import time
import logging
import asyncio
from typing import Dict, Any

import aiohttp
from config import Config

class VLLMClient:
    def __init__(self, config: Config):
        self.config = config
        self.api_base = config.vllm_api_base
        self.model = config.vllm_model
        
        # 限速控制
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 限制每秒最多10个请求
        
    async def _rate_limit(self):
        """实现简单的请求速率限制"""
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
            
        self.last_request_time = time.time()
    
    async def generate(
        self, 
        session, 
        prompt: str, 
        image_base64: str, 
        temperature: float = 0.7, 
        top_p: float = 0.9, 
        top_k: int = 50,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """调用VLLM API生成回答"""
        # 应用速率限制
        await self._rate_limit()
        
        start_time = time.time()
        
        headers = {
            "Content-Type": "application/json"
        }
        
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_base64}}
            ]}
        ]
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens
        }
        
        try:
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers=headers, 
                json=data,
                timeout=60
            ) as response:
                response_json = await response.json()
                end_time = time.time()
                
                if response.status != 200:
                    error_msg = f"API错误: 状态码 {response.status}, 响应: {response_json}"
                    logging.error(error_msg)
                    return {"error": error_msg, "content": "", "usage": {}, "latency": end_time - start_time}
                
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    result = {
                        "content": response_json["choices"][0]["message"]["content"],
                        "usage": response_json.get("usage", {}),
                        "latency": end_time - start_time
                    }
                    return result
                else:
                    error_msg = f"无效的API响应格式: {response_json}"
                    logging.error(error_msg)
                    return {"error": error_msg, "content": "", "usage": {}, "latency": end_time - start_time}
                
        except asyncio.TimeoutError:
            error_msg = "请求超时"
            logging.error(error_msg)
            return {"error": error_msg, "content": "", "usage": {}, "latency": time.time() - start_time}
            
        except Exception as e:
            error_msg = f"请求出错: {str(e)}"
            logging.error(error_msg)
            return {"error": error_msg, "content": "", "usage": {}, "latency": time.time() - start_time}
