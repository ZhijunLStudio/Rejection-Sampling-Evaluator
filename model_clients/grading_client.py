import re
import time
import logging
import asyncio
from typing import Dict, Any

import aiohttp
from config import Config

class GradingClient:
    def __init__(self, config: Config):
        self.config = config
        self.api_base = config.grading_api_base
        self.api_key = config.grading_api_key
        self.model = config.grading_model
        
        # 限速控制
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 限制每秒最多5个评分请求
        
    async def _rate_limit(self):
        """实现简单的请求速率限制"""
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
            
        self.last_request_time = time.time()
    
    async def grade(self, session, prompt: str) -> Dict[str, Any]:
        """调用评分API评估生成的答案"""
        # 应用速率限制
        await self._rate_limit()
        
        start_time = time.time()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # 低温度使评分更加确定性
            "max_tokens": 1000
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
                    error_msg = f"评分API错误: 状态码 {response.status}, 响应: {response_json}"
                    logging.error(error_msg)
                    return {"error": error_msg, "content": "", "score": 0, "usage": {}, "latency": end_time - start_time}
                
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    content = response_json["choices"][0]["message"]["content"]
                    score = self._extract_score(content)
                    
                    result = {
                        "content": content,
                        "score": score,
                        "usage": response_json.get("usage", {}),
                        "latency": end_time - start_time
                    }
                    return result
                else:
                    error_msg = f"无效的评分API响应格式: {response_json}"
                    logging.error(error_msg)
                    return {"error": error_msg, "content": "", "score": 0, "usage": {}, "latency": end_time - start_time}
                
        except asyncio.TimeoutError:
            error_msg = "评分请求超时"
            logging.error(error_msg)
            return {"error": error_msg, "content": "", "score": 0, "usage": {}, "latency": time.time() - start_time}
            
        except Exception as e:
            error_msg = f"评分请求出错: {str(e)}"
            logging.error(error_msg)
            return {"error": error_msg, "content": "", "score": 0, "usage": {}, "latency": time.time() - start_time}
    
    def _extract_score(self, text: str) -> float:
        """从评分文本中提取分数"""
        try:
            # 尝试多种可能的格式提取分数，英文版
            patterns = [
                r'score[: ]*(\d+(?:\.\d+)?)',
                r'rating[: ]*(\d+(?:\.\d+)?)',
                r'grade[: ]*(\d+(?:\.\d+)?)',
                r'(\d+)[/]100',
                r'^(\d+)$'  # 单独的数字
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    # 确保分数在0-100范围内
                    return max(0, min(100, score))
            
            # 如果没有找到明确的分数格式，尝试提取任意数字
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
            for num in numbers:
                score = float(num)
                if 0 <= score <= 100:
                    return score
            
            # 如果无法提取分数，返回默认值
            logging.warning(f"无法从文本中提取分数: {text}")
            return 0
        except Exception as e:
            logging.error(f"提取分数时出错: {str(e)}")
            return 0
