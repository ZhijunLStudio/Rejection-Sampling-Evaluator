import re
import time
import logging
import asyncio
from typing import Dict, List, Any

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
    
    async def grade_multiple_candidates(self, session, prompt: str, candidates: List[Dict[str, Any]], image_base64: str) -> Dict[str, Any]:
        """评分多个候选结果，同时考虑图像内容"""
        # 应用速率限制
        await self._rate_limit()
        
        start_time = time.time()
        
        # 构建候选答案字符串
        candidate_texts = []
        for i, candidate in enumerate(candidates):
            candidate_texts.append(f"Candidate #{i+1}:\nPrompt: {candidate['prompt_key']}\nParameters: temp={candidate['params']['temperature']}, top_p={candidate['params']['top_p']}, top_k={candidate['params']['top_k']}\n\n{candidate['content']}\n\n")
        
        candidate_answers = "\n---\n".join(candidate_texts)
        
        # 填充提示模板
        formatted_prompt = prompt.replace("{candidate_answers}", candidate_answers)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 构建包含图像的消息
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": formatted_prompt},
                {"type": "image_url", "image_url": {"url": image_base64}}
            ]}
        ]
        
        data = {
            "model": self.model,
            "messages": messages
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
                    return {"error": error_msg, "content": "", "scores": [], "best_index": -1, "usage": {}, "latency": end_time - start_time}
                
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    content = response_json["choices"][0]["message"]["content"]
                    
                    # 提取各候选结果的分数
                    scores = self._extract_multiple_scores(content, len(candidates))
                    
                    # 确定最高分的索引
                    best_index = -1
                    if scores:
                        best_index = scores.index(max(scores))
                    
                    result = {
                        "content": content,
                        "scores": scores,
                        "best_index": best_index,
                        "usage": response_json.get("usage", {}),
                        "latency": end_time - start_time
                    }
                    return result
                else:
                    error_msg = f"无效的评分API响应格式: {response_json}"
                    logging.error(error_msg)
                    return {"error": error_msg, "content": "", "scores": [], "best_index": -1, "usage": {}, "latency": end_time - start_time}
                
        except asyncio.TimeoutError:
            error_msg = "评分请求超时"
            logging.error(error_msg)
            return {"error": error_msg, "content": "", "scores": [], "best_index": -1, "usage": {}, "latency": time.time() - start_time}
            
        except Exception as e:
            error_msg = f"评分请求出错: {str(e)}"
            logging.error(error_msg)
            return {"error": error_msg, "content": "", "scores": [], "best_index": -1, "usage": {}, "latency": time.time() - start_time}
    
    def _extract_multiple_scores(self, text: str, num_candidates: int) -> List[float]:
        """从评分文本中提取多个候选的分数"""
        scores = []
        try:
            # 尝试为每个候选结果找到分数
            for i in range(num_candidates):
                candidate_num = i + 1
                
                # 尝试多种可能的提取模式
                patterns = [
                    rf"Candidate #{candidate_num}[^0-9]*(\d+)(?:\.\d+)?(?:/100)?",
                    rf"Candidate {candidate_num}[^0-9]*(\d+)(?:\.\d+)?(?:/100)?",
                    rf"#{candidate_num}[^0-9]*(\d+)(?:\.\d+)?(?:/100)?",
                    rf"得分.{candidate_num}.+?(\d+)(?:\.\d+)?",
                    rf"score for candidate #{candidate_num}[^0-9]*(\d+)(?:\.\d+)?",
                    rf"score for candidate {candidate_num}[^0-9]*(\d+)(?:\.\d+)?"
                ]
                
                score_found = False
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if match:
                        score = float(match.group(1))
                        scores.append(score)
                        score_found = True
                        break
                
                if not score_found:
                    # 如果找不到特定候选结果的分数，尝试在文本的相应部分提取数字
                    parts = text.split(f"Candidate #{candidate_num}")
                    if len(parts) > 1:
                        # 在相应部分中寻找第一个有效分数
                        candidate_part = parts[1]
                        # 找到下一个候选或文本末尾
                        if i < num_candidates - 1:
                            next_candidate = f"Candidate #{candidate_num + 1}"
                            if next_candidate in candidate_part:
                                candidate_part = candidate_part.split(next_candidate)[0]
                        
                        # 在这一部分中查找分数
                        number_match = re.search(r"(\d{1,3})(?:\.\d+)?(?:/100)?", candidate_part)
                        if number_match:
                            score = float(number_match.group(1))
                            if 0 <= score <= 100:  # 验证分数范围
                                scores.append(score)
                                score_found = True
                
                # 如果仍未找到分数，添加默认值
                if not score_found:
                    logging.warning(f"无法找到候选 #{candidate_num} 的分数，使用默认值 0")
                    scores.append(0.0)
            
            return scores
            
        except Exception as e:
            logging.error(f"提取多个分数时出错: {str(e)}")
            # 返回默认分数列表
            return [0.0] * num_candidates
            
    # 保留旧方法以兼容性
    async def grade(self, session, prompt: str) -> Dict[str, Any]:
        """原始评分方法（已废弃）"""
        logging.warning("使用了废弃的grade方法，请改用grade_multiple_candidates")
        
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
            ]
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
                
        except Exception as e:
            error_msg = f"评分请求出错: {str(e)}"
            logging.error(error_msg)
            return {"error": error_msg, "content": "", "score": 0, "usage": {}, "latency": time.time() - start_time}
