import logging
from typing import List, Dict, Any

from config import Config

class RejectionSampler:
    def __init__(self, config: Config):
        self.config = config
        
    def select_best_candidate(self, scored_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从评分后的候选结果中选择最佳结果"""
        if not scored_candidates:
            logging.error("没有可选的候选结果")
            raise ValueError("没有可选的候选结果")
        
        # 按分数降序排序
        sorted_candidates = sorted(
            scored_candidates, 
            key=lambda x: x["score"], 
            reverse=True
        )
        
        # 找到分数大于阈值的候选结果
        high_quality_candidates = [
            c for c in sorted_candidates 
            if c["score"] >= self.config.score_threshold
        ]
        
        # 如果有高质量候选，返回最高分的
        if high_quality_candidates:
            best_candidate = high_quality_candidates[0]
            logging.info(f"找到高质量候选结果，分数: {best_candidate['score']:.1f}")
            return best_candidate
        
        # 否则返回最高分的候选
        best_candidate = sorted_candidates[0]
        logging.info(f"没有找到高质量候选结果，使用最高分候选，分数: {best_candidate['score']:.1f}")
        return best_candidate
    
    def analyze_candidates(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析候选结果"""
        if not candidates:
            return {}
            
        scores = [c["score"] for c in candidates]
        params = [c["params"] for c in candidates]
        prompts = [c["prompt_key"] for c in candidates]
        
        # 找出最佳参数组合
        best_idx = scores.index(max(scores))
        best_params = params[best_idx]
        best_prompt = prompts[best_idx]
        
        # 统计信息
        analysis = {
            "best_score": max(scores),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "score_range": max(scores) - min(scores),
            "best_params": best_params,
            "best_prompt": best_prompt
        }
        
        return analysis
