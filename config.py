from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Config:
    # 路径配置
    image_dir: str = "images/"  # 从目录读取图片
    output_dir: str = "results/"
    prompt_path: str = "prompts/generation_prompts.json"
    grading_prompt_path: str = "prompts/grading_prompt.json"
    
    # 模型配置
    vllm_api_base: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen/Qwen2.5-VL"
    
    grading_api_base: str = "https://api.openai.com/v1"
    grading_api_key: str = "your-api-key"
    grading_model: str = "o4-mini"
    
    # Prompt选择
    prompt_keys: List[str] = None  # 指定要使用的prompt键
    
    # 采样配置
    temperature_range: List[float] = field(default_factory=lambda: [0.3, 0.7, 1.0])
    top_p_range: List[float] = field(default_factory=lambda: [0.7, 0.9])
    top_k_range: List[int] = field(default_factory=lambda: [40, 60, 80])
    score_threshold: float = 80.0  # 高质量结果分数阈值
    
    # 处理配置
    eval_samples: int = -1  # -1表示评估所有样本
    gen_workers: int = 4    # 生成模型最大并发数
    grade_workers: int = 8  # 评分模型最大并发数
    checkpoint_interval: int = 10  # 检查点保存间隔
    
    # 调试配置
    verbose: bool = True
