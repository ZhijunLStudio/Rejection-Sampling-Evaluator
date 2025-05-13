#!/usr/bin/env python3
import os
import asyncio
import argparse
import logging
from pathlib import Path

from config import Config
from evaluator import Evaluator
from utils import setup_logging, load_checkpoint, save_checkpoint

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="基于拒绝采样的多模态评测工具")
    
    # 基础配置
    parser.add_argument("--image-dir", type=str, required=True, help="图片目录路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--prompts", type=str, default="prompts/generation_prompts.json", 
                      help="生成提示词文件路径")
    parser.add_argument("--grading-prompt", type=str, default="prompts/grading_prompt.json",
                      help="评分提示词文件路径")
                      
    # 模型配置
    parser.add_argument("--vllm-api", type=str, default="http://localhost:8000/v1", 
                      help="VLLM API地址")
    parser.add_argument("--vllm-model", type=str, default="Qwen/Qwen2.5-VL", 
                      help="VLLM模型名称")
    parser.add_argument("--grading-api", type=str, required=True, 
                      help="评分API地址")
    parser.add_argument("--grading-key", type=str, required=True, 
                      help="评分API密钥")
    parser.add_argument("--grading-model", type=str, default="o4-mini", 
                      help="评分模型名称")
                      
    # Prompt选择
    parser.add_argument("--prompt-keys", type=str, nargs="+", 
                      help="要使用的prompt键列表，例如: prompt1 prompt2 prompt3")
                      
    # 采样配置
    parser.add_argument("--temperature-range", type=str, default="0.3,0.7,1.0", 
                      help="温度参数范围，逗号分隔")
    parser.add_argument("--top-p-range", type=str, default="0.7,0.9", 
                      help="top_p参数范围，逗号分隔") 
    parser.add_argument("--top-k-range", type=str, default="40,60,80", 
                      help="top_k参数范围，逗号分隔")
    parser.add_argument("--score-threshold", type=float, default=80.0, 
                      help="高质量结果分数阈值")
    
    # 处理控制
    parser.add_argument("--gen-workers", type=int, default=4, 
                      help="生成模型最大并发数")
    parser.add_argument("--grade-workers", type=int, default=8, 
                      help="评分模型最大并发数")
    parser.add_argument("--samples", type=int, default=-1, 
                      help="处理样本数量，-1表示全部")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                      help="检查点保存间隔（处理N个样本后）")
    parser.add_argument("--resume", action="store_true",
                      help="从检查点恢复运行")
    parser.add_argument("--log-level", type=str, default="INFO", 
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                      help="日志级别")
    
    return parser.parse_args()

def create_config_from_args(args):
    """从命令行参数创建配置对象"""
    # 解析范围参数
    temp_range = [float(x) for x in args.temperature_range.split(',')]
    top_p_range = [float(x) for x in args.top_p_range.split(',')]
    top_k_range = [int(x) for x in args.top_k_range.split(',')]
    
    # 创建配置对象
    config = Config(
        # 路径配置
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        prompt_path=args.prompts,
        grading_prompt_path=args.grading_prompt,
        
        # 模型配置
        vllm_api_base=args.vllm_api,
        vllm_model=args.vllm_model,
        grading_api_base=args.grading_api,
        grading_api_key=args.grading_key,
        grading_model=args.grading_model,
        
        # Prompt选择
        prompt_keys=args.prompt_keys,
        
        # 采样配置
        temperature_range=temp_range,
        top_p_range=top_p_range,
        top_k_range=top_k_range,
        score_threshold=args.score_threshold,
        
        # 处理配置
        gen_workers=args.gen_workers,
        grade_workers=args.grade_workers,
        eval_samples=args.samples,
        checkpoint_interval=args.checkpoint_interval
    )
    
    return config

async def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 创建配置对象
    config = create_config_from_args(args)
    
    # 确保输出目录存在
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "high_quality"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "low_quality"), exist_ok=True)
    
    # 检查点路径
    checkpoint_path = os.path.join(config.output_dir, "checkpoint.json")
    
    # 加载检查点（如果需要恢复运行）
    checkpoint_data = None
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint_data = load_checkpoint(checkpoint_path)
        logging.info(f"从检查点恢复运行，已完成: {len(checkpoint_data['processed_items'])}个样本")
    
    # 创建评估器
    evaluator = Evaluator(config, checkpoint_data)
    
    try:
        # 运行评估
        await evaluator.run()
        logging.info("评估完成！")
        
    except KeyboardInterrupt:
        logging.info("接收到中断信号，正在保存检查点...")
        save_checkpoint(checkpoint_path, evaluator.get_checkpoint_data())
        logging.info(f"检查点已保存到: {checkpoint_path}")
        return 1
    
    except Exception as e:
        logging.error(f"评估过程出错: {str(e)}", exc_info=True)
        # 尝试保存检查点
        try:
            save_checkpoint(checkpoint_path, evaluator.get_checkpoint_data())
            logging.info(f"错误发生，检查点已保存到: {checkpoint_path}")
        except:
            logging.error("保存检查点失败", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    exit_code = asyncio.run(main())
    exit(exit_code)
