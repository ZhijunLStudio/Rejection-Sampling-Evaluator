import os
import json
import time
import asyncio
import logging
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm

from config import Config
from image_processor import ImageProcessor
from model_clients.vllm_client import VLLMClient
from model_clients.grading_client import GradingClient
from rejection_sampler import RejectionSampler
from utils import save_json, format_timestamp

class Evaluator:
    def __init__(self, config: Config, checkpoint_data: Optional[Dict[str, Any]] = None):
        self.config = config
        self.image_processor = ImageProcessor()
        self.prompts = self._load_prompts()
        self.grading_prompt = self._load_grading_prompt()
        
        # 结果和统计
        self.results = []
        self.stats = {
            "total_samples": 0,
            "processed_samples": 0,
            "high_quality_count": 0,
            "low_quality_count": 0,
            "total_generation_time": 0,
            "total_grading_time": 0,
            "start_time": time.time(),
        }
        
        # 添加: 跟踪最近的高分
        self.recent_scores = []  # 存储最近5个结果的分数
        self.recent_best_scores = []  # 存储最近5个结果的最高分数
        self.recent_max_len = 5  # 保留最近几个样本的数据
        
        # 过滤要使用的prompt
        if config.prompt_keys:
            filtered_prompts = {}
            for key in config.prompt_keys:
                if key in self.prompts:
                    filtered_prompts[key] = self.prompts[key]
                else:
                    logging.warning(f"提示词 '{key}' 不存在于提示词文件中")
            
            if not filtered_prompts:
                logging.error("没有有效的提示词可以使用!")
                raise ValueError("没有有效的提示词可以使用")
                
            self.prompts = filtered_prompts
        
        # 初始化客户端
        self.vllm_client = VLLMClient(config)
        self.grading_client = GradingClient(config)
        self.rejection_sampler = RejectionSampler(config)
        
        # 结果和统计
        self.results = []
        self.stats = {
            "total_samples": 0,
            "processed_samples": 0,
            "high_quality_count": 0,
            "low_quality_count": 0,
            "total_generation_time": 0,
            "total_grading_time": 0,
            "start_time": time.time(),
        }
        
        # 检查点恢复数据
        self.processed_items = set()
        if checkpoint_data:
            self.processed_items = set(checkpoint_data.get('processed_items', []))
            self.stats.update(checkpoint_data.get('stats', {}))
            # 更新开始时间为当前时间
            self.stats['start_time'] = time.time()
            
        # 进度锁
        self.progress_lock = asyncio.Lock()
        
        # 文件锁
        self.file_lock = asyncio.Lock()
    
    def _load_prompts(self) -> Dict[str, str]:
        """加载生成提示词"""
        try:
            with open(self.config.prompt_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"无法加载提示词文件: {str(e)}")
            raise ValueError(f"无法加载提示词文件: {str(e)}")
    
    def _load_grading_prompt(self) -> Dict[str, str]:
        """加载评分提示词"""
        try:
            with open(self.config.grading_prompt_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"无法加载评分提示词文件: {str(e)}")
            raise ValueError(f"无法加载评分提示词文件: {str(e)}")
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """获取检查点数据，用于断点续跑"""
        return {
            'processed_items': list(self.processed_items),
            'stats': self.stats
        }
    
    async def run(self):
        """运行评估流程"""
        # 获取图像文件
        image_files = self._get_image_files()
        if not image_files:
            logging.error(f"在 {self.config.image_dir} 中未找到图像文件")
            return
        
        # 限制样本数量
        if self.config.eval_samples > 0:
            image_files = image_files[:self.config.eval_samples]
        
        # 过滤已处理的项目
        if self.processed_items:
            original_count = len(image_files)
            image_files = [img for img in image_files if img not in self.processed_items]
            logging.info(f"跳过已处理的{original_count - len(image_files)}个样本，继续处理{len(image_files)}个样本")
        
        self.stats['total_samples'] = len(image_files)
        
        # 显示配置信息
        logging.info(f"加载了{len(image_files)}个图像样本进行评估")
        logging.info(f"使用{len(self.prompts)}个不同的提示词: {', '.join(self.prompts.keys())}")
        logging.info(f"生成并发数: {self.config.gen_workers}, 评分并发数: {self.config.grade_workers}")
        logging.info(f"高质量结果阈值: {self.config.score_threshold}")
        
        # 初始化进度条
        total_tasks = len(image_files)
        progress_bar = tqdm(total=total_tasks, desc="评估进度")
        
        # 更新已处理的进度
        progress_bar.update(len(self.processed_items))
        
        # 创建HTTP会话
        async with aiohttp.ClientSession() as session:
            # 限制并发
            gen_semaphore = asyncio.Semaphore(self.config.gen_workers)
            grade_semaphore = asyncio.Semaphore(self.config.grade_workers)
            
            tasks = []
            
            # 创建处理任务
            for i, img_path in enumerate(image_files):
                task = self._process_image(session, gen_semaphore, grade_semaphore, img_path, i, progress_bar)
                tasks.append(task)
            
            # 等待所有任务完成
            await asyncio.gather(*tasks)
        
        # 关闭进度条
        progress_bar.close()
        
        # 保存最终统计结果
        self._save_final_stats()
        
        logging.info(f"评估完成! 处理了 {self.stats['processed_samples']} 个样本")
        logging.info(f"高质量结果: {self.stats['high_quality_count']}, 低质量结果: {self.stats['low_quality_count']}")
    
    def _get_image_files(self) -> List[str]:
        """获取图像目录中所有的图像文件"""
        image_dir = Path(self.config.image_dir)
        if not image_dir.exists() or not image_dir.is_dir():
            logging.error(f"图像目录不存在: {image_dir}")
            return []
            
        # 支持的图像扩展名
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # 收集所有图像文件路径
        image_files = []
        for ext in extensions:
            image_files.extend(list(image_dir.glob(f"*{ext}")))
            image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))
        
        # 转换为字符串路径
        return [str(path) for path in sorted(image_files)]
    
    async def _process_image(self, session, gen_semaphore, grade_semaphore, img_path: str, index: int, progress_bar):
        """处理单个图像"""
        # 如果已经处理过，跳过
        if img_path in self.processed_items:
            return
        
        try:
            # 确认文件存在
            if not os.path.exists(img_path):
                logging.error(f"图片不存在: {img_path}")
                return
                    
            # 编码图片
            image_base64 = self.image_processor.encode_image(img_path)
            
            # 为每个提示词生成结果
            candidate_results = await self._generate_for_all_prompts(session, gen_semaphore, img_path, image_base64)
            if not candidate_results:
                logging.warning(f"无法为图像生成候选结果: {img_path}")
                return
            
            # 一次性评分所有候选结果
            scored_candidates = await self._grade_candidates(session, grade_semaphore, img_path, image_base64, candidate_results)
            if not scored_candidates:
                logging.warning(f"无法为图像评分: {img_path}")
                return
            
            if scored_candidates:
                # 选择最佳结果
                best_result = self.rejection_sampler.select_best_candidate(scored_candidates)
                
                # 保存结果
                await self._save_result(img_path, best_result, scored_candidates)
                
                # 更新最近分数列表
                async with self.progress_lock:
                    # 记录所有候选的平均分
                    avg_score = sum(c["score"] for c in scored_candidates) / len(scored_candidates)
                    self.recent_scores.append(avg_score)
                    
                    # 记录最高分
                    best_score = best_result["score"]
                    self.recent_best_scores.append(best_score)
                    
                    # 保持列表长度
                    if len(self.recent_scores) > self.recent_max_len:
                        self.recent_scores.pop(0)
                    if len(self.recent_best_scores) > self.recent_max_len:
                        self.recent_best_scores.pop(0)
            
            # 选择最佳结果 - 使用评分API确定的最佳结果或分数最高的
            best_result = None
            for candidate in scored_candidates:
                if candidate.get("is_best", False):
                    best_result = candidate
                    break
            
            # 如果评分API没有指定最佳结果，则选择分数最高的
            if not best_result:
                best_result = self.rejection_sampler.select_best_candidate(scored_candidates)
            
            # 保存结果
            await self._save_result(img_path, best_result, scored_candidates)
                    
            # 更新统计信息
            async with self.progress_lock:
                self.stats['processed_samples'] += 1
                if best_result['score'] >= self.config.score_threshold:
                    self.stats['high_quality_count'] += 1
                else:
                    self.stats['low_quality_count'] += 1
            
            # 更新已处理项目
            self.processed_items.add(img_path)
            
            # 更新进度条
            progress_bar.update(1)
            
            # 定期保存检查点
            if index > 0 and index % self.config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(self.config.output_dir, "checkpoint.json")
                checkpoint_data = self.get_checkpoint_data()
                save_json(checkpoint_path, checkpoint_data)
                logging.info(f"检查点已保存 ({index}/{self.stats['total_samples']})")
        
        except Exception as e:
            logging.error(f"处理图像时出错 {img_path}: {str(e)}", exc_info=True)
            
            
        finally:
            # 更新进度条信息
            if self.recent_best_scores:
                recent_max = max(self.recent_best_scores)
                recent_avg = sum(self.recent_best_scores) / len(self.recent_best_scores)
                
                progress_desc = (
                    f"进度: {self.stats['processed_samples']}/{self.stats['total_samples']} | "
                    f"最近最高分: {recent_max:.1f} | "
                    f"最近平均最高分: {recent_avg:.1f} | "
                    f"高/低质量: {self.stats['high_quality_count']}/{self.stats['low_quality_count']}"
                )
                progress_bar.set_description(progress_desc)
            
            # 更新进度条和已处理项
            progress_bar.update(1)
            self.processed_items.add(img_path)

    
    async def _generate_for_all_prompts(self, session, semaphore, img_path: str, image_base64: str) -> List[Dict[str, Any]]:
        """为所有提示词生成结果"""
        all_candidates = []
        
        # 为每个提示词创建任务
        tasks = []
        for prompt_key, prompt_text in self.prompts.items():
            task = self._generate_with_prompt(session, semaphore, img_path, image_base64, prompt_key, prompt_text)
            tasks.append(task)
        
        # 并行执行所有任务
        results = await asyncio.gather(*tasks)
        
        # 收集所有有效结果
        for result in results:
            if result:
                all_candidates.append(result)
        
        return all_candidates
    
    async def _generate_with_prompt(self, session, semaphore, img_path: str, image_base64: str, prompt_key: str, prompt_text: str) -> Optional[Dict[str, Any]]:
        """使用特定提示词生成结果"""
        async with semaphore:
            try:
                # 随机选择参数
                temperature = random.choice(self.config.temperature_range)
                top_p = random.choice(self.config.top_p_range)
                top_k = random.choice(self.config.top_k_range)
                
                # 调用模型生成
                start_time = time.time()
                result = await self.vllm_client.generate(
                    session, 
                    prompt_text, 
                    image_base64, 
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
                generation_time = time.time() - start_time
                
                # 更新统计
                self.stats['total_generation_time'] += generation_time
                
                # 保存结果
                if "error" not in result:
                    return {
                        "image_path": img_path,
                        "prompt_key": prompt_key,
                        "prompt_text": prompt_text,
                        "params": {
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": top_k
                        },
                        "content": result["content"],
                        "generation_time": generation_time,
                        "timestamp": time.time(),
                        "formatted_time": format_timestamp(time.time())
                    }
                else:
                    logging.warning(f"生成结果时出错 [提示词: {prompt_key}]: {result.get('error', 'Unknown error')}")
                    return None
                
            except Exception as e:
                logging.error(f"为提示词 {prompt_key} 生成结果时出错: {str(e)}")
                return None
    
    async def _grade_candidates(self, session, grade_semaphore, img_path: str, image_base64: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """一次性评分所有候选结果"""
        if not candidates:
            return []
            
        async with grade_semaphore:
            try:
                # 构建评分提示
                grading_template = self.grading_prompt["template"]
                
                # 调用评分API，同时发送图像和所有候选
                start_time = time.time()
                grading_result = await self.grading_client.grade_multiple_candidates(
                    session, 
                    grading_template, 
                    candidates,
                    image_base64
                )
                grading_time = time.time() - start_time
                
                # 更新统计
                self.stats['total_grading_time'] += grading_time
                
                # 处理评分结果
                if "error" not in grading_result:
                    # 为每个候选更新评分
                    for i, candidate in enumerate(candidates):
                        if i < len(grading_result["scores"]):
                            # 添加评分信息到候选结果
                            candidate["score"] = grading_result["scores"][i]
                            candidate["is_best"] = (i == grading_result["best_index"])
                            candidate["grading_time"] = grading_time
                    
                    # 记录整体评分解释
                    candidates[0]["score_explanation"] = grading_result["content"]
                    
                    return candidates
                else:
                    logging.warning(f"评分时出错: {grading_result.get('error', 'Unknown error')}")
                    return []
                
            except Exception as e:
                logging.error(f"评分候选结果时出错: {str(e)}", exc_info=True)
                return []

    
    async def _grade_single_candidate(self, session, semaphore, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """评分单个候选结果"""
        async with semaphore:
            try:
                # 构建评分提示
                grading_template = self.grading_prompt["template"]
                grading_prompt = grading_template.format(
                    generated_answer=candidate["content"]
                )
                
                # 调用评分API
                start_time = time.time()
                grading_result = await self.grading_client.grade(session, grading_prompt)
                grading_time = time.time() - start_time
                
                # 更新统计
                self.stats['total_grading_time'] += grading_time
                
                # 处理评分结果
                if "error" not in grading_result:
                    # 添加评分信息到候选结果
                    candidate["score"] = grading_result["score"]
                    candidate["score_explanation"] = grading_result["content"]
                    candidate["grading_time"] = grading_time
                    return candidate
                else:
                    logging.warning(f"评分时出错: {grading_result.get('error', 'Unknown error')}")
                    return None
            
            except Exception as e:
                logging.error(f"评分候选结果时出错: {str(e)}")
                return None
    
    async def _save_result(self, img_path: str, best_result: Dict[str, Any], all_candidates: List[Dict[str, Any]]):
        """保存处理结果"""
        # 确定保存位置
        if best_result["score"] >= self.config.score_threshold:
            result_dir = os.path.join(self.config.output_dir, "high_quality")
        else:
            result_dir = os.path.join(self.config.output_dir, "low_quality")
        
        # 创建安全的文件名
        img_name = os.path.basename(img_path)
        safe_img_name = os.path.splitext(img_name)[0]
        safe_img_name = "".join(c if c.isalnum() else "_" for c in safe_img_name)
        
        # 构建完整结果
        full_result = {
            "image_info": {
                "path": img_path,
                "filename": img_name
            },
            "best_result": best_result,
            "all_candidates": all_candidates,
            "timestamp": time.time(),
            "formatted_time": format_timestamp(time.time())
        }
        
        # 保存结果
        async with self.file_lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(result_dir, f"{safe_img_name}_{timestamp}.json")
            save_json(result_file, full_result)
    
    def _save_final_stats(self):
        """保存最终统计信息"""
        # 计算耗时
        total_time = time.time() - self.stats["start_time"]
        
        # 构建统计摘要
        summary = {
            "stats": {
                "total_samples": self.stats["total_samples"],
                "processed_samples": self.stats["processed_samples"],
                "high_quality_count": self.stats["high_quality_count"],
                "low_quality_count": self.stats["low_quality_count"],
                "total_generation_time": self.stats["total_generation_time"],
                "total_grading_time": self.stats["total_grading_time"],
                "total_time": total_time,
                "average_generation_time": self.stats["total_generation_time"] / max(1, self.stats["processed_samples"]),
                "average_grading_time": self.stats["total_grading_time"] / max(1, self.stats["processed_samples"]),
                "samples_per_second": self.stats["processed_samples"] / max(1, total_time),
                "high_quality_percentage": (self.stats["high_quality_count"] / max(1, self.stats["processed_samples"])) * 100,
            },
            "config": {
                "prompt_keys": list(self.prompts.keys()),
                "score_threshold": self.config.score_threshold,
                "temperature_range": self.config.temperature_range,
                "top_p_range": self.config.top_p_range,
                "top_k_range": self.config.top_k_range,
            },
            "timestamp": time.time(),
            "formatted_time": format_timestamp(time.time())
        }
        
        # 保存摘要
        summary_file = os.path.join(self.config.output_dir, "summary.json")
        save_json(summary_file, summary)
        logging.info(f"统计摘要已保存到 {summary_file}")
