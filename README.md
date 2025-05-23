# Rejection-Sampling-Evaluator

一个用于多模态模型输出优化的拒绝采样评估工具。该工具通过生成多个候选结果并使用评分模型选择最佳结果，提高生成质量。

---

## 项目简介

**Rejection-Sampling-Evaluator** 使用变温度采样策略为每张图像生成多个候选回答，然后通过调用评分模型评估每个候选的质量，选择最佳结果。

工具支持多种参数组合（温度、top-p、top-k）自动构建最优推理策略，并将高质量结果（得分高于阈值）与低质量结果分别保存，便于后续分析。

---

## 特点

- ✅ **参数变化采样**：自动尝试不同的温度、top-p 和 top-k 组合  
- ✅ **多提示词策略**：支持使用多个提示词生成多样化结果  
- ✅ **异步并行处理**：高效处理大量图像样本  
- ✅ **自动质量评估**：使用强大的多模态模型评分  
- ✅ **断点续跑**：支持中断后从检查点恢复  
- ✅ **详细统计分析**：提供完整的评估报告和性能分析  

---

## 运行流程

### 1. 启动 VLLM 服务
需先部署多模态模型服务，推荐使用 VLLM 框架加速推理：
```bash
vllm serve /path/to/your/model/checkpoint \
   --served-model-name Qwen2.5-VL-32B-Custom \
   --tensor-parallel-size 8 \  # GPU数量
   --dtype bfloat16 \          # 精度设置
   --port 36000                # API端口
```
详细参数可参考https://blog.csdn.net/weixin_45921929/article/details/147927467?spm=1001.2014.3001.5501


## 运行流程

### 2. 准备提示词文件

#### 生成提示词 (`prompts/generation_prompts.json`)
用于控制网表生成的多模态模型输出格式，建议设计多样化提示策略：
```json
{
  "prompt1": "请分析这张系统框图，识别所有组件及其连接关系，以网表形式输出。",
  "prompt2": "详细描述这张系统框图中的所有元件和连接路径。",
  "prompt3": "这张图显示了一个系统框图，请以网表形式给出完整的组件列表和连接关系。",
  "prompt4": "分析此系统框图，提取并描述所有模块及其之间的连接。"
}
```


#### 评分提示词 (`prompts/grading_prompt.json`)

```json
{
  "template": "You are an expert system diagram evaluator. Please assess the quality of the following netlist description generated by an AI model by comparing it to the diagram shown in the image.\n\nGenerated Answer:\n{generated_answer}\n\nPlease carefully examine the system diagram in the image and evaluate how accurately the description captures the components and their connections shown in the diagram.\n\nProvide a score between 0 and 100, where higher scores indicate better accuracy and completeness. Give a brief explanation for your score, focusing on whether the description correctly identifies all the key components and accurately represents how they are connected in the diagram."
}
```


## 3. 运行评估程序

### 基本用法

```bash
python main.py \
  --image-dir datasets/system_diagrams \
  --output-dir results/system_diagram_analysis \
  --vllm-api [http://0.0.0.0:36000/v1](http://0.0.0.0:36000/v1) \
  --vllm-model Qwen2.5-VL-32B-Custom \
  --grading-api [https://api.openai-proxy.org/v1](https://api.openai-proxy.org/v1) \
  --grading-key your-api-key \
  --grading-model o4-mini
```


### 完整参数示例
```bash
python main.py \
  --image-dir datasets/1-stage \
  --output-dir results/circuit_analysis_1_stage \
  --prompts prompts/generation_prompts.json \
  --grading-prompt prompts/grading_prompt.json \
  --prompt-keys prompt1 prompt3 prompt5 prompt6 \
  --vllm-api [http://0.0.0.0:36000/v1](http://0.0.0.0:36000/v1) \
  --vllm-model Qwen2.5-VL-32B-Custom \
  --grading-api [https://api.openai-proxy.org/v1](https://api.openai-proxy.org/v1) \
  --grading-key your-api-key \
  --grading-model o4-mini \
  --temperature-range 0.3,0.5,0.7,0.9,1.2 \
  --top-p-range 0.5,0.7,0.9 \
  --top-k-range 20,40,60,80 \
  --score-threshold 80 \
  --gen-workers 16 \
  --grade-workers 10 \
  --samples -1 \
  --checkpoint-interval 5 \
  --log-level INFO
```

### 断点续跑
```bash
python main.py \
  --image-dir datasets/1-stage \
  --output-dir results/circuit_analysis_1_stage \
  --vllm-api [http://0.0.0.0:36000/v1](http://0.0.0.0:36000/v1) \
  --vllm-model Qwen2.5-VL-32B-Custom \
  --grading-api [https://api.openai-proxy.org/v1](https://api.openai-proxy.org/v1) \
  --grading-key your-api-key \
  --grading-model o4-mini \
  --resume
```

### 4. 查看结果

程序运行完成后，结果将保存在指定的输出目录中：

- `high_quality/`: 包含高质量结果（得分>=阈值）
- `low_quality/`: 包含低质量结果（得分&lt;阈值）
- `summary.json`: 包含评估统计信息和性能分析
- `logs/`: 包含详细的运行日志


## 常见问题
- Q: 如何处理评分API超时问题？
- A: 调整 --grade-workers 参数减少并发请求数，或使用更可靠的API服务。

- Q: 评分结果全为0分怎么办？
- A: 检查评分提示词模板是否正确，并确保评分API能正确处理图像和文本。

- Q: 为什么生成过程很慢？
- A: 调整 --gen-workers 参数提高并发数，但注意不要超出GPU内存限制。