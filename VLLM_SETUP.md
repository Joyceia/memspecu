# vLLM Setup Guide

本文档说明如何配置和使用vLLM来运行本项目中的开源大模型。

## 什么是vLLM?

vLLM是一个高性能的大语言模型推理框架，提供：
- **快速推理**: 优化的推理性能，支持多种开源模型
- **OpenAI兼容API**: 完全兼容OpenAI API，方便集成
- **低成本**: 使用开源模型，无需付费API调用

## 前置要求

### 硬件要求
- GPU推荐（至少8GB VRAM）
- 例如: NVIDIA A100, A10, V100, RTX 4090等
- 可选: 也支持CPU推理（但速度较慢）

### 软件要求
- Python 3.8+
- CUDA 11.8+ (如果使用NVIDIA GPU)

## 安装

### 1. 安装vLLM

```bash
# 最新版本
pip install vllm

# 或指定版本
pip install vllm==0.4.0
```

### 2. 安装模型依赖

根据要使用的模型，可能需要安装额外的依赖：

```bash
# 如果使用Hugging Face模型
pip install transformers

# 如果需要特定模型支持
pip install peft  # for LoRA models
```

### 3. 下载模型

推荐模型（中文和英文能力强）:

```bash
# 使用huggingface-cli下载
huggingface-cli download meta-llama/Llama-2-7b-chat-hf

# 或使用transformers自动下载
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')"
```

推荐的开源模型：
- `meta-llama/Llama-2-7b-chat-hf` - Meta Llama 2 (7B)
- `mistralai/Mistral-7B-Instruct-v0.1` - Mistral 7B
- `NousResearch/Nous-Hermes-2-Mistral-7B-DPO` - Nous Hermes 2
- `meta-llama/Llama-2-13b-chat-hf` - Meta Llama 2 (13B, 需要更多VRAM)
- `Qwen/Qwen2-7B-Instruct` - 阿里 Qwen 2 (支持中文)

## 配置

### 1. 更新 `src/constants.py`

```python
# 本地vLLM模型配置
vllm_api_url = "http://localhost:8000/v1"  # vLLM API端点
vllm_api_key = "dummy-key"  # 本地不需要真实密钥

# 模型名称配置
vllm_model_name = "meta-llama/Llama-2-7b-chat-hf"  # 代理模型
vllm_guess_model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # 推测模型（可选）
```

### 2. （可选）设置Hugging Face Token

如果使用需要认证的模型（如Llama 2）：

```bash
huggingface-cli login
# 输入你的Hugging Face API Token
```

## 启动vLLM服务器

### 方式1: 使用命令行启动

```bash
# 基础启动
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --port 8000

# 使用GPU
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --port 8000 \
  --gpu-memory-utilization 0.9

# 使用多个GPU
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --port 8000 \
  --tensor-parallel-size 2

# 使用量化模型（节省显存）
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --port 8000 \
  --quantization awq  # 或 'gptq', 'bitsandbytes'
```

### 方式2: 使用Python脚本启动

```python
# start_vllm_server.py
from vllm.entrypoints.openai.api_server import run_server
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
args = parser.parse_args()

run_server(
    host="0.0.0.0",
    port=args.port,
    model=args.model,
    gpu_memory_utilization=args.gpu_memory_utilization,
)
```

运行：
```bash
python start_vllm_server.py --model meta-llama/Llama-2-7b-chat-hf --port 8000
```

## 运行实验

### 使用vLLM作为代理模型

```bash
# 使用vLLM的Llama 2模型作为代理，Mistral作为推测模型
python run.py \
  --modelname "vllm:meta-llama/Llama-2-7b-chat-hf" \
  --guessmodelname "vllm:mistralai/Mistral-7B-Instruct-v0.1"
```

### 使用vLLM作为推测模型

```bash
# 使用GPT-4作为代理，vLLM的Mistral作为推测模型
python run.py \
  --modelname "openai/gpt-4" \
  --guessmodelname "vllm:mistralai/Mistral-7B-Instruct-v0.1"
```

### 完全使用本地模型

```bash
# 两个模型都使用本地vLLM
python run.py \
  --modelname "vllm:meta-llama/Llama-2-13b-chat-hf" \
  --guessmodelname "vllm:mistralai/Mistral-7B-Instruct-v0.1"
```

## 故障排查

### 1. 连接拒绝错误

**错误**: `Connection refused to http://localhost:8000/v1`

**解决方案**:
- 确保vLLM服务器已启动
- 检查端口号是否正确
- 确认防火墙未阻止连接

```bash
# 测试连接
curl http://localhost:8000/v1/models
```

### 2. 内存不足（OOM）

**错误**: `CUDA out of memory`

**解决方案**:
- 使用更小的模型（如7B而不是13B）
- 启用量化：`--quantization awq`
- 降低GPU内存利用率：`--gpu-memory-utilization 0.7`
- 使用CPU推理（较慢）

### 3. 模型未找到

**错误**: `Model not found`

**解决方案**:
- 确保已下载模型：`huggingface-cli download <model-id>`
- 检查模型路径是否正确
- 确认Hugging Face Token有效（如果需要）

### 4. vLLM响应缓慢

**解决方案**:
- 增加`--gpu-memory-utilization`（例如0.9）
- 使用更强大的GPU
- 启用批处理优化
- 检查系统资源使用情况

## 性能优化

### 1. 启用量化

```bash
# AWQ量化 (推荐)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --quantization awq

# GPTQ量化
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Mistral-7B-Instruct-v0.1-GPTQ \
  --quantization gptq
```

### 2. 多GPU并行

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-13b-chat-hf \
  --tensor-parallel-size 2  # 使用2块GPU
```

### 3. 调整批处理大小

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --max-model-len 2048 \
  --max-num-batched-tokens 5120
```

## 与API模型混合使用

可以同时使用本地vLLM模型和付费API模型进行比较：

```bash
# 场景1: 本地模型作为快速推测器，API模型作为准确代理
python run.py \
  --modelname "openai/gpt-4" \
  --guessmodelname "vllm:mistralai/Mistral-7B-Instruct-v0.1"

# 场景2: 本地模型作为代理，API模型作为推测器
python run.py \
  --modelname "vllm:meta-llama/Llama-2-13b-chat-hf" \
  --guessmodelname "openai/gpt-3.5-turbo"
```

## 高级配置

### 自定义API URL

如果vLLM运行在远程服务器上，修改 `src/constants.py`：

```python
vllm_api_url = "http://your-server-ip:8000/v1"  # 远程服务器地址
```

### 使用Docker运行vLLM

```bash
# 创建容器
docker run --gpus all \
  -p 8000:8000 \
  --name vllm-server \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-2-7b-chat-hf

# 或保持运行
docker run -d \
  --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-2-7b-chat-hf
```

## 参考资源

- [vLLM官方文档](https://docs.vllm.ai/)
- [Hugging Face模型库](https://huggingface.co/models)
- [vLLM与OpenAI API兼容性](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

