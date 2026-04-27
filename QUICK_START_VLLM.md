# vLLM快速入门指南

## 5分钟快速开始

### 第1步: 安装依赖（1分钟）

```bash
# 更新pip
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt

# 安装vLLM (包含在requirements.txt中)
# 或单独安装
pip install vllm
```

### 第2步: 下载模型（3-10分钟）

选择一个模型并下载:

```bash
# 选项A: Llama 2 7B (推荐，平衡质量和速度)
pip install huggingface-hub
huggingface-cli download meta-llama/Llama-2-7b-chat-hf

# 选项B: Mistral 7B (更快)
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.1

# 选项C: 使用Qwen 2 (支持中文)
hf download Qwen/Qwen3-1.7B
hf download Qwen/Qwen3-32B

```

> **提示**: 如果需要下载需要认证的模型（如Llama 2），先运行 `huggingface-cli login`

### 第3步: 启动vLLM服务器（1分钟）

在一个终端中运行:

```bash
# 启动vLLM服务器
python start_vllm_server.py --model /model/Qwen3-32B --port 8000
```

你应该看到类似的输出:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
...
```

### 第4步: 在另一个终端运行实验

```bash
# 运行实验（20个样本）
python run.py \
  --modelname "vllm:meta-llama/Llama-2-7b-chat-hf" \
  --guessmodelname "vllm:mistralai/Mistral-7B-Instruct-v0.1"

  python run.py \
  --modelname "vllm:/model/Qwen3-32B" \
  --guessmodelname "vllm:/model/Qwen3-1.7B"
```

完成！🎉

## 常用命令速查

### 启动服务器

```bash
# 基础启动
python start_vllm_server.py

# 指定模型
python start_vllm_server.py --model mistralai/Mistral-7B-Instruct-v0.1

# 启用量化（节省显存）
python start_vllm_server.py --model meta-llama/Llama-2-7b-chat-hf --quantization awq

# 高性能配置
python start_vllm_server.py --model meta-llama/Llama-2-7b-chat-hf --gpu-util 0.95

python start_vllm_server.py --model /models/Qwen3-32B --tensor-parallel-size 3 --guessmodel /models/Qwen3-1.7B --port 8000
```

### 运行实验

```bash
# 基础运行
python run.py

# 指定模型
python run.py \
  --modelname "vllm:/models/Qwen3-14B" \
  --guessmodelname "vllm:/models/Qwen3-1.7B"

# 混合使用（本地+API）
python run.py \
  --modelname "vllm:meta-llama/Llama-2-13b-chat-hf" \
  --guessmodelname "openai/gpt-3.5-turbo"

# 更改样本数量
# 编辑 src/constants.py 中的 n_samples_to_run
```

### 计算和可视化

```bash
# 计算指标
python run.py --norun --getmetric --savemetrics

# 生成图表
python run.py --norun --graph
```

## 常见问题

### Q: 需要GPU吗?
**A**: GPU可选，但强烈推荐。没有GPU会很慢。

### Q: 最少需要多少显存?
**A**: 
- 7B模型: 8GB (可接受)
- 7B + 量化: 4GB (推荐)
- 13B模型: 24GB (推荐)

### Q: 如何节省显存?
**A**: 
1. 使用更小的模型 (7B而不是13B)
2. 启用量化: `--quantization awq`
3. 降低GPU利用率: `--gpu-util 0.7`

### Q: 模型速度如何?
**A**: 大约每个查询 0.5-2 秒，取决于：
- GPU类型和性能
- 模型大小
- 是否启用量化
- 输入/输出长度

### Q: 我想用另一个模型怎么办?
**A**: 
1. 下载模型: `huggingface-cli download <model-id>`
2. 启动服务器: `python start_vllm_server.py --model <model-id>`
3. 运行实验: `python run.py --modelname "vllm:<model-id>"`

## 下一步

- 详细文档: 查看 [VLLM_SETUP.md](VLLM_SETUP.md)
- 配置示例: 查看 [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md)
- 项目信息: 查看 [README.md](README.md)

## 需要帮助?

- 遇到错误? 查看 [VLLM_SETUP.md](VLLM_SETUP.md#故障排查)
- 想要调优性能? 查看 [VLLM_SETUP.md](VLLM_SETUP.md#性能优化)
- 想要了解更多? 查看 [VLLM_INTEGRATION_SUMMARY.md](VLLM_INTEGRATION_SUMMARY.md)

祝你使用愉快！ 🚀
