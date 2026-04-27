# vLLM Integration Summary

本文档总结了为项目添加vLLM支持所做的更改。

## 更改列表

### 1. 核心修改

#### `src/llm_client.py`
- **新增vLLM客户端支持**: 添加了vLLM作为新的LLM提供商
- **修改点**:
  - 更新类文档字符串以包含vLLM
  - 在`__init__`中添加vLLM客户端初始化（使用OpenAI兼容API）
  - 修改`call()`方法以路由vLLM请求
  - 新增`_vllm_call()`方法用于vLLM API调用

```python
# 示例使用
def call(self, prompt, stop=None):
    if self.model_name.startswith("vllm"):
        return self._vllm_call(prompt, stop)
```

#### `src/constants.py`
- **新增配置参数**:
  - `openai_api_key`: OpenAI API密钥
  - `gemini_api_key`: Gemini API密钥  
  - `vllm_api_url`: vLLM服务器URL (默认: `http://localhost:8000/v1`)
  - `vllm_api_key`: vLLM API密钥 (本地不需要真实密钥)
  - `vllm_model_name`: vLLM模型名称

#### `requirements.txt`
- 添加了`vllm`作为新依赖

### 2. 新增文件

#### `VLLM_SETUP.md` (详细设置指南)
完整的vLLM配置和使用教程，包含:
- vLLM简介和优点
- 硬件/软件要求
- 安装步骤
- 模型下载指南
- 启动vLLM服务器的方式
- 运行实验的命令示例
- 故障排查指南
- 性能优化建议
- Docker使用方式

#### `start_vllm_server.py` (便利启动脚本)
Python脚本用于便捷启动vLLM服务器，支持:
- 选择模型 (`--model`)
- 配置端口 (`--port`)
- GPU内存利用率 (`--gpu-util`)
- 量化方法 (`--quantization`: awq, gptq, bitsandbytes)
- 张量并行 (`--tensor-parallel-size`)
- 数据类型配置 (`--dtype`)
- 其他高级选项

**使用示例**:
```bash
python start_vllm_server.py --model meta-llama/Llama-2-7b-chat-hf --gpu-util 0.9
python start_vllm_server.py --model mistralai/Mistral-7B-Instruct-v0.1 --quantization awq
```

#### `CONFIG_EXAMPLES.md` (配置示例)
展示不同提供商的配置示例:
- vLLM本地模型配置
- OpenAI API配置
- Google Gemini配置
- OpenRouter配置
- 混合配置（本地+API）
- 推荐的模型组合
- 模型选择指南
- 性能优化技巧

### 3. 文档更新

#### `README.md`
- 更新了项目布局，添加了vLLM相关文件
- 扩展了"Getting Started"部分，添加了vLLM选项
- 提供了vLLM服务器启动示例
- 添加了vLLM参考链接

## 使用方式

### 基础用法

#### 启动vLLM服务器
```bash
# 使用默认配置（Llama 2 7B）
python start_vllm_server.py

# 使用Mistral模型
python start_vllm_server.py --model mistralai/Mistral-7B-Instruct-v0.1

# 使用量化模型（节省显存）
python start_vllm_server.py --model meta-llama/Llama-2-7b-chat-hf --quantization awq
```

#### 运行实验

**使用vLLM作为代理和推测模型**:
```bash
python run.py \
  --modelname "vllm:meta-llama/Llama-2-7b-chat-hf" \
  --guessmodelname "vllm:mistralai/Mistral-7B-Instruct-v0.1"
```

**混合使用（本地+API）**:
```bash
# 本地agent + API speculator
python run.py \
  --modelname "vllm:meta-llama/Llama-2-13b-chat-hf" \
  --guessmodelname "openai/gpt-3.5-turbo"

# API agent + 本地speculator
python run.py \
  --modelname "openai/gpt-4" \
  --guessmodelname "vllm:mistralai/Mistral-7B-Instruct-v0.1"
```

## 主要优点

### 1. **成本节省**
- 使用开源模型无需付费API调用
- 尤其适合大规模实验

### 2. **本地控制**
- 完全本地化推理，无需上传数据到云
- 可以微调模型以获得更好的性能

### 3. **灵活性**
- 支持多种开源模型
- 可以混合使用API和本地模型
- 轻松切换模型进行对比

### 4. **性能优化**
- vLLM的高效推理引擎
- 支持量化以降低内存需求
- 支持多GPU张量并行

## 推荐模型

### Agent模型（主要推理模型）
- **Llama 2 7B**: 综合能力最强的7B开源模型
- **Llama 2 13B**: 质量更高但需要更多显存
- **Mistral 7B**: 速度快，显存效率高
- **Qwen 7B**: 中文能力强

### Speculator模型（快速预测模型）
- **Mistral 7B**: 最快的开源模型之一
- **Qwen 7B**: 快速且支持中文
- **GPT-3.5-turbo**: 如果需要质量+速度平衡

## 技术细节

### 模型前缀约定
- `vllm:` - 使用vLLM本地模型
- `gpt` - 使用OpenAI API
- `gemini` - 使用Google Gemini API
- 其他 - 使用OpenRouter API

### API兼容性
vLLM提供OpenAI兼容的API接口，因此可以使用相同的OpenAI客户端与之通信，降低集成复杂度。

### 模型名称处理
```python
# 自动去除vllm:前缀
model_name = "vllm:meta-llama/Llama-2-7b-chat-hf"
# 实际传给vLLM的是: "meta-llama/Llama-2-7b-chat-hf"
```

## 后续可能的增强

1. **支持更多推理框架** (Ollama, TextGeneration WebUI等)
2. **模型管理工具** (自动下载/更新模型)
3. **性能监控面板** (实时显示GPU使用率、吞吐量等)
4. **分布式推理** (多机器部署)
5. **模型微调支持** (使用自己的数据微调模型)

## 相关资源

- [vLLM官方文档](https://docs.vllm.ai/)
- [Hugging Face模型库](https://huggingface.co/models)
- [Meta Llama](https://www.meta.com/research/llama/)
- [Mistral AI](https://www.mistral.ai/)
