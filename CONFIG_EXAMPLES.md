# Configuration Examples for Different LLM Providers

This file shows configuration examples for using different LLM providers with the HotPotQA speculative execution framework.

## 1. Using vLLM with Local Open-Source Models

**Best for**: Cost-effective local inference, full control over models

```python
# In src/constants.py

# vLLM configuration
vllm_api_url = "http://localhost:8000/v1"  # Change if running on different server
vllm_api_key = "dummy-key"

# Model names (use "vllm:" prefix in command line or here)
vllm_model_name = "meta-llama/Llama-2-7b-chat-hf"
```

**Start vLLM server**:
```bash
python start_vllm_server.py --model meta-llama/Llama-2-7b-chat-hf
```

**Run experiment**:
```bash
python run.py \
  --modelname "vllm:meta-llama/Llama-2-7b-chat-hf" \
  --guessmodelname "vllm:mistralai/Mistral-7B-Instruct-v0.1"
```

## 2. Using OpenAI API

**Best for**: Production use, access to latest models (GPT-4, GPT-4-turbo)

```python
# In src/constants.py

openai_api_key = "your-openai-api-key"  # Get from https://platform.openai.com

# Model configurations
```

**Run experiment**:
```bash
python run.py \
  --modelname "gpt-4" \
  --guessmodelname "gpt-3.5-turbo"
```

## 3. Using Google Gemini API

**Best for**: Low-cost alternatives to OpenAI

```python
# In src/constants.py

gemini_api_key = "your-gemini-api-key"  # Get from https://ai.google.dev/
```

**Run experiment**:
```bash
python run.py \
  --modelname "gemini-2.0-flash" \
  --guessmodelname "gemini-2.0-flash"
```

## 4. Using OpenRouter (Multi-Provider)

**Best for**: Access to many models through a single API

```python
# In src/constants.py

openrouter_api_key = "your-openrouter-api-key"  # Get from https://openrouter.ai/
openrouter_model_name = "openai/gpt-4"          # Agent model
openrouter_guess_model_name = "openai/gpt-3.5-turbo"  # Speculator model
```

**Run experiment**:
```bash
python run.py \
  --modelname "openai/gpt-4" \
  --guessmodelname "openai/gpt-3.5-turbo"
```

## 5. Hybrid: Local Agent + API Speculator

**Best for**: Balance between cost and quality

```bash
# Use local Llama 2 as agent, fast API model as speculator
python run.py \
  --modelname "vllm:meta-llama/Llama-2-13b-chat-hf" \
  --guessmodelname "openai/gpt-3.5-turbo"
```

## 6. Hybrid: API Agent + Local Speculator

**Best for**: High-quality reasoning with low-cost prediction

```bash
# Use powerful GPT-4 as agent, local Mistral as fast speculator
python run.py \
  --modelname "openai/gpt-4" \
  --guessmodelname "vllm:mistralai/Mistral-7B-Instruct-v0.1"
```

## Recommended Model Combinations

### For Research (Best Quality)
```bash
python run.py \
  --modelname "openai/gpt-4" \
  --guessmodelname "openai/gpt-3.5-turbo"
```

### For Budget-Conscious (Local Only)
```bash
# Start vLLM with 7B models
python start_vllm_server.py --model meta-llama/Llama-2-7b-chat-hf --port 8000

# Run with two local models
python run.py \
  --modelname "vllm:meta-llama/Llama-2-7b-chat-hf" \
  --guessmodelname "vllm:mistralai/Mistral-7B-Instruct-v0.1"
```

### For Best Performance (Local Only)
```bash
# Requires 2+ GPUs or high VRAM (80GB+)
python start_vllm_server.py \
  --model meta-llama/Llama-2-13b-chat-hf \
  --gpu-util 0.9

python run.py \
  --modelname "vllm:meta-llama/Llama-2-13b-chat-hf" \
  --guessmodelname "vllm:Qwen/Qwen2-7B-Instruct"
```

## Model Selection Guide

### Agent Model (Primary Model)
Choose based on task complexity and budget:

| Model | Quality | Speed | Cost | Notes |
|-------|---------|-------|------|-------|
| gpt-4 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 💰💰💰 | Best quality |
| gpt-3.5-turbo | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 💰 | Good balance |
| gemini-2.0-flash | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 💰 | Multi-modal |
| Llama 2 13B | ⭐⭐⭐ | ⭐⭐⭐ | Free | Local, good for HQA |
| Mistral 7B | ⭐⭐⭐ | ⭐⭐⭐⭐ | Free | Local, very fast |

### Speculator Model (Fast Predictor)
Choose a faster/cheaper model:

| Model | Speed | Cost | Best For |
|-------|-------|------|----------|
| gpt-3.5-turbo | ⭐⭐⭐⭐ | 💰 | General purpose |
| gpt-4-mini | ⭐⭐⭐ | 💰 | Accuracy |
| Mistral 7B | ⭐⭐⭐⭐⭐ | Free | Speed |
| Qwen 7B | ⭐⭐⭐⭐⭐ | Free | Speed + Chinese |

## Performance Optimization Tips

### For vLLM (Local Models)

1. **Use quantization** (reduces memory, slight quality loss):
   ```bash
   python start_vllm_server.py --model meta-llama/Llama-2-7b-chat-hf --quantization awq
   ```

2. **Use smaller models** when possible:
   - 7B models are much faster than 13B
   - 3-5B models work well as speculators

3. **Monitor GPU memory**:
   ```bash
   nvidia-smi watch -n 1
   ```

4. **Adjust batch size** in constants.py:
   ```python
   max_output_tokens = 500  # Reduce if running out of memory
   ```

### For API Models

1. **Use faster models for speculator**:
   ```bash
   --guessmodelname "openai/gpt-3.5-turbo"
   ```

2. **Reduce token limits** if you don't need long outputs:
   ```python
   max_output_tokens = 500  # Default is 1000
   ```

## Troubleshooting

### vLLM Connection Error
```bash
# Check if server is running
curl http://localhost:8000/v1/models
```

### Out of Memory Error
- Use smaller model (7B instead of 13B)
- Enable quantization
- Reduce batch size
- Use fewer tokens

### API Rate Limits
- Reduce `n_samples_to_run` in constants.py
- Add delay between requests
- Use OpenRouter which supports multiple providers

## Additional Resources

- [vLLM Docs](https://docs.vllm.ai/)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Google Gemini API](https://ai.google.dev/)
