openrouter_api_key = "your-openrouter-api-key"
openrouter_model_name = "openai/gpt-4"
openrouter_guess_model_name = "openai/gpt-5-nano"

# OpenAI API configuration
openai_api_key = "your-openai-api-key"

# Gemini API configuration
gemini_api_key = "your-gemini-api-key"

# vLLM configuration for local models
vllm_model_name = None  # e.g., "meta-llama/Llama-2-7b-chat-hf" or "mistralai/Mistral-7B-Instruct-v0.1"
vllm_api_url = "http://localhost:8000/v1"  # vLLM OpenAI-compatible API endpoint
vllm_guess_api_url = "http://localhost:8001/v1"  # Optional second vLLM endpoint for guess model, e.g., "http://localhost:8001/v1"
vllm_api_key = "dummy-key"  # vLLM doesn't require a real key for local servers

prompts_folder = "./prompts/"
prompt_file = "prompts_naive.json"
agent_role = "Question Answering Agent"

random_seed = 248
num = 7405
n_steps_to_run = 8
n_samples_to_run = 20
client_error_sleep_time = 60
server_error_sleep_time = 60
trajectory_filenames = ["log.txt", "normalobs.json", "simobs.json", "metrics.json"]
guess_num_actions = 3
max_agent_retries = 1
max_guess_retries = 3

# Agent LLM settings
max_output_tokens = 1000
top_p = 1
temperature = 1

# Guess LLM settings
max_guess_output_tokens = 100
guess_top_p = 0.9
guess_temperature = 0.1

# Wikipedia request settings
wiki_request_timeout = 15
wiki_max_retries = 2
wiki_fallback_to_guess = True
wiki_fallback_notice = True
