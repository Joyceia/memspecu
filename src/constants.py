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
n_samples_to_run = 30
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
wiki_fallback_to_guess = False
wiki_fallback_notice = False

# Memory mechanism configuration
memory_enabled = True
memory_store_path = "./data/memory_store.json"
memory_insights_path = "./data/memory_insights.json"
memory_max_entries = 5000
memory_top_k_success = 1
memory_top_k_failure = 1
memory_entity_overlap_weight = 0.5
memory_step_match_weight = 0.2
memory_prev_action_match_weight = 0.15
memory_action_type_match_weight = 0.15

# Insight extraction configuration
insight_extraction_min_new = 15
insight_match_threshold = 0.4
insight_max_in_prompt = 3
insight_min_support = 3
