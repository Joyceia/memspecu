import openai
from google import genai
from google.genai import types

from . import constants


class LLMClient:
    """Unified LLM client supporting Gemini, OpenAI, OpenRouter, and vLLM APIs."""

    def __init__(self, model_name, temperature, max_tokens, top_p):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.gemini_client = genai.Client(api_key=constants.gemini_api_key)
        self.openai_client = openai.OpenAI(api_key=constants.openai_api_key)
        self.openrouter_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=constants.openrouter_api_key,
        )
        # vLLM client (OpenAI-compatible)
        if hasattr(constants, 'vllm_api_url') and constants.vllm_api_url:
            self.vllm_client = openai.OpenAI(
                base_url=constants.vllm_api_url,
                api_key=constants.vllm_api_key or "dummy-key",
            )
        else:
            self.vllm_client = None

    def call(self, prompt, stop=None):
        if self.model_name.startswith("gemini"):
            return self._gemini_call(prompt, stop)
        elif self.model_name.startswith("gpt"):
            return self._openai_call(prompt, stop)
        elif self.model_name.startswith("vllm"):
            return self._vllm_call(prompt, stop)
        else:
            return self._openrouter_call(prompt, stop)

    def _gemini_call(self, prompt, stop):
        if stop is not None:
            config = types.GenerateContentConfig(stop_sequences=stop)
        else:
            config = types.GenerateContentConfig()
        response = self.gemini_client.models.generate_content(
            model=self.model_name, contents=prompt, config=config
        )
        return str(response.text)

    def _openai_call(self, prompt, stop):
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content

    def _openrouter_call(self, prompt, stop):
        response = self.openrouter_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content

    def _vllm_call(self, prompt, stop):
        """Call vLLM via OpenAI-compatible API."""
        if not self.vllm_client:
            raise RuntimeError("vLLM client not initialized. Check constants.vllm_api_url")
        
        # Extract the actual model name (remove 'vllm:' prefix if present)
        model_name = self.model_name.replace("vllm:", "") if self.model_name.startswith("vllm:") else self.model_name
        
        response = self.vllm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
        )
        return response.choices[0].message.content
