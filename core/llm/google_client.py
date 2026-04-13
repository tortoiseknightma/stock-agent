"""
Google Gemini LLM Client
=========================
Supports: Gemini models via Google Generative AI API.
"""

import os
import time
from typing import List, Optional

from .base import BaseLLMClient, LLMMessage, LLMResponse


class GoogleClient(BaseLLMClient):
    """
    LLM client for Google Gemini models.

    Usage:
        client = GoogleClient("gemini-2.0-flash")
        response = client.chat([
            LLMMessage("system", "You are a financial analyst."),
            LLMMessage("user", "Analyze AAPL"),
        ])
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def chat(self, messages: List[LLMMessage],
             temperature: float = 0.0,
             max_tokens: int = 4096) -> LLMResponse:
        """Send chat completion via Google Generative AI SDK."""
        import google.generativeai as genai

        api_key = self.kwargs.get("api_key") or os.environ.get("GOOGLE_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)

        # Build conversation history
        system_prompt = ""
        history = []
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            elif m.role == "user":
                history.append({"role": "user", "parts": [m.content]})
            elif m.role == "assistant":
                history.append({"role": "model", "parts": [m.content]})

        start = time.time()

        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt if system_prompt else None,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        # If history has more than 1 message, use chat; otherwise single generate
        if len(history) > 1:
            chat = model.start_chat(history=history[:-1])
            response = chat.send_message(history[-1]["parts"][0])
        else:
            response = model.generate_content(history[0]["parts"][0] if history else "")

        latency = (time.time() - start) * 1000

        content = response.text if hasattr(response, "text") else str(response)

        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
            }

        return LLMResponse(
            content=content,
            model=self.model,
            provider="google",
            usage=usage,
            latency_ms=round(latency, 1),
            raw=response,
        )
