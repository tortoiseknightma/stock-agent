"""Tests for core/llm — factory, base class contract, provider routing."""

import pytest
from unittest.mock import MagicMock, patch
from core.llm.base import BaseLLMClient, LLMMessage, LLMResponse
from core.llm.factory import create_llm_client, _OPENAI_COMPATIBLE


# ---------------------------------------------------------------------------
# LLMMessage / LLMResponse dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_llm_message_fields(self):
        msg = LLMMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_llm_response_defaults(self):
        resp = LLMResponse(content="OK")
        assert resp.content == "OK"
        assert resp.model == ""
        assert resp.provider == ""
        assert resp.usage == {}
        assert resp.latency_ms == 0

    def test_llm_response_full(self):
        resp = LLMResponse(
            content="Analysis done",
            model="gpt-4o",
            provider="openai",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            latency_ms=423.5,
        )
        assert resp.usage["total_tokens"] == 150
        assert resp.latency_ms == pytest.approx(423.5)


# ---------------------------------------------------------------------------
# BaseLLMClient contract
# ---------------------------------------------------------------------------

class ConcreteClient(BaseLLMClient):
    """Minimal concrete implementation for contract testing."""
    def chat(self, messages, temperature=0.0, max_tokens=4096) -> LLMResponse:
        return LLMResponse(content="test response", model=self.model,
                           provider="test")


class TestBaseLLMClient:
    def test_instantiation(self):
        client = ConcreteClient(model="test-model")
        assert client.model == "test-model"

    def test_chat_simple_calls_chat_and_returns_content(self):
        client = ConcreteClient(model="test-model")
        result = client.chat_simple(system="You are a helper.", user="Hello")
        assert result == "test response"

    def test_chat_simple_passes_both_messages(self):
        calls = []

        class SpyClient(BaseLLMClient):
            def chat(self, messages, temperature=0.0, max_tokens=4096):
                calls.append(messages)
                return LLMResponse(content="ok")

        client = SpyClient(model="m")
        client.chat_simple("sys prompt", "user msg")
        assert len(calls) == 1
        roles = [m.role for m in calls[0]]
        assert "system" in roles
        assert "user" in roles

    def test_provider_name_derived_from_class_name(self):
        client = ConcreteClient(model="m")
        assert client.provider_name == "concrete"

    def test_base_url_stored(self):
        client = ConcreteClient(model="m", base_url="http://localhost:8000/v1")
        assert client.base_url == "http://localhost:8000/v1"

    def test_abstract_class_not_instantiable(self):
        with pytest.raises(TypeError):
            BaseLLMClient(model="m")


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------

class TestFactory:
    @pytest.mark.parametrize("provider", list(_OPENAI_COMPATIBLE))
    def test_openai_compatible_providers_return_client(self, provider):
        """create_llm_client should return an OpenAICompatibleClient for any known provider."""
        import sys
        # Stub the openai module so the import inside _get_client() doesn't fail
        fake_openai = MagicMock()
        fake_openai.OpenAI = MagicMock()
        with patch.dict(sys.modules, {"openai": fake_openai}):
            client = create_llm_client(provider, "any-model")
        assert client is not None
        assert hasattr(client, "chat")
        assert hasattr(client, "chat_simple")

    def test_anthropic_provider_returns_client(self):
        from core.llm.anthropic_client import AnthropicClient
        client = create_llm_client("anthropic", "claude-opus-4-6")
        assert isinstance(client, AnthropicClient)

    def test_google_provider_returns_client(self):
        from core.llm.google_client import GoogleClient
        client = create_llm_client("google", "gemini-2.0-flash")
        assert isinstance(client, GoogleClient)

    def test_unknown_provider_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_llm_client("nonexistent_provider", "model")

    def test_case_insensitive_provider(self):
        """Provider name should be case-insensitive."""
        import sys
        fake_openai = MagicMock()
        fake_openai.OpenAI = MagicMock()
        with patch.dict(sys.modules, {"openai": fake_openai}):
            client = create_llm_client("OpenAI", "gpt-4o")
        assert client is not None

    def test_base_url_forwarded_to_client(self):
        import sys
        fake_openai = MagicMock()
        fake_openai.OpenAI = MagicMock()
        with patch.dict(sys.modules, {"openai": fake_openai}):
            client = create_llm_client("openai", "gpt-4o",
                                       base_url="http://localhost:8000/v1")
        assert client.base_url == "http://localhost:8000/v1"


# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------

class TestOpenAICompatibleClient:
    def test_chat_returns_llm_response(self):
        from core.llm.openai_compat import OpenAICompatibleClient

        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from GPT"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.model = "gpt-4o"
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 5
        mock_completion.usage.total_tokens = 15
        mock_openai.chat.completions.create.return_value = mock_completion

        # Inject mock directly (avoids patching lazy import)
        client = OpenAICompatibleClient("gpt-4o", provider="openai")
        client._client = mock_openai
        response = client.chat([LLMMessage("user", "hello")])

        assert response.content == "Hello from GPT"
        assert response.usage["total_tokens"] == 15

    def test_provider_name_set_correctly(self):
        from core.llm.openai_compat import OpenAICompatibleClient
        client = OpenAICompatibleClient("deepseek-chat", provider="deepseek")
        assert client.provider_name == "deepseek"


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

class TestAnthropicClient:
    def test_chat_returns_llm_response(self):
        from core.llm.anthropic_client import AnthropicClient

        mock_anthropic = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.text = "Claude response"
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.model = "claude-opus-4-6"
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 10
        mock_anthropic.messages.create.return_value = mock_response

        client = AnthropicClient("claude-opus-4-6")
        client._client = mock_anthropic

        response = client.chat([
            LLMMessage("system", "You are helpful."),
            LLMMessage("user", "Analyze AAPL"),
        ])

        assert response.content == "Claude response"
        assert response.provider == "anthropic"
        assert response.usage["total_tokens"] == 30

    def test_system_message_extracted(self):
        from core.llm.anthropic_client import AnthropicClient

        mock_anthropic = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.text = "OK"
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.model = "claude-opus-4-6"
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 2
        mock_anthropic.messages.create.return_value = mock_response

        client = AnthropicClient("claude-opus-4-6")
        client._client = mock_anthropic

        client.chat([
            LLMMessage("system", "Be concise."),
            LLMMessage("user", "Hello"),
        ])

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert call_kwargs["system"] == "Be concise."
        # user message should be in messages list, not system
        assert all(m["role"] != "system" for m in call_kwargs["messages"])
