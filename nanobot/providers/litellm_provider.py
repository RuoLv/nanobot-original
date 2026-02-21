"""LiteLLM provider implementation for multi-provider support."""

import asyncio
import json
import json_repair
import os
from typing import Any, Callable

import litellm
from litellm import acompletion
from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.registry import find_by_model, find_gateway


# Standard OpenAI chat-completion message keys; extras (e.g. reasoning_content) are stripped for strict providers.
_ALLOWED_MSG_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name"})


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.
    
    Supports OpenRouter, Anthropic, OpenAI, Gemini, MiniMax, and many other providers through
    a unified interface.  Provider-specific logic is driven by the registry
    (see providers/registry.py) — no if-elif chains needed here.
    """
    
    def __init__(
        self, 
        api_key: str | None = None, 
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
        fallback_model: str | None = None,
        error_callback: Callable | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.fallback_model = fallback_model
        self.extra_headers = extra_headers or {}
        self.error_callback = error_callback
        self.session_key = "unknown"  # For token tracking, set by AgentLoop
        
        # Detect gateway / local deployment.
        # provider_name (from config key) is the primary signal;
        # api_key / api_base are fallback for auto-detection.
        self._gateway = find_gateway(provider_name, api_key, api_base)
        
        # Configure environment variables
        if api_key:
            self._setup_env(api_key, api_base, default_model)
        
        if api_base:
            litellm.api_base = api_base
        
        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True
        # Drop unsupported parameters for providers (e.g., gpt-5 rejects some params)
        litellm.drop_params = True
    
    def _setup_env(self, api_key: str, api_base: str | None, model: str) -> None:
        """Set environment variables based on detected provider."""
        spec = self._gateway or find_by_model(model)
        if not spec:
            return
        if not spec.env_key:
            # OAuth/provider-only specs (for example: openai_codex)
            return

        # Gateway/local overrides existing env; standard provider doesn't
        if self._gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)

        # Resolve env_extras placeholders:
        #   {api_key}  → user's API key
        #   {api_base} → user's api_base, falling back to spec.default_api_base
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key)
            resolved = resolved.replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)
    
    def _resolve_model(self, model: str) -> str:
        """Resolve model name by applying provider/gateway prefixes."""
        if self._gateway:
            # Gateway mode: apply gateway prefix, skip provider-specific prefixes
            prefix = self._gateway.litellm_prefix
            if self._gateway.strip_model_prefix:
                model = model.split("/")[-1]
            if prefix and not model.startswith(f"{prefix}/"):
                model = f"{prefix}/{model}"
            return model
        
        # Standard mode: auto-prefix for known providers
        spec = find_by_model(model)
        if spec and spec.litellm_prefix:
            model = self._canonicalize_explicit_prefix(model, spec.name, spec.litellm_prefix)
            if not any(model.startswith(s) for s in spec.skip_prefixes):
                model = f"{spec.litellm_prefix}/{model}"

        return model

    @staticmethod
    def _canonicalize_explicit_prefix(model: str, spec_name: str, canonical_prefix: str) -> str:
        """Normalize explicit provider prefixes like `github-copilot/...`."""
        if "/" not in model:
            return model
        prefix, remainder = model.split("/", 1)
        if prefix.lower().replace("-", "_") != spec_name:
            return model
        return f"{canonical_prefix}/{remainder}"
    
    def _supports_cache_control(self, model: str) -> bool:
        """Return True when the provider supports cache_control on content blocks."""
        if self._gateway is not None:
            return self._gateway.supports_prompt_caching
        spec = find_by_model(model)
        return spec is not None and spec.supports_prompt_caching

    def _apply_cache_control(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Return copies of messages and tools with cache_control injected."""
        new_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg["content"]
                if isinstance(content, str):
                    new_content = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
                else:
                    new_content = list(content)
                    new_content[-1] = {**new_content[-1], "cache_control": {"type": "ephemeral"}}
                new_messages.append({**msg, "content": new_content})
            else:
                new_messages.append(msg)

        new_tools = tools
        if tools:
            new_tools = list(tools)
            new_tools[-1] = {**new_tools[-1], "cache_control": {"type": "ephemeral"}}

        return new_messages, new_tools

    def _apply_model_overrides(self, model: str, kwargs: dict[str, Any]) -> None:
        """Apply model-specific parameter overrides from the registry."""
        model_lower = model.lower()
        # First check gateway overrides if available
        if self._gateway:
            for pattern, overrides in self._gateway.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    return
        # Then check standard provider overrides
        spec = find_by_model(model)
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    logger.debug(f"Applied standard overrides: {overrides}")
                    return
    
    @staticmethod
    def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip non-standard keys and ensure assistant messages have a content key."""
        sanitized = []
        for msg in messages:
            clean = {k: v for k, v in msg.items() if k in _ALLOWED_MSG_KEYS}
            # Strict providers require "content" even when assistant only has tool_calls
            if clean.get("role") == "assistant" and "content" not in clean:
                clean["content"] = None
            sanitized.append(clean)
        return sanitized

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4-5').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        original_model = model or self.default_model
        model = self._resolve_model(original_model)

        if self._supports_cache_control(original_model):
            messages, tools = self._apply_cache_control(messages, tools)

        # Clamp max_tokens to at least 1 — negative or zero values cause
        # LiteLLM to reject the request with "max_tokens must be at least 1".
        max_tokens = max(1, max_tokens)
        
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._sanitize_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Apply model-specific overrides (e.g. kimi-k2.5 temperature)
        self._apply_model_overrides(model, kwargs)
        
        # Pass api_key directly — more reliable than env vars alone
        if self.api_key:
            kwargs["api_key"] = self.api_key
        
        # Pass api_base for custom endpoints
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        # Pass extra headers (e.g. APP-Code for AiHubMix)
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        max_retries = 2  # Try original model twice (initial + 1 retry)
        retry_delay = 1  # Initial delay in seconds
        
        # Try original model first
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries} to call LLM: {model}")
                response = await acompletion(**kwargs)
                parsed_response = self._parse_response(response)
                
                # Store model name in response for channel to use
                parsed_response.model = model
                
                # Record token usage with original model name
                self._record_token_usage(parsed_response, original_model)
                
                return parsed_response
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
                logger.info(f"Error details: {error_msg}")
                
                # Send error notification to client if callback is provided
                error_notification = f"**{model}** 调用失败 (尝试 {attempt + 1}/{max_retries})\n原因: {error_msg}\n正在进行重试..."
                await self._send_error_notification(error_notification)
                
                # Check if this is the last attempt
                if attempt == max_retries - 1:
                    logger.info(f"All attempts failed, checking for fallback model")
                    break
                
                # Wait before retrying
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
        
        # Try fallback model if original failed
        logger.info(f"Checking fallback model configuration: self.fallback_model = {self.fallback_model}")
        if self.fallback_model:
            logger.info(f"Fallback model is configured: {self.fallback_model}")
            # Resolve fallback model name to compare with current model
            fallback_model_resolved = self._resolve_model(self.fallback_model)
            logger.info(f"Resolved fallback model: {fallback_model_resolved}")
            
            if model != fallback_model_resolved:
                logger.info(f"Switching to fallback model: {self.fallback_model} (resolved: {fallback_model_resolved}, current: {model})")
                
                # Create fallback kwargs with the same parameters but different model
                fallback_kwargs = kwargs.copy()
                fallback_kwargs["model"] = fallback_model_resolved
                
                # Apply model overrides for fallback model
                self._apply_model_overrides(fallback_model_resolved, fallback_kwargs)
                
                try:
                    response = await acompletion(**fallback_kwargs)
                    parsed_response = self._parse_response(response)
                    
                    # Store model name in response for channel to use
                    parsed_response.model = fallback_model_resolved
                    
                    # Record token usage for fallback model with original model name
                    self._record_token_usage(parsed_response, self.fallback_model)
                    
                    return parsed_response
                except Exception as e:
                    fallback_error = str(e)
                    logger.error(f"Fallback model call failed: {fallback_error}")
                    logger.info(f"Fallback error details: {fallback_error}")
                    
                    # Send error notification to client if callback is provided
                    error_notification = f"**{fallback_model_resolved}** (备用模型) 调用失败\n原因: {fallback_error}\n所有尝试都已失败，将返回错误信息..."
                    await self._send_error_notification(error_notification)
        
        # All attempts failed
        logger.error(f"LLM call failed after all attempts")
        
        # Return simple error message since user already received detailed notifications
        return LLMResponse(
            content="LLM调用失败: 所有模型尝试均已失败，请稍后重试或检查配置。",
            finish_reason="error",
        )
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string if needed
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json_repair.loads(args)
                
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        reasoning_content = getattr(message, "reasoning_content", None)
        
        # Handle case where content might be a dict instead of string
        content = message.content
        if isinstance(content, dict):
            content = str(content)
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )

    async def _send_error_notification(self, notification: str) -> None:
        """Send error notification to client if callback is provided."""
        if self.error_callback:
            try:
                if asyncio.iscoroutinefunction(self.error_callback):
                    await self.error_callback(notification)
                else:
                    self.error_callback(notification)
            except Exception as callback_error:
                logger.warning(f"Error sending error notification: {callback_error}")

    def _record_token_usage(self, response: LLMResponse, model: str) -> None:
        """Record token usage to SQLite database."""
        if response.usage:
            try:
                from nanobot.session.token_store import TokenStore
                token_store = TokenStore()
                token_store.record_usage(
                    session_key=self.session_key,
                    model=model,
                    prompt_tokens=response.usage.get("prompt_tokens", 0),
                    completion_tokens=response.usage.get("completion_tokens", 0)
                )
                logger.debug(
                    f"Recorded token usage: {self.session_key} / {model} "
                    f"= {response.usage.get('prompt_tokens', 0)} + {response.usage.get('completion_tokens', 0)}"
                )
            except Exception as e:
                logger.warning(f"Failed to record token usage: {e}")

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
