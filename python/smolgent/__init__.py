"""Small, stateful agents backed by the smolgent Rust library."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, overload

from ._native import NativeAgent, SmolgentError, set_api_key
from ._media import (
    Audio, Content, File, Image, MultimodalResult, Video, serialize_prompt, serialize_tool_result,
)
from ._schema import infer_parameters

__all__ = ["Agent", "Audio", "File", "Image", "MultimodalResult", "Response",
           "SmolgentError", "Tool", "Video", "set_api_key", "tool"]

Path = str | os.PathLike[str]


@dataclass(frozen=True)
class Response:
    """Final answer plus the unmodified provider payload and reasoning metadata."""

    text: str
    message: dict[str, Any]
    reasoning: dict[str, Any] | None
    raw: dict[str, Any]

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class Tool:
    """A Python callable accepting keyword arguments described by a JSON Schema.

    Synchronous callables run in asyncio's thread pool. Async callables run on the
    caller's event loop. Return JSON values, media wrappers, Pillow images, or a
    MultimodalResult. Failures are sent back to the model as tool errors, allowing
    it to correct arguments and retry.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.name):
            raise ValueError("tool name must contain 1–64 letters, digits, underscores or hyphens")
        if self.parameters.get("type") != "object":
            raise ValueError("tool parameters must be a JSON Schema with type='object'")
        if not callable(self.handler):
            raise TypeError("tool handler must be callable")
        json.dumps(self.parameters, allow_nan=False)

    def _definition(self) -> str:
        return json.dumps({"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters,
        }}, allow_nan=False)

    async def _invoke(self, arguments_json: str) -> str:
        arguments = json.loads(arguments_json)
        if not isinstance(arguments, dict):
            raise TypeError("tool arguments must be an object")
        if inspect.iscoroutinefunction(self.handler):
            result = await self.handler(**arguments)
        else:
            result = await asyncio.to_thread(self.handler, **arguments)
            if inspect.isawaitable(result):
                result = await result
        # Image encoding can be expensive; keep it off the caller's event loop.
        return await asyncio.to_thread(serialize_tool_result, result)


@overload
def tool(handler: Callable[..., Any], /, *, parameters: dict[str, Any] | None = None,
         name: str | None = None, description: str | None = None) -> Tool: ...


@overload
def tool(*, parameters: dict[str, Any] | None = None, name: str | None = None,
         description: str | None = None) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(handler: Callable[..., Any] | None = None, /, *,
         parameters: dict[str, Any] | None = None, name: str | None = None,
         description: str | None = None) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Create a tool using argument type hints and the function's docstring.

    Use @tool, @tool(), or @tool(name=..., description=...). Parameters with
    defaults may be omitted. Explicit parameters= overrides schema inference.
    Missing or unsupported argument annotations require an explicit schema.
    """
    def decorate(handler: Callable[..., Any]) -> Tool:
        return Tool(name or handler.__name__,
                    description if description is not None else inspect.getdoc(handler) or "",
                    parameters if parameters is not None else infer_parameters(handler), handler)
    return decorate if handler is None else decorate(handler)


class Agent:
    """A provider, conversation, and tool registry in one object.

    Use :meth:`openrouter`, :meth:`deepseek`, :meth:`llama_cpp`, or :meth:`compatible` to select a
    provider. No filesystem tools are enabled unless roots are supplied. Write
    roots also grant read access. Relative paths resolve against the process's
    working directory, not against the first root.

    One run at a time is allowed per agent. Failed or cancelled runs discard their
    conversation changes, but completed external tool effects remain. Running
    synchronous tools cannot be forcibly stopped on cancellation.
    """

    def __init__(
        self,
        model: str,
        *,
        provider: str = "openrouter",
        endpoint: str | None = None,
        api_key: str | None = None,
        keyring_id: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
        tools: Iterable[Tool] = (),
        read_roots: Iterable[Path] = (),
        write_roots: Iterable[Path] = (),
        max_tool_rounds: int = 32,
        compaction: bool = True,
        timeout: float = 120.0,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if provider not in {"openrouter", "deepseek", "llama_cpp", "compatible"}:
            raise ValueError("unknown provider")
        if thinking is not None and not isinstance(thinking, bool):
            raise ValueError("thinking must be a bool or None")
        if reasoning_effort is not None and reasoning_effort not in {"none", "low", "high", "max"}:
            raise ValueError("reasoning_effort must be none, low, high, or max")
        if thinking is not None or reasoning_effort is not None:
            if provider != "deepseek":
                raise ValueError("thinking and reasoning_effort are currently DeepSeek options")
            if thinking is not None and reasoning_effort is not None and thinking == (reasoning_effort == "none"):
                raise ValueError("thinking conflicts with reasoning_effort")
        if provider == "openrouter" and endpoint is not None:
            raise ValueError("use Agent.compatible() for a custom endpoint")
        if api_key is not None and keyring_id is not None:
            raise ValueError("choose api_key or keyring_id")
        if keyring_id is not None and not keyring_id.strip():
            raise ValueError("keyring_id must be nonempty")
        if provider in {"openrouter", "deepseek"} and api_key is None and keyring_id is None:
            variable = f"{provider.upper()}_API_KEY"
            api_key = os.environ.get(variable)
            if not api_key:
                raise ValueError(f"set {variable} or supply api_key or keyring_id")
        if api_key is not None and not api_key.strip():
            raise ValueError("api_key must be nonempty")
        if isinstance(max_tool_rounds, bool) or not isinstance(max_tool_rounds, int) or max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must be a nonnegative integer")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self._tools = tuple(tools)
        if not all(isinstance(item, Tool) for item in self._tools):
            raise TypeError("tools must contain Tool objects (use @tool or Tool(...))")
        names = [item.name for item in self._tools]
        reserved = {"compact_remove_messages", "compact_summarize_messages"}
        read_paths = _paths(read_roots)
        write_paths = _paths(write_roots)
        if read_paths or write_paths:
            reserved.update({"read", "ripgrep"})
        if write_paths:
            reserved.update({"create_file", "delete_file", "apply_patch"})
        if len(set(names)) != len(names) or reserved.intersection(names):
            raise ValueError("duplicate or reserved tool name")
        self._native = NativeAgent(json.dumps({
            "provider": provider, "model": model, "endpoint": endpoint,
            "api_key": api_key, "keyring_id": keyring_id,
            "system_prompt": system_prompt, "read_roots": read_paths,
            "write_roots": write_paths, "max_tool_rounds": max_tool_rounds,
            "compaction": compaction, "timeout": timeout,
            "thinking": thinking, "reasoning_effort": reasoning_effort,
        }, allow_nan=False))

    @classmethod
    def openrouter(cls, model: str, **kwargs: Any) -> Agent:
        """Use OpenRouter, reading OPENROUTER_API_KEY unless credentials are supplied."""
        return cls(model, provider="openrouter", **kwargs)

    @classmethod
    def deepseek(cls, model: str = "deepseek-flash", *, base_url: str | None = None,
                 thinking: bool | None = None, reasoning_effort: str | None = None,
                 **kwargs: Any) -> Agent:
        """Use DeepSeek, reading DEEPSEEK_API_KEY unless credentials are supplied.

        Omitted thinking options use the API defaults. Set thinking=False to
        disable thinking, or reasoning_effort to low, high, or max to enable it
        at that effort (none disables it). base_url is a server root or prefix,
        such as https://api.deepseek.com/v1, without /chat/completions.
        """
        return cls(model, provider="deepseek", endpoint=base_url, thinking=thinking,
                   reasoning_effort=reasoning_effort, **kwargs)

    @classmethod
    def llama_cpp(cls, model: str, *, base_url: str = "http://127.0.0.1:8080",
                  **kwargs: Any) -> Agent:
        """Use a llama.cpp server root URL (without /v1/chat/completions)."""
        return cls(model, provider="llama_cpp", endpoint=base_url, **kwargs)

    @classmethod
    def compatible(cls, model: str, *, chat_completions_url: str,
                   **kwargs: Any) -> Agent:
        """Use an OpenAI-compatible provider's full chat-completions URL."""
        return cls(model, provider="compatible", endpoint=chat_completions_url, **kwargs)

    async def arun(self, prompt: Content) -> Response:
        """Run the Rust agent loop, executing tools until a final answer arrives.

        Accepts text, media wrappers, Pillow images, MultimodalResult, or a list
        mixing these parts with OpenAI-style content dictionaries.
        The timeout option limits each HTTP request, not the entire agent run.
        Use asyncio.wait_for for an overall deadline.
        """
        active_tools: set[asyncio.Task[Any]] = set()
        closed = False

        def adapter(item: Tool) -> Callable[[str], Awaitable[str]]:
            async def invoke(arguments: str) -> str:
                # A callback may already be queued when cancellation crosses from
                # Python to Rust. Prevent that callback from starting new work.
                if closed:
                    raise asyncio.CancelledError()
                task = asyncio.current_task()
                assert task is not None
                active_tools.add(task)
                try:
                    return await item._invoke(arguments)
                finally:
                    active_tools.discard(task)
            return invoke

        try:
            result = await self._native.arun(
                await asyncio.to_thread(serialize_prompt, prompt),
                [(item._definition(), adapter(item)) for item in self._tools],
            )
            return Response(**json.loads(result))
        finally:
            # Dropping the Rust future alone does not cancel Python callbacks.
            closed = True
            pending = tuple(active_tools)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    def run(self, prompt: Content) -> Response:
        """Blocking convenience method. In notebooks/async apps, await arun instead."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(prompt))
        raise RuntimeError("an event loop is running; use 'await agent.arun(prompt)'")

    @property
    def messages(self) -> list[dict[str, Any]]:
        """A detached snapshot of conversation history, including reasoning and tools."""
        return json.loads(self._native.messages())

    def reset(self) -> None:
        """Clear conversation history while preserving the configured system prompt."""
        self._native.reset()


def _paths(paths: Iterable[Path]) -> list[str]:
    if isinstance(paths, (str, bytes, os.PathLike)):
        raise TypeError("roots must be a sequence of paths, for example read_roots=['.']")
    return [os.fsdecode(os.fspath(path)) for path in paths]
