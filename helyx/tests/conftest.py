from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from helyx.llm import Completion


@dataclass
class FakeLLM:
    """Stands in for LLMClient. Scripted replies, recorded prompts.

    The negotiator makes two kinds of call per turn: one *with* tools (to hear
    what the supplier offered) and one *without* (to produce speech). They are
    scripted separately so a test can drive each independently.
    """

    #: replies for calls that pass tools -> (text, tool_arguments | None)
    replies: list[tuple[str, dict[str, Any] | None]] = field(default_factory=list)
    #: replies for calls with no tools -> the spoken line
    speech: list[str] = field(default_factory=list)
    tool_name: str = "record_parameters"
    calls: list[list[dict[str, Any]]] = field(default_factory=list)
    model: str = "openai/gpt-5.6"
    fallback_model: str = "openai/gpt-5.4"

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        timeout: float = 90.0,
    ) -> Completion:
        self.calls.append(messages)
        if not tools:
            line = self.speech.pop(0) if self.speech else ""
            return Completion(text=line, served_model="gpt-5.6-sol")
        if not self.replies:
            return Completion(text="ok", served_model="gpt-5.6-sol")
        text, args = self.replies.pop(0)
        tool_calls = []
        if args is not None:
            name = (tools[0]["function"]["name"] if tools else self.tool_name)
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ]
        return Completion(
            text=text,
            tool_calls=tool_calls,
            requested_model=self.model,
            served_model="gpt-5.6-sol",
        )


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM()
