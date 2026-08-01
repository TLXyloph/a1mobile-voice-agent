"""LLM access with an explicit, single-valued model choice and a real fallback.

Findings that shaped this module (all established by live probe, not assumed):

* ``OPENAI_BASE_URL`` in config/.env points at an event-provided Lambda. Every
  route tested returned 404 -- ``/models`` and ``/chat/completions``, GET and
  POST, with and without the ``/openai/v1`` prefix. The bodies were
  OpenAI-shaped errors, so it is an OpenAI-compatible proxy that currently
  serves no routes. It is unusable.
* The live path is LiveKit's inference gateway at
  ``https://agent-gateway.livekit.cloud/v1``, authenticated with a short-lived
  LiveKit JWT carrying ``InferenceGrants(perform=True)``. This is what
  ``INFERENCE=livekit`` in the repo's env refers to.
* ``openai/gpt-5.6`` is real on that gateway and resolves to ``gpt-5.6-sol``.

The model is a single configurable value (``HELYX_MODEL``) defaulting to
``openai/gpt-5.6``, with ``HELYX_FALLBACK_MODEL`` (``openai/gpt-5.4``) used
only if the primary genuinely fails. Which model served a request is recorded
and surfaced on the dashboard, so nothing has to be taken on trust.
"""

from __future__ import annotations

import datetime
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .config import secret, settings

logger = logging.getLogger("helyx.llm")

_TOKEN_TTL_SECONDS = 900


class LLMError(RuntimeError):
    """Raised when no configured model could serve a request."""


@dataclass
class Completion:
    """The result of one gateway call, including which model actually served it."""

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    requested_model: str = ""
    served_model: str = ""
    fell_back: bool = False

    def first_tool_args(self, name: str) -> dict[str, Any] | None:
        for call in self.tool_calls:
            fn = call.get("function", {})
            if fn.get("name") == name:
                try:
                    return json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    logger.warning("tool call %s had unparseable arguments", name)
                    return None
        return None


def _access_token() -> str:
    """Mint a short-lived LiveKit inference token. Never logged."""
    from livekit import api

    key, sec = secret("LIVEKIT_API_KEY"), secret("LIVEKIT_API_SECRET")
    if not key or not sec:
        raise LLMError("LIVEKIT_API_KEY / LIVEKIT_API_SECRET are required for inference")
    return (
        api.AccessToken(key, sec)
        .with_identity("helyx")
        .with_inference_grants(api.access_token.InferenceGrants(perform=True))
        .with_ttl(datetime.timedelta(seconds=_TOKEN_TTL_SECONDS))
        .to_jwt()
    )


def _post(url: str, payload: dict[str, Any], token: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class LLMClient:
    """Thin OpenAI-compatible chat client against the LiveKit gateway."""

    def __init__(
        self,
        model: str | None = None,
        fallback_model: str | None = None,
        gateway_url: str | None = None,
    ) -> None:
        cfg = settings()
        self.model = model or cfg.model
        self.fallback_model = fallback_model or cfg.fallback_model
        self.gateway_url = (gateway_url or cfg.gateway_url).rstrip("/")
        #: Populated after the first successful call, for dashboard display.
        self.last_served_model: str = ""

    @property
    def candidates(self) -> list[str]:
        out = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            out.append(self.fallback_model)
        return out

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        timeout: float = 90.0,
    ) -> Completion:
        """Call the gateway, falling back to the secondary model on failure."""
        token = _access_token()
        errors: list[str] = []

        for index, model in enumerate(self.candidates):
            payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            # gpt-5.x are reasoning models and reject sampling parameters.
            if temperature is not None and not _is_reasoning(model):
                payload["temperature"] = temperature

            try:
                data = _post(
                    f"{self.gateway_url}/chat/completions", payload, token, timeout
                )
            except urllib.error.HTTPError as exc:
                body = exc.read()[:300].decode("utf-8", "replace")
                errors.append(f"{model}: HTTP {exc.code} {body}")
                logger.warning("model %s failed: HTTP %s", model, exc.code)
                continue
            except Exception as exc:  # noqa: BLE001 - surface any transport failure
                errors.append(f"{model}: {exc!r}")
                logger.warning("model %s failed: %r", model, exc)
                continue

            message = data["choices"][0]["message"]
            self.last_served_model = data.get("model", model)
            return Completion(
                text=(message.get("content") or "").strip(),
                tool_calls=list(message.get("tool_calls") or []),
                requested_model=model,
                served_model=data.get("model", model),
                fell_back=index > 0,
            )

        raise LLMError("all candidate models failed: " + " | ".join(errors))


def _is_reasoning(model: str) -> bool:
    name = model.split("/")[-1]
    return name.startswith(("gpt-5", "o1", "o3", "o4"))


def probe_models(models: list[str]) -> dict[str, dict[str, Any]]:
    """Live check of which model ids actually answer. Used by scripts/probe_models."""
    token = _access_token()
    url = settings().gateway_url.rstrip("/") + "/chat/completions"
    out: dict[str, dict[str, Any]] = {}
    for m in models:
        payload = {
            "model": m,
            "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
            "stream": False,
        }
        try:
            data = _post(url, payload, token, 90.0)
            out[m] = {
                "ok": True,
                "served_model": data.get("model"),
                "reply": (data["choices"][0]["message"].get("content") or "").strip()[:40],
            }
        except urllib.error.HTTPError as exc:
            out[m] = {
                "ok": False,
                "error": f"HTTP {exc.code}",
                "body": exc.read()[:200].decode("utf-8", "replace"),
            }
        except Exception as exc:  # noqa: BLE001
            out[m] = {"ok": False, "error": repr(exc)[:200]}
    return out
