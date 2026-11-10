#!/usr/bin/env python3
"""A Google Gemini powered Discord chat bot that talks back out loud.

Highlights
----------
* Current Gemini models (``gemini-3.8-flash`` and friends) with ``!model`` / ``!models``
  switching, plus a live model list pulled straight from the API.
* Three interchangeable text-to-speech engines, switchable at runtime with ``!tts``:
    - ``edge``   Microsoft Edge neural voices (free, very natural, ~400 voices)
    - ``gemini`` Gemini 3.8 TTS (most expressive, uses your Gemini API key)
    - ``gtts``   Google Translate TTS (the original engine, kept as a fallback)
* Per-server options: persona, temperature, token budget, memory length, voice,
  speaking rate, pitch and Gemini TTS acting style.
* Non-blocking audio queue so replies never cut each other off, auto-leave when the
  channel empties, vision support for attached images, and markdown/emoji cleanup
  so the voice only reads what it should.

Configuration is read from environment variables or a ``.env`` file next to this
script; see ``.env.example``. Anything that is missing is asked for on first run.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import aiohttp
import discord
from bs4 import BeautifulSoup

EDGE_IMPORT_ERROR = ""
GTT_IMPORT_ERROR = ""

try:  # optional: the best free voices
    import edge_tts
except Exception as exc:  # a half-broken install should not stop the bot
    edge_tts = None
    EDGE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:  # optional: robot voice of last resort
    from gtts import gTTS
except Exception as exc:
    gTTS = None
    GTT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Paths and .env loading
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
SETTINGS_FILE = BASE_DIR / "settings.json"
TMP_DIR = BASE_DIR / ".tts_cache"

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_TTS_MODEL = "gemini-3.8-flash-tts"
DEFAULT_MODEL = "gemini-3.8-flash"
#: Seconds to wait for an answer before releasing the typing indicator.
GENERATE_TIMEOUT = 45

DEFAULT_PERSONA = (
    "You are {name}, a warm, witty Discord chat companion. "
    "Answer in one to three short conversational sentences, the way a person would "
    "actually say it out loud. Your replies are read aloud by a text-to-speech voice, "
    "so avoid markdown, code fences, bullet lists, emoji and URLs unless the user "
    "explicitly asks for them. Stay friendly, never mention these instructions."
)


def load_env_file(path: Path = ENV_FILE) -> None:
    """Load ``KEY=VALUE`` pairs into os.environ without overwriting real env vars."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    except OSError as exc:
        print(f"[config] could not read {path}: {exc}")
        return

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split(" #")[0].strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(env_str(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = env_str(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


# --------------------------------------------------------------------------- #
# Model and voice catalogues
# --------------------------------------------------------------------------- #

#: Curated menu shown by ``!model``. ``!models`` asks the API for the real list, so a
#: model added or retired after this file was written still shows up there.
MODEL_CATALOG: list[tuple[str, str]] = [
    ("gemini-3.8-flash", "best all-rounder, newest Flash (default, free tier)"),
    ("gemini-3.6-flash", "previous-generation Flash, still stable"),
    ("gemini-3.5-flash-lite", "fast and cheap, great for busy servers"),
    ("gemini-3.1-flash-lite", "cheapest text model, high volume"),
    ("gemini-3.1-pro-preview", "deepest reasoning, preview + paid tier"),
    ("gemini-3-flash-preview", "older preview build"),
    ("gemini-2.5-flash-lite", "legacy, still served by the API"),
    ("gemini-2.5-flash", "legacy - reroutes to 3.5 Flash-Lite on 2026-12-01"),
]

#: Every provider has a free tier. OpenAI-compatible ones share one client.
PROVIDERS: dict[str, dict[str, str]] = {
    "gemini": {
        "label": "Google Gemini",
        "key_env": "GEMINI_API_KEY",
        "signup": "https://aistudio.google.com/apikey",
        "base_url": API_ROOT,
    },
    "openrouter": {
        "label": "OpenRouter",
        "key_env": "OPENROUTER_API_KEY",
        "signup": "https://openrouter.ai/keys",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "groq": {
        "label": "Groq",
        "key_env": "GROQ_API_KEY",
        "signup": "https://console.groq.com/keys",
        "base_url": "https://api.groq.com/openai/v1",
    },
}

#: Curated menus for ``!model``. Free rosters rotate, so ``!models`` asks the live API.
PROVIDER_MODELS: dict[str, list[tuple[str, str]]] = {
    "gemini": MODEL_CATALOG,
    "openrouter": [
        ("openrouter/free", "auto-picks a working free model (default)"),
        ("google/gemma-4-31b-it:free", "Google Gemma 4 31B, 262K context"),
        ("google/gemma-4-26b-a4b-it:free", "smaller Gemma 4, quick"),
        ("meta-llama/llama-3.3-70b-instruct:free", "Meta Llama 3.3 70B"),
        ("meta-llama/llama-4-maverick:free", "Meta Llama 4 Maverick"),
        ("qwen/qwen3-next-80b-a3b-instruct:free", "Qwen3 Next 80B"),
        ("deepseek/deepseek-chat-v3-0324:free", "DeepSeek V3 chat"),
        ("openai/gpt-oss-20b:free", "OpenAI GPT-OSS 20B"),
        ("nvidia/nemotron-3-super-120b-a12b:free", "NVIDIA Nemotron 3 Super 120B"),
    ],
    "groq": [
        ("openai/gpt-oss-120b", "flagship open-weight model, very fast (default)"),
        ("openai/gpt-oss-20b", "smaller and faster"),
        ("qwen/qwen3.8-27b", "Alibaba Qwen 3.8 27B"),
    ],
}

DEFAULT_MODELS: dict[str, str] = {
    "gemini": DEFAULT_MODEL,
    "openrouter": "openrouter/free",
    "groq": "openai/gpt-oss-120b",
}


def configured_providers() -> list[str]:
    """Providers that have a key, in this file's preference order."""
    found: list[str] = []
    for name, info in PROVIDERS.items():
        key = env_str(info["key_env"])
        if not key and name == "gemini":
            key = env_str("GOOGLE_API_KEY")
        if key:
            found.append(name)
    return found


def resolve_provider(name: str, available: Optional[list[str]] = None) -> str:
    """``auto`` picks the first configured provider; an explicit choice is kept."""
    available = configured_providers() if available is None else available
    name = (name or "auto").lower().strip()
    if name in PROVIDERS:
        return name
    return available[0] if available else "gemini"


def default_model_for(provider: str) -> str:
    return DEFAULT_MODELS.get(provider, DEFAULT_MODEL)


def model_looks_like(model: str, provider: str) -> bool:
    """Whether a stored model id plausibly belongs to ``provider``."""
    model = (model or "").strip()
    if not model:
        return False
    if provider == "gemini":
        return model.startswith("gemini-")
    return "/" in model or model == "openrouter/free"


#: (voice id, alias, locale, gender, note)
EDGE_VOICES: list[tuple[str, str, str, str, str]] = [
    ("en-US-AvaNeural", "ava", "en-US", "female", "warm and natural (default)"),
    ("en-US-AndrewNeural", "andrew", "en-US", "male", "friendly, conversational"),
    ("en-US-EmmaNeural", "emma", "en-US", "female", "bright and upbeat"),
    ("en-US-BrianNeural", "brian", "en-US", "male", "casual and relaxed"),
    ("en-US-JennyNeural", "jenny", "en-US", "female", "clear, assistant-like"),
    ("en-US-GuyNeural", "guy", "en-US", "male", "energetic, newscaster"),
    ("en-US-AriaNeural", "aria", "en-US", "female", "expressive, chatty"),
    ("en-GB-SoniaNeural", "sonia", "en-GB", "female", "smooth British"),
    ("en-GB-RyanNeural", "ryan", "en-GB", "male", "calm British"),
    ("en-AU-NatashaNeural", "natasha", "en-AU", "female", "Australian"),
    ("en-CA-ClaraNeural", "clara", "en-CA", "female", "Canadian"),
    ("en-IN-NeerjaNeural", "neerja", "en-IN", "female", "Indian English"),
    ("es-ES-ElviraNeural", "elvira", "es-ES", "female", "Spanish"),
    ("fr-FR-DeniseNeural", "denise", "fr-FR", "female", "French"),
    ("de-DE-KatjaNeural", "katja", "de-DE", "female", "German"),
    ("ja-JP-NanamiNeural", "nanami", "ja-JP", "female", "Japanese"),
    ("pt-BR-FranciscaNeural", "francisca", "pt-BR", "female", "Brazilian Portuguese"),
]

#: Gemini TTS prebuilt voices (30 exist; these are the most distinct).
GEMINI_VOICES: list[tuple[str, str]] = [
    ("Kore", "firm, natural, good default"),
    ("Puck", "upbeat and playful"),
    ("Charon", "informative, deep"),
    ("Aoede", "breezy and light"),
    ("Fenrir", "excitable"),
    ("Leda", "youthful"),
    ("Orus", "firm and steady"),
    ("Zephyr", "bright"),
    ("Enceladus", "breathy, soft"),
    ("Algieba", "smooth"),
    ("Achird", "casual"),
    ("Sadaltager", "knowledgeable"),
]

TTS_ENGINES = ("auto", "edge", "gemini", "gtts", "off")

# --------------------------------------------------------------------------- #
# Small text utilities
# --------------------------------------------------------------------------- #

URL_RE = re.compile(r"https?://\S+")
CUSTOM_EMOJI_RE = re.compile(r"<a?:\w+:\d+>")
MENTION_RE = re.compile(r"<@[!&]?\d+>|@everyone|@here")
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`]*)`")
EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "\u2b00-\u2bff"
    "\ufe0f\u200d\u20e3"
    "]"
)
MARKDOWN_RE = re.compile(r"[*_#>~|]+")
#: ASCII emoticons (:) ;P <3 ...), but never times or ratios like 10:30.
EMOTICON_RE = re.compile(r"(?<![\w:])(?:[:;=8][-^']?[)(\[\]dDpP/\\|*]|<3)")


def text_for_speech(text: str) -> str:
    """Strip everything a TTS voice should not literally read out."""
    cleaned = CODE_BLOCK_RE.sub(" (code shared in chat) ", text)
    cleaned = INLINE_CODE_RE.sub(r"\1", cleaned)
    cleaned = CUSTOM_EMOJI_RE.sub("", cleaned)
    cleaned = MENTION_RE.sub("", cleaned)
    cleaned = URL_RE.sub(" link ", cleaned)
    cleaned = MARKDOWN_RE.sub(" ", cleaned)
    cleaned = EMOTICON_RE.sub("", cleaned)
    cleaned = EMOJI_RE.sub("", cleaned)
    cleaned = re.sub(r"\s*\n+\s*", ". ", cleaned)
    cleaned = re.sub(r"\s*\.\s*\.+", ".", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" .")


def split_text(text: str, limit: int = 1_900) -> list[str]:
    """Split text into Discord-sized chunks, preferring paragraph/sentence breaks."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = max(
            window.rfind("\n\n"),
            window.rfind("\n"),
            window.rfind(". "),
            window.rfind("! "),
            window.rfind("? "),
            window.rfind(" "),
        )
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return [chunk for chunk in chunks if chunk]


def clip_for_speech(text: str, limit: int = 1_200) -> str:
    if len(text) <= limit:
        return text
    window = text[:limit]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut < limit // 3:
        cut = window.rfind(" ")
    return (window[:cut] if cut > 0 else window).strip() + " ..."


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


# --------------------------------------------------------------------------- #
# Persistent per-server settings
# --------------------------------------------------------------------------- #


class SettingsStore:
    """Guild settings persisted to ``settings.json``, seeded from the environment."""

    def __init__(self, path: Path, defaults: dict[str, Any]):
        self.path = path
        self.defaults = dict(defaults)
        self._data: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._data = data if isinstance(data, dict) else {}
        except FileNotFoundError:
            self._data = {}
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[settings] ignoring unreadable {self.path.name}: {exc}")
            self._data = {}

    def get(self, guild_id: int) -> dict[str, Any]:
        merged = dict(self.defaults)
        stored = self._data.get(str(guild_id))
        if isinstance(stored, dict):
            merged.update(stored)
        return merged

    def update(self, guild_id: int, **values: Any) -> dict[str, Any]:
        entry = self._data.setdefault(str(guild_id), {})
        for key, value in values.items():
            if value is None or value == "":
                entry.pop(key, None)  # empty means "go back to the default"
            else:
                entry[key] = value
        self.save()
        return self.get(guild_id)

    def reset(self, guild_id: int) -> dict[str, Any]:
        self._data.pop(str(guild_id), None)
        self.save()
        return self.get(guild_id)

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            print(f"[settings] could not save: {exc}")


# --------------------------------------------------------------------------- #
# Gemini API client
# --------------------------------------------------------------------------- #


class GeminiError(RuntimeError):
    """Friendly, user-reportable Gemini API failure."""


def _api_error_message(status: int, body: str) -> str:
    detail = ""
    try:
        payload = json.loads(body)
        detail = payload.get("error", {}).get("message", "")
    except (ValueError, AttributeError):
        detail = body[:200]
    hints = {
        400: "the request was rejected (check the model name)",
        401: "the API key was rejected",
        402: "this key has no credit left",
        403: "this API key is not allowed to use that model",
        404: "that model does not exist any more - try !models",
        429: "rate limit reached, give it a few seconds",
    }
    hint = hints.get(status, "the API returned an error")
    return f"{hint} ({status})" + (f": {detail}" if detail else "")


class GeminiClient:
    """Thin async wrapper around the REST ``generateContent`` endpoints."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=60, connect=15, sock_read=45),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _post(self, path: str, payload: dict, retries: int = 3) -> dict:
        session = await self._get_session()
        delay = 1.0
        last_error = "request failed"
        for attempt in range(1, retries + 1):
            try:
                async with session.post(f"{API_ROOT}/{path}", json=payload) as resp:
                    body = await resp.text()
                    if resp.status == 200:
                        try:
                            return json.loads(body)
                        except ValueError as exc:
                            raise GeminiError(f"could not parse the API response: {exc}") from exc
                    if resp.status in (429, 500, 502, 503, 504) and attempt < retries:
                        last_error = _api_error_message(resp.status, body)
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    raise GeminiError(_api_error_message(resp.status, body))
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = f"network problem talking to Google: {exc}"
                if attempt >= retries:
                    break
                await asyncio.sleep(delay)
                delay *= 2
        raise GeminiError(last_error)

    async def generate(
        self,
        model: str,
        contents: list[dict],
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": max(0.0, min(2.0, temperature)),
                "maxOutputTokens": max(1, min(8_192, max_tokens)),
                "topP": 0.95,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        data = await self._post(f"models/{model}:generateContent", payload)
        return _extract_text(data)

    async def list_models(self) -> list[str]:
        session = await self._get_session()
        try:
            async with session.get(f"{API_ROOT}/models", params={"pageSize": 200}) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise GeminiError(_api_error_message(resp.status, body))
                data = json.loads(body)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise GeminiError(f"network problem talking to Google: {exc}") from exc

        skip = ("embedding", "image", "tts", "live", "transcribe", "aqa", "banana", "veo", "imagen")
        names: set[str] = set()
        for model in data.get("models", []):
            if "generateContent" not in (model.get("supportedGenerationMethods") or []):
                continue
            name = str(model.get("name", "")).split("/")[-1]
            if not name or any(token in name for token in skip):
                continue
            names.add(name)
        return sorted(names, reverse=True)

    async def text_to_speech(self, text: str, voice: str, style: str) -> bytes:
        """Return a WAV file for ``text`` using Gemini 3.8 Flash TTS."""
        part: dict[str, Any] = {"text": text}
        if style:
            part["speechMetadata"] = {"style": style}
        payload = {
            "contents": [{"role": "user", "parts": [part]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"voice": voice}},
            },
        }
        data = await self._post(f"models/{GEMINI_TTS_MODEL}:generateContent", payload)
        candidates = data.get("candidates") or []
        for part in (candidates[0].get("content", {}).get("parts") or []) if candidates else []:
            blob = part.get("inlineData") or part.get("inline_data")
            if blob and blob.get("data"):
                return base64.b64decode(blob["data"])
        raise GeminiError("Gemini TTS returned no audio (try !tts edge instead)")


def contents_to_messages(contents: list[dict], system: str = "") -> list[dict]:
    """Translate the Gemini-style conversation into OpenAI chat messages."""
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    for turn in contents:
        role = "assistant" if turn.get("role") == "model" else "user"
        parts = turn.get("parts") or []
        texts = [str(part.get("text")) for part in parts if part.get("text")]
        images = [part["inlineData"] for part in parts if part.get("inlineData")]
        if images:
            content: list[dict[str, Any]] = [
                {"type": "text", "text": "\n".join(texts) or "What is in this image?"}
            ]
            for blob in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:%s;base64,%s"
                            % (blob.get("mimeType", "image/png"), blob.get("data", ""))
                        },
                    }
                )
            messages.append({"role": role, "content": content})
        elif texts:
            messages.append({"role": role, "content": "\n".join(texts)})
    return messages


class OpenAICompatClient:
    """Minimal async client for OpenAI-compatible free tiers (OpenRouter, Groq)."""

    def __init__(self, provider: str, api_key: str):
        self.provider = provider
        self.label = PROVIDERS[provider]["label"]
        self.base_url = PROVIDERS[provider]["base_url"].rstrip("/")
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            if self.provider == "openrouter":
                headers["X-Title"] = "Discord Gemini Bot"
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60, connect=15, sock_read=45),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _post(self, path: str, payload: dict, retries: int = 3) -> dict:
        session = await self._get_session()
        delay = 1.0
        last_error = "request failed"
        for attempt in range(1, retries + 1):
            try:
                async with session.post(f"{self.base_url}/{path}", json=payload) as resp:
                    body = await resp.text()
                    if resp.status == 200:
                        try:
                            return json.loads(body)
                        except ValueError as exc:
                            raise GeminiError(f"{self.label} sent unreadable JSON: {exc}") from exc
                    if resp.status in (429, 500, 502, 503, 504) and attempt < retries:
                        last_error = _api_error_message(resp.status, body)
                        print(f"[ai] {self.label} attempt {attempt} failed: {last_error}")
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    raise GeminiError(_api_error_message(resp.status, body))
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = f"network problem talking to {self.label}: {exc}"
                print(f"[ai] {self.label} attempt {attempt}/{retries} failed: {exc}")
                if attempt >= retries:
                    break
                await asyncio.sleep(delay)
                delay *= 2
        raise GeminiError(last_error)

    async def generate(
        self,
        model: str,
        contents: list[dict],
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": contents_to_messages(contents, system),
            "temperature": max(0.0, min(2.0, temperature)),
            "max_tokens": max(1, min(8_192, max_tokens)),
        }
        data = await self._post("chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            error = (data.get("error") or {}).get("message")
            raise GeminiError(error or f"{self.label} returned no answer")
        message = choices[0].get("message") or {}
        answer = (message.get("content") or "").strip()
        if not answer:
            if message.get("reasoning") or message.get("reasoning_content"):
                raise GeminiError(
                    f"{self.label} only returned reasoning for `{model}`, try another model"
                )
            raise GeminiError(
                f"{self.label} returned an empty answer "
                f"({choices[0].get('finish_reason') or 'unknown'})"
            )
        return answer

    async def list_models(self) -> list[str]:
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/models") as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise GeminiError(_api_error_message(resp.status, body))
                data = json.loads(body)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise GeminiError(f"network problem talking to {self.label}: {exc}") from exc

        names: set[str] = set()
        for entry in data.get("data") or data.get("models") or []:
            name = str(entry.get("id") or entry.get("name") or "").strip()
            if not name:
                continue
            if self.provider == "openrouter" and not (
                name.endswith(":free") or name == "openrouter/free"
            ):
                continue
            skip = ("whisper", "tts", "embed", "guard", "rerank", "moderation")
            if any(token in name.lower() for token in skip):
                continue
            names.add(name)
        return sorted(names)


def _extract_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason")
        if reason:
            raise GeminiError(f"the model refused that request ({reason.lower()})")
        raise GeminiError("the model returned no answer")

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(part.get("text") or "" for part in parts if not part.get("thought")).strip()
    finish = candidate.get("finishReason")

    if not text:
        if finish in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}:
            raise GeminiError("that answer was blocked by the safety filters")
        if finish == "MAX_TOKENS":
            raise GeminiError("the model hit the token limit before answering - raise it with !tokens 1024")
        raise GeminiError(f"empty answer from the model (reason: {finish or 'unknown'})")
    if finish == "MAX_TOKENS":
        text += " ..."
    return text


# --------------------------------------------------------------------------- #
# Text to speech
# --------------------------------------------------------------------------- #


def resolve_ffmpeg() -> Optional[str]:
    candidates = [
        env_str("FFMPEG_PATH"),
        str(BASE_DIR / "ffmpeg.exe"),
        str(BASE_DIR / "ffmpeg"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return shutil.which("ffmpeg")


FFMPEG_PATH = resolve_ffmpeg()


def resolve_engine(name: str) -> str:
    """Turn ``auto``/``off`` into a concrete engine name."""
    name = (name or "auto").lower()
    if name == "auto":
        if edge_tts is not None:
            return "edge"
        if gTTS is not None:
            return "gtts"
        return "none"
    if name in {"edge", "gemini", "gtts", "off"}:
        return name
    return "edge" if edge_tts is not None else "gtts"


def resolve_edge_voice(name: str) -> str:
    needle = (name or "").strip()
    for voice_id, alias, *_ in EDGE_VOICES:
        if needle.lower() == alias or needle.lower() == voice_id.lower():
            return voice_id
    if "-" in needle:  # a raw Edge ShortName such as en-US-AndrewMultilingualNeural
        return needle
    for voice_id, alias, *_ in EDGE_VOICES:
        if needle.lower() in alias:
            return voice_id
    return EDGE_VOICES[0][0]


def resolve_gemini_voice(name: str) -> str:
    needle = (name or "").strip()
    for voice, _ in GEMINI_VOICES:
        if needle.lower() == voice.lower():
            return voice
    return GEMINI_VOICES[0][0]


def audio_source(path: Path) -> discord.AudioSource:
    kwargs: dict[str, Any] = {}
    if FFMPEG_PATH:
        kwargs["executable"] = FFMPEG_PATH
    return discord.FFmpegPCMAudio(str(path), **kwargs)


async def _edge_speech(text: str, voice: str, rate: str, pitch: str, path: Path) -> None:
    if edge_tts is None:
        raise RuntimeError("edge-tts is not installed (pip install edge-tts)")
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(str(path))
    if not path.exists() or path.stat().st_size < 512:
        raise RuntimeError("Edge TTS produced no audio")


async def _gtts_speech(text: str, lang: str, path: Path, slow: bool = False) -> None:
    if gTTS is None:
        raise RuntimeError("gTTS is not installed (pip install gTTS)")
    await asyncio.to_thread(gTTS(text=text, lang=lang, slow=slow).save, str(path))


async def _gemini_speech(client: GeminiClient, text: str, voice: str, style: str, path: Path) -> None:
    raw = await client.text_to_speech(text, voice, style)
    if raw[:4] == b"RIFF":
        path.write_bytes(raw)
        return
    if not FFMPEG_PATH:
        raise RuntimeError("ffmpeg is required to convert raw Gemini audio")
    pcm = path.with_suffix(".pcm")
    pcm.write_bytes(raw)
    try:
        proc = await asyncio.create_subprocess_exec(
            FFMPEG_PATH, "-y", "-loglevel", "error", "-f", "s16le",
            "-ar", "24000", "-ac", "1", "-i", str(pcm), str(path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to convert Gemini audio: {stderr.decode()[-300:]}")
    finally:
        pcm.unlink(missing_ok=True)


@dataclass
class SpeechJob:
    text: str
    settings: dict[str, Any]
    author: str = ""


class GuildPlayer:
    """Serial audio queue for one guild, with idle auto-disconnect."""

    def __init__(self, bot: "GeminiDiscordBot", voice_client: discord.VoiceClient):
        self.bot = bot
        self.vc = voice_client
        self.queue: asyncio.Queue[SpeechJob] = asyncio.Queue(maxsize=20)
        self.text_channel: Optional[discord.abc.Messageable] = None
        self.task: Optional[asyncio.Task] = None
        self.idle_since: Optional[float] = None
        self.stopping = False
        self.warned: set[str] = set()

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> None:
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._worker(), name=f"audio-{self.vc.guild.id}")

    async def disconnect(self, reason: str = "") -> None:
        self.stopping = True
        channel = self.text_channel
        try:
            if self.vc.is_playing():
                self.vc.stop()
            if self.vc.is_connected():
                await self.vc.disconnect(force=True)
        except Exception:  # pragma: no cover - discord edge cases
            pass
        finally:
            self.bot.players.pop(self.vc.guild.id, None)
            while not self.queue.empty():
                self.queue.get_nowait()
                self.queue.task_done()
        if reason and channel is not None:
            try:
                await channel.send(reason)
            except discord.HTTPException:
                pass

    def cancel(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()

    # -- queueing ----------------------------------------------------------- #
    def enqueue(self, text: str, settings: dict[str, Any], channel: Optional[discord.abc.Messageable], author: str) -> bool:
        if channel is not None:
            self.text_channel = channel
        if resolve_engine(settings["tts_engine"]) in {"off", "none"}:
            return False
        try:
            self.queue.put_nowait(SpeechJob(text=text, settings=dict(settings), author=author))
        except asyncio.QueueFull:
            print("[voice] queue full, dropping a line")
            return False
        self.idle_since = None
        return True

    # -- worker ------------------------------------------------------------- #
    async def _worker(self) -> None:
        try:
            while not self.stopping and self.vc.is_connected():
                try:
                    job = await asyncio.wait_for(self.queue.get(), timeout=10)
                except asyncio.TimeoutError:
                    if not self.vc.is_connected():
                        break
                    await self._maybe_leave()
                    continue

                try:
                    await self._speak(job)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # keep the queue alive no matter what
                    print(f"[voice] {exc}")
                    await self._report_failure(str(exc))
                finally:
                    self.queue.task_done()
                    self.idle_since = time.monotonic()
        finally:
            self.bot.players.pop(self.vc.guild.id, None)

    async def _maybe_leave(self) -> None:
        if not self.queue.empty():
            self.idle_since = None
            return
        now = time.monotonic()
        if self.idle_since is None:
            self.idle_since = now
            return
        alone = not self._humans_present()
        timeout = 30.0 if alone else float(self.bot.config["idle_leave_seconds"])
        if now - self.idle_since > timeout:
            reason = "I left the voice channel because everybody else did." if alone else "I left the voice channel after sitting idle."
            await self.disconnect(reason)

    def _humans_present(self) -> bool:
        channel = getattr(self.vc, "channel", None)
        if channel is None:
            return False
        return any(not member.bot for member in getattr(channel, "members", []))

    async def _speak(self, job: SpeechJob) -> None:
        speech = clip_for_speech(text_for_speech(job.text))
        if not speech:
            return
        path = await synthesize(speech, job.settings, self.bot.gemini, job.author)
        try:
            await self._play(path)
        finally:
            path.unlink(missing_ok=True)
            await asyncio.sleep(0.25)

    async def _play(self, path: Path) -> None:
        loop = asyncio.get_running_loop()
        finished: asyncio.Future[None] = loop.create_future()

        def _after(error: Optional[Exception]) -> None:
            if error:
                print(f"[voice] playback error: {error}")
            try:
                loop.call_soon_threadsafe(
                    lambda: finished.done() or finished.set_result(None)
                )
            except RuntimeError:  # loop already closed
                pass

        self.vc.play(audio_source(path), after=_after)
        try:
            await asyncio.wait_for(finished, timeout=300)
        except asyncio.TimeoutError:
            print("[voice] playback timed out, skipping")
            if self.vc.is_playing():
                self.vc.stop()

    async def _report_failure(self, message: str) -> None:
        key = message.split("(")[0][:60]
        if key in self.warned or self.text_channel is None:
            return
        self.warned.add(key)
        try:
            await self.text_channel.send(
                f"Voice output failed: {message[:300]}\n"
                "Try `!tts edge` (or `!tts gtts`) if this keeps happening.",
                delete_after=25,
            )
        except (discord.HTTPException, AttributeError):
            pass


async def synthesize(
    text: str,
    settings: dict[str, Any],
    client: GeminiClient,
    author: str = "",
) -> Path:
    """Render ``text`` to a local audio file using the configured engine."""
    TMP_DIR.mkdir(exist_ok=True)
    engine = resolve_engine(settings["tts_engine"])
    if engine in {"off", "none"}:
        raise RuntimeError("text-to-speech is switched off")

    # unique-ish filename so overlapping replies never collide
    stamp = f"{time.time_ns()}-{random.randrange(10_000):04d}"
    path = TMP_DIR / f"reply-{stamp}.{'wav' if engine == 'gemini' else 'mp3'}"

    try:
        if engine == "edge":
            await _edge_speech(
                text,
                resolve_edge_voice(settings["edge_voice"]),
                settings["tts_rate"],
                settings["tts_pitch"],
                path,
            )
        elif engine == "gemini":
            await _gemini_speech(
                client, text, resolve_gemini_voice(settings["gemini_voice"]), settings["tts_style"], path
            )
        else:
            await _gtts_speech(text, settings["tts_lang"], path)
    except Exception as exc:
        if engine == "gemini":
            raise
        # Edge/gTTS failing should not silence the bot entirely.
        print(f"[voice] {engine} failed ({exc}); falling back to gTTS")
        if gTTS is None:
            raise
        fallback = path.with_suffix(".mp3")
        await _gtts_speech(text, settings["tts_lang"], fallback)
        return fallback

    return path


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


@dataclass
class Command:
    name: str
    handler: Callable[..., Any]
    usage: str
    description: str
    group: str = "Chat"
    admin: bool = False
    aliases: tuple[str, ...] = field(default_factory=tuple)


COMMANDS: dict[str, Command] = {}


def command(
    name: str,
    usage: str = "",
    description: str = "",
    group: str = "Chat",
    admin: bool = False,
    aliases: tuple[str, ...] = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def register(func: Callable[..., Any]) -> Callable[..., Any]:
        spec = Command(name, func, usage, description, group, admin, aliases)
        COMMANDS[name] = spec
        for alias in aliases:
            COMMANDS[alias] = spec
        return func

    return register


def unique_commands() -> list[Command]:
    seen: set[str] = set()
    result: list[Command] = []
    for name, spec in COMMANDS.items():
        if spec.name in seen:
            continue
        seen.add(spec.name)
        result.append(spec)
    return result


JOKES = [
    "Why don't scientists trust atoms? Because they make up everything.",
    "I told my computer I needed a break, and it said no problem, I'll go to sleep.",
    "Why did the scarecrow win an award? Because he was outstanding in his field.",
    "I'm reading a book about anti-gravity. It's impossible to put down.",
    "What do you call a fake noodle? An impasta.",
    "Why did the developer go broke? Because he used up all his cache.",
    "I would tell you a UDP joke, but you might not get it.",
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "A SQL query walks into a bar, sees two tables and asks: may I join you?",
    "Why was the JavaScript developer sad? Because he didn't know how to null his feelings.",
]

QUOTES = [
    "The best way to get started is to quit talking and begin doing. - Walt Disney",
    "Don't let yesterday take up too much of today. - Will Rogers",
    "The only limit to our realization of tomorrow is our doubts of today. - Franklin D. Roosevelt",
    "It is never too late to be what you might have been. - George Eliot",
    "Simplicity is the soul of efficiency. - Austin Freeman",
    "Any sufficiently advanced technology is indistinguishable from magic. - Arthur C. Clarke",
    "The secret of getting ahead is getting started. - Mark Twain",
    "Done is better than perfect. - Sheryl Sandberg",
    "Whether you think you can or you think you can't, you're right. - Henry Ford",
    "Make it work, make it right, make it fast. - Kent Beck",
]


def guild_key(message: discord.Message) -> int:
    return message.guild.id if message.guild else 0


def can_configure(bot: "GeminiDiscordBot", message: discord.Message) -> bool:
    if bot.config["allow_everyone_to_configure"]:
        return True
    if message.guild is None:
        return True
    if message.author.id == bot.config["owner_id"]:
        return True
    if message.author == message.guild.owner:
        return True
    perms = message.channel.permissions_for(message.author)
    return bool(perms.administrator or perms.manage_guild)


async def _ensure_player(bot: "GeminiDiscordBot", message: discord.Message) -> Optional[GuildPlayer]:
    """Join the author's voice channel if needed and return the player."""
    if message.guild is None:
        await message.reply("Voice only works inside a server, not in DMs.")
        return None
    channel = getattr(message.author.voice, "channel", None)
    vc = message.guild.voice_client
    if vc and vc.is_connected():
        if channel and vc.channel and vc.channel.id != channel.id:
            await vc.move_to(channel)
    elif channel:
        try:
            vc = await channel.connect(self_deaf=True)
        except discord.ClientException as exc:
            await message.reply(f"I could not join {channel.mention}: {exc}")
            return None
        except asyncio.TimeoutError:
            await message.reply("Joining the voice channel timed out. Give it another try.")
            return None
    else:
        await message.reply("You need to be in a voice channel first.")
        return None

    player = bot.players.get(message.guild.id)
    if player is None or player.vc is not vc:
        player = GuildPlayer(bot, vc)
        bot.players[message.guild.id] = player
        player.start()
    return player


@command("help", "!help", "Show this list of commands.", group="Chat")
async def cmd_help(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    embed = discord.Embed(
        title="Commands",
        description=(
            f"Prefix: `{bot.prefix}`  |  model: `{settings['model']}`  |  "
            f"voice: `{resolve_engine(settings['tts_engine'])}`"
        ),
        colour=discord.Colour.blurple(),
    )
    groups: dict[str, list[str]] = {}
    for spec in sorted(unique_commands(), key=lambda s: (s.group, s.name)):
        line = f"`{spec.usage or bot.prefix + spec.name}` - {spec.description}"
        if spec.admin and not bot.config["allow_everyone_to_configure"]:
            line += " *(manage server)*"
        groups.setdefault(spec.group, []).append(line)
    for group, lines in groups.items():
        embed.add_field(name=group, value="\n".join(lines)[:1020], inline=False)
    embed.set_footer(text="Mention me, use my name, or !ask to chat. Settings are per server.")
    await message.reply(embed=embed, mention_author=False)


@command("ask", "!ask <message>", "Ask Gemini directly without mentioning the bot.", group="Chat")
async def cmd_ask(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    if not args:
        await message.reply(f"Usage: `{bot.prefix}ask what is a black hole?`")
        return
    await bot.handle_chat(message, args, settings)


@command("reset", "!reset [settings]", "Forget the conversation history (or reset all server settings).", group="Chat",
         aliases=("forget",))
async def cmd_reset(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    if args.strip().lower() in {"settings", "all", "config"}:
        bot.store.reset(guild_key(message))
        await message.reply("Server settings are back to the defaults.")
        return
    cleared = bot.clear_memory(message)
    await message.reply(f"Conversation memory cleared ({cleared} exchanges).")


@command("persona", "!persona [text|reset]", "Show or change the bot's personality prompt.", group="Options", admin=True)
async def cmd_persona(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    if not args:
        current = settings["persona"]
        await message.reply(
            "Current persona:\n>>> " + (current[:1500] if current else "(empty - uses the default)")
        )
        return
    if args.strip().lower() in {"reset", "default", "clear"}:
        bot.store.update(guild_key(message), persona="")
        bot.clear_guild_memory(guild_key(message))
        await message.reply("Persona reset to the default.")
        return
    bot.store.update(guild_key(message), persona=args.strip()[:2000])
    bot.clear_guild_memory(guild_key(message))
    await message.reply("Persona updated (conversation memory cleared).")


@command("temp", "!temp <0.0-2.0>", "Set how creative the replies are (default 0.8).", group="Options", admin=True,
         aliases=("temperature",))
async def cmd_temp(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    if not args:
        await message.reply(f"Temperature is `{settings['temperature']}`. Try `{bot.prefix}temp 1.2`.")
        return
    try:
        value = float(args.replace(",", "."))
    except ValueError:
        await message.reply("Give me a number between 0 and 2, for example `!temp 1.1`.")
        return
    if not 0.0 <= value <= 2.0:
        await message.reply("Temperature must be between 0 and 2.")
        return
    bot.store.update(guild_key(message), temperature=round(value, 2))
    await message.reply(f"Temperature set to `{value:.2f}`.")


@command("tokens", "!tokens <100-8192>", "Set the maximum reply length in tokens.", group="Options", admin=True)
async def cmd_tokens(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    if not args:
        await message.reply(f"Reply budget is `{settings['max_tokens']}` tokens.")
        return
    if not args.isdigit() or not 100 <= int(args) <= 8192:
        await message.reply("Pick a whole number between 100 and 8192.")
        return
    bot.store.update(guild_key(message), max_tokens=int(args))
    await message.reply(f"Reply budget set to `{int(args)}` tokens.")


@command("memory", "!memory <0-50>", "How many past exchanges to remember per user (0 = off).", group="Options",
         admin=True)
async def cmd_memory(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    if not args:
        turns = settings["memory_turns"]
        state = "off" if not turns else f"{turns} exchanges"
        await message.reply(f"Memory is {state} per user.")
        return
    if not args.isdigit() or int(args) > 50:
        await message.reply("Pick a whole number between 0 and 50.")
        return
    bot.store.update(guild_key(message), memory_turns=int(args))
    bot.clear_memory(message)
    await message.reply(f"Memory set to `{int(args)}` exchanges.")


@command("settings", "!settings", "Show every option for this server.", group="Options")
async def cmd_settings(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    engine = resolve_engine(settings["tts_engine"])
    voice = (
        resolve_gemini_voice(settings["gemini_voice"]) if engine == "gemini"
        else resolve_edge_voice(settings["edge_voice"])
    )
    persona = settings["persona"] or bot.store.defaults["persona"]
    embed = discord.Embed(title="Server settings", colour=discord.Colour.dark_teal())
    embed.add_field(name="Provider", value=f"`{bot.provider_for(settings)}`", inline=True)
    embed.add_field(name="Model", value=f"`{settings['model']}`", inline=True)
    embed.add_field(name="Temperature", value=f"`{settings['temperature']}`", inline=True)
    embed.add_field(name="Max tokens", value=f"`{settings['max_tokens']}`", inline=True)
    embed.add_field(name="Memory", value=f"`{settings['memory_turns']}` exchanges", inline=True)
    embed.add_field(name="TTS engine", value=f"`{engine}`", inline=True)
    embed.add_field(name="Voice", value=f"`{voice}`", inline=True)
    embed.add_field(name="Rate / pitch", value=f"`{settings['tts_rate']}` / `{settings['tts_pitch']}`", inline=True)
    embed.add_field(name="TTS style", value=f"`{settings['tts_style'] or '-'}`", inline=True)
    embed.add_field(name="Idle leave", value=f"`{format_seconds(settings['idle_leave_seconds'])}`", inline=True)
    embed.add_field(name="Persona", value=f">>> {persona[:900]}", inline=False)
    embed.set_footer(text=f"guild {guild_key(message)} | +{settings['ffmpeg'] or 'ffmpeg from PATH'}")
    await message.reply(embed=embed, mention_author=False)


@command("model", "!model [number|name]", "Switch the model for this server.", group="Models", admin=True)
async def cmd_model(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    provider = bot.provider_for(settings)
    catalog = PROVIDER_MODELS.get(provider, MODEL_CATALOG)
    if not args:
        lines = [
            f"`{index}.` `{model_id}` - {note}"
            + ("  **<- current**" if model_id == settings["model"] else "")
            for index, (model_id, note) in enumerate(catalog, 1)
        ]
        await message.reply(
            f"Current model: `{settings['model']}` via {PROVIDERS[provider]['label']}\n"
            + "\n".join(lines)
            + f"\nUse `{bot.prefix}model 1` or a full name. `{bot.prefix}models` lists what the API offers right now.",
            mention_author=False,
        )
        return

    choice = args.strip()
    if choice.isdigit():
        index = int(choice)
        if not 1 <= index <= len(catalog):
            await message.reply(f"Pick a number between 1 and {len(catalog)}.")
            return
        choice = catalog[index - 1][0]
    if not re.fullmatch(r"[A-Za-z0-9._:/@-]+", choice):
        await message.reply("That does not look like a model name.")
        return

    bot.store.update(guild_key(message), model=choice)
    await message.reply(f"Model switched to `{choice}` on {PROVIDERS[provider]['label']}.")


@command("provider", "!provider [gemini|openrouter|groq]", "Pick which free AI provider to use.", group="Models", admin=True)
async def cmd_provider(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    available = bot.available_providers
    current = bot.provider_for(settings)
    if not args:
        lines = []
        for name, info in PROVIDERS.items():
            status = "ready" if name in available else f"needs {info['key_env']}"
            mark = "  **<- current**" if name == current else ""
            lines.append(f"`{name}` - {info['label']} ({status}){mark}")
        await message.reply(
            "Free AI providers:\n"
            + "\n".join(lines)
            + f"\nSwitch with `{bot.prefix}provider groq`. Missing keys go in `.env`.",
            mention_author=False,
        )
        return

    choice = args.strip().lower()
    if choice not in PROVIDERS:
        await message.reply(f"Pick one of: {', '.join(PROVIDERS)}")
        return
    if choice not in available:
        info = PROVIDERS[choice]
        await message.reply(
            f"{info['label']} needs a key. Add `{info['key_env']}` to `.env` "
            f"(free key: {info['signup']}) and restart me."
        )
        return

    model = default_model_for(choice)
    bot.store.update(guild_key(message), provider=choice, model=model)
    bot.clear_guild_memory(guild_key(message))
    await message.reply(
        f"Now using {PROVIDERS[choice]['label']} with `{model}`. History cleared, "
        f"and `{bot.prefix}model` shows the other models."
    )


@command("models", "!models", "List every model your provider offers right now.", group="Models")
async def cmd_models(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    provider = bot.provider_for(settings)
    info = PROVIDERS[provider]
    client = bot.providers.get(provider)
    if client is None:
        await message.reply(
            f"{info['label']} is not set up. Add `{info['key_env']}` to `.env` "
            f"(free key: {info['signup']})."
        )
        return
    async with message.channel.typing():
        try:
            models = await client.list_models()
        except GeminiError as exc:
            await message.reply(f"Could not reach {info['label']}: {exc}")
            return
    if not models:
        await message.reply(f"{info['label']} did not report any text models for this key.")
        return
    shown = models[:60]
    body = "\n".join(("* " if model == settings["model"] else "  ") + model for model in shown)
    more = "" if len(models) <= len(shown) else f"\n...and {len(models) - len(shown)} more"
    await message.reply(
        f"{info['label']} models available ({len(models)}):\n```\n{body}{more}\n```\nSwitch with `{bot.prefix}model <name>`.",
        mention_author=False,
    )


@command("tts", "!tts [auto|edge|gemini|gtts|off]", "Choose the text-to-speech engine.", group="Voice", admin=True)
async def cmd_tts(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    if not args:
        engine = resolve_engine(settings["tts_engine"])
        note = ""
        if edge_tts is None:
            note = "\n\nedge-tts is not installed, so `auto` falls back to gTTS. Run `Run.bat` again to install it."
        await message.reply(
            f"TTS engine: `{settings['tts_engine']}` (resolves to `{engine}`)\n"
            "`edge` - free neural voices, sounds human\n"
            "`gemini` - Gemini 3.8 TTS, most expressive (uses API credits)\n"
            "`gtts` - the classic robot voice\n"
            "`off` - text only" + note
        )
        return
    choice = args.strip().lower()
    if choice in {"gemini", "gemini-tts", "pro"}:
        choice = "gemini"
    if choice not in TTS_ENGINES:
        await message.reply(f"Pick one of: {', '.join(TTS_ENGINES)}")
        return
    if choice == "edge" and edge_tts is None:
        await message.reply("edge-tts is not installed. Run `pip install edge-tts` and restart me.")
        return
    if choice == "gtts" and gTTS is None:
        await message.reply("gTTS is not installed. Run `pip install gTTS` and restart me.")
        return
    bot.store.update(guild_key(message), tts_engine=choice)
    await message.reply(f"TTS engine set to `{choice}`.")


@command("voice", "!voice [alias|name]", "Pick the speaking voice.", group="Voice", admin=True)
async def cmd_voice(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    engine = resolve_engine(settings["tts_engine"])
    if not args:
        if engine == "gemini":
            current = resolve_gemini_voice(settings["gemini_voice"])
            lines = [
                f"`{index}.` `{voice}` - {note}" + ("  **<- current**" if voice == current else "")
                for index, (voice, note) in enumerate(GEMINI_VOICES, 1)
            ]
            await message.reply("Gemini TTS voices:\n" + "\n".join(lines))
            return
        current = resolve_edge_voice(settings["edge_voice"])
        lines = [
            f"`{index}.` `{alias}` ({locale}, {gender}) - {note}"
            + ("  **<- current**" if voice_id == current else "")
            for index, (voice_id, alias, locale, gender, note) in enumerate(EDGE_VOICES, 1)
        ]
        await message.reply(
            f"Current voice: `{current}`\n" + "\n".join(lines)
            + f"\nPick with `{bot.prefix}voice ava`. `{bot.prefix}voices <text>` searches the full Edge catalogue."
        )
        return

    needle = args.strip()
    if engine == "gemini":
        voice = resolve_gemini_voice(needle)
        bot.store.update(guild_key(message), gemini_voice=voice)
        await message.reply(f"Gemini voice set to `{voice}`.")
        return
    voice = resolve_edge_voice(needle)
    bot.store.update(guild_key(message), edge_voice=voice)
    await message.reply(f"Voice set to `{voice}`.")


@command("voices", "!voices [filter]", "Search the installed voice catalogue.", group="Voice")
async def cmd_voices(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    engine = resolve_engine(settings["tts_engine"])
    if engine == "gemini":
        lines = [f"`{voice}` - {note}" for voice, note in GEMINI_VOICES]
        await message.reply("Gemini TTS voices:\n" + "\n".join(lines))
        return
    if args and edge_tts is not None:
        async with message.channel.typing():
            try:
                catalogue = await edge_tts.list_voices()
            except Exception as exc:
                await message.reply(f"Could not fetch the Edge voice list: {exc}")
                return
        needle = args.strip().lower()
        matches = [
            voice for voice in catalogue
            if needle in voice.get("ShortName", "").lower()
            or needle in voice.get("Locale", "").lower()
            or needle in voice.get("Gender", "").lower()
        ]
        if not matches:
            await message.reply(f"No Edge voices matched `{needle}`.")
            return
        shown = matches[:30]
        body = "\n".join(
            f"{voice['ShortName']} ({voice.get('Locale', '?')}, {voice.get('Gender', '?')})"
            for voice in shown
        )
        extra = "" if len(matches) <= len(shown) else f"\n...and {len(matches) - len(shown)} more"
        await message.reply(
            f"Edge voices matching `{needle}` ({len(matches)}):\n```\n{body}{extra}\n```\n"
            f"Pick one with `{bot.prefix}voice <ShortName>`.",
            mention_author=False,
        )
        return
    lines = [
        f"`{alias}` ({locale}, {gender}) - {note}" for _, alias, locale, gender, note in EDGE_VOICES
    ]
    await message.reply("Favourite Edge voices:\n" + "\n".join(lines) + f"\nSearch all of them with `{bot.prefix}voices british`.")


@command("rate", "!rate [+10%|-10%]", "Speak faster or slower.", group="Voice", admin=True, aliases=("speed",))
async def cmd_rate(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    if not args:
        await message.reply(f"Speaking rate is `{settings['tts_rate']}`. Try `{bot.prefix}rate +15%`.")
        return
    value = args.strip().replace(" ", "")
    if not re.fullmatch(r"[+-]?\d{1,3}%", value):
        await message.reply("Use a percentage, for example `!rate -10%` or `!rate +20%`.")
        return
    if not value.startswith(("+", "-")):
        value = "+" + value
    if abs(int(value.rstrip("%"))) > 100:
        await message.reply("Keep the rate within plus or minus 100 percent.")
        return
    bot.store.update(guild_key(message), tts_rate=value)
    await message.reply(f"Speaking rate set to `{value}`.")


@command("pitch", "!pitch [+5Hz|-5Hz]", "Raise or lower the voice pitch.", group="Voice", admin=True)
async def cmd_pitch(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    if not args:
        await message.reply(f"Pitch is `{settings['tts_pitch']}`. Try `{bot.prefix}pitch -10Hz`.")
        return
    value = args.strip().replace(" ", "")
    if not re.fullmatch(r"[+-]?\d{1,3}Hz", value, re.IGNORECASE):
        await message.reply("Use hertz, for example `!pitch -15Hz` or `!pitch +10Hz`.")
        return
    if not value.startswith(("+", "-")):
        value = "+" + value
    if abs(int(value[:-2])) > 100:
        await message.reply("Keep the pitch within plus or minus 100 hertz.")
        return
    bot.store.update(guild_key(message), tts_pitch=value)
    await message.reply(f"Pitch set to `{value}`.")


@command("style", "!style [emotion|reset]", "Gemini TTS acting style, e.g. 'cheerful and quick'.", group="Voice",
         admin=True)
async def cmd_style(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    if not args:
        await message.reply(
            f"TTS style is `{settings['tts_style'] or '-'}`. Works with `!tts gemini`, for example "
            "`!style warm and playful, speaking quickly`."
        )
        return
    if args.strip().lower() in {"reset", "none", "off", "clear"}:
        bot.store.update(guild_key(message), tts_style="")
        await message.reply("TTS style cleared.")
        return
    bot.store.update(guild_key(message), tts_style=args.strip()[:300])
    await message.reply(f"TTS style set to `{args.strip()[:300]}` (Gemini engine only).")


@command("say", "!say <text>", "Speak any text aloud with the current voice.", group="Voice")
async def cmd_say(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    if not args:
        await message.reply(f"Usage: `{bot.prefix}say hello there`")
        return
    player = await _ensure_player(bot, message)
    if player is None:
        return
    if not player.enqueue(args, settings, message.channel, str(message.author)):
        await message.reply("Text-to-speech is off. Turn it on with `!tts edge`.")
        return
    await message.add_reaction("\N{SPEAKER WITH THREE SOUND WAVES}")


@command("joinvc", "!joinvc", "Join your voice channel.", group="Voice")
async def cmd_joinvc(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    player = await _ensure_player(bot, message)
    if player is not None:
        await message.reply(f"Joined {player.vc.channel.mention}. Voice replies are on.")


@command("leavevc", "!leavevc", "Leave the voice channel and clear the queue.", group="Voice", aliases=("stopvc",))
async def cmd_leavevc(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    if message.guild is None or message.guild.voice_client is None:
        await message.reply("I'm not in a voice channel.")
        return
    player = bot.players.get(message.guild.id)
    if player is not None:
        player.cancel()
        await player.disconnect()
    else:
        await message.guild.voice_client.disconnect(force=True)
    await message.reply("Left the voice channel.")


@command("skip", "!skip", "Skip the sentence that is playing right now.", group="Voice", aliases=("next",))
async def cmd_skip(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    player = bot.players.get(message.guild.id) if message.guild else None
    if player is None or not player.vc.is_playing():
        await message.reply("Nothing is playing.")
        return
    player.vc.stop()
    await message.reply("Skipped.")


@command("joke", "!joke", "Tell a random joke.", group="Fun")
async def cmd_joke(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    await message.reply(random.choice(JOKES))


@command("quote", "!quote", "Share a random quote.", group="Fun")
async def cmd_quote(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    await message.reply(random.choice(QUOTES))


@command("goodreads", "!goodreads", "Pull a random quote from Goodreads.", group="Fun")
async def cmd_goodreads(bot: "GeminiDiscordBot", message: discord.Message, args: str, settings: dict) -> None:
    async with message.channel.typing():
        quote = await fetch_goodreads_quote()
    await message.reply(quote)


async def fetch_goodreads_quote() -> str:
    url = "https://www.goodreads.com/quotes"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    return random.choice(QUOTES)
                html = await resp.text()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        print(f"[goodreads] {exc}")
        return random.choice(QUOTES)

    soup = BeautifulSoup(html, "html.parser")
    quotes: list[str] = []
    for node in soup.find_all("div", class_="quoteText"):
        full_text = node.get_text(separator=" ").strip()
        parts = full_text.split("\N{HORIZONTAL BAR}")
        if len(parts) < 2:
            parts = full_text.split("\u2015")
        if len(parts) < 2:
            continue
        quote_text, author = parts[0].strip(), parts[1].strip()
        if quote_text and author:
            quotes.append(f'"{quote_text}" - {author}')
    if not quotes:
        return random.choice(QUOTES)
    return random.choice(quotes)


# --------------------------------------------------------------------------- #
# The bot
# --------------------------------------------------------------------------- #


class GeminiDiscordBot(discord.Client):
    def __init__(
        self,
        config: dict[str, Any],
        store: SettingsStore,
        gemini: GeminiClient,
        providers: Optional[dict[str, Any]] = None,
    ):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        super().__init__(intents=intents)
        self.config = config
        self.store = store
        self.gemini = gemini
        self.providers = dict(providers or {"gemini": gemini})
        self.prefix = config["prefix"]
        self.players: dict[int, GuildPlayer] = {}
        self.memory: dict[tuple[int, int], list[dict[str, Any]]] = {}

    # -- helpers ------------------------------------------------------------ #
    def settings_for(self, message: discord.Message) -> dict[str, Any]:
        return self.store.get(guild_key(message))

    @property
    def available_providers(self) -> list[str]:
        return [name for name in PROVIDERS if name in self.providers]

    def provider_for(self, settings: dict[str, Any]) -> str:
        return resolve_provider(settings.get("provider", "auto"), self.available_providers)

    async def ask(self, settings: dict[str, Any], contents: list[dict]) -> str:
        """Send the conversation to whichever provider this server selected."""
        provider = self.provider_for(settings)
        client = self.providers.get(provider)
        if client is None:
            info = PROVIDERS[provider]
            raise GeminiError(
                f"{info['label']} is not set up - add {info['key_env']} to .env "
                f"(free key: {info['signup']})"
            )
        return await client.generate(
            settings["model"],
            contents,
            self.persona_for(settings),
            float(settings["temperature"]),
            int(settings["max_tokens"]),
        )

    def clear_memory(self, message: discord.Message) -> int:
        key = (guild_key(message), message.author.id)
        history = self.memory.pop(key, [])
        return len(history) // 2

    def clear_guild_memory(self, guild_id: int) -> int:
        keys = [key for key in self.memory if key[0] == guild_id]
        for key in keys:
            self.memory.pop(key, None)
        return len(keys)

    def persona_for(self, settings: dict[str, Any]) -> str:
        persona = settings.get("persona") or self.store.defaults["persona"]
        return persona.replace("{name}", self.user.display_name if self.user else "the bot")

    # -- events ------------------------------------------------------------- #
    async def on_ready(self) -> None:
        engine = resolve_engine(self.store.defaults["tts_engine"])
        print("=" * 62)
        print(f"  Logged in as {self.user} (id {self.user.id})")
        print(f"  Provider   : {self.provider_for(self.store.defaults)}")
        print(f"  Model      : {self.store.defaults['model']}")
        print(f"  TTS engine : {engine}")
        print(f"  Voice      : {self.store.defaults['gemini_voice'] if engine == 'gemini' else self.store.defaults['edge_voice']}")
        print(f"  ffmpeg     : {FFMPEG_PATH or 'NOT FOUND (voice will fail!)'}")
        print(f"  Guilds     : {len(self.guilds)}   Prefix: {self.prefix}")
        print("=" * 62)
        await self.change_presence(activity=discord.Game(name=f"{self.prefix}help"))

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or message.webhook_id:
            return
        if message.content.startswith(self.prefix) and len(message.content) > len(self.prefix):
            await self.dispatch_command(message)
            return

        settings = self.settings_for(message)
        if not self.is_addressed(message, settings):
            return

        user_input = self.strip_bot_names(message)
        images = await self.collect_images(message)
        if not user_input and not images:
            return
        await self.handle_chat(message, user_input, settings, images)

    async def dispatch_command(self, message: discord.Message) -> None:
        body = message.content[len(self.prefix):].strip()
        if not body:
            return
        name, _, args = body.partition(" ")
        spec = COMMANDS.get(name.lower())
        if spec is None:
            return
        settings = self.settings_for(message)
        if spec.admin and not can_configure(self, message):
            await message.reply("Only server managers can change that. Ask an admin, or set `ALLOW_EVERYONE_TO_CONFIGURE=true`.")
            return
        try:
            await spec.handler(self, message, args.strip(), settings)
        except Exception:
            traceback.print_exc()
            await message.reply(f"`{spec.usage or self.prefix + spec.name}` failed. Check the console for details.")

    def is_addressed(self, message: discord.Message, settings: dict[str, Any]) -> bool:
        if self.user and self.user.mentioned_in(message):
            return True
        names = []
        if self.user:
            names.append(self.user.name)
        if message.guild and message.guild.me:
            names.append(message.guild.me.display_name)
        for name in names:
            if len(name) >= 3 and re.search(rf"\b{re.escape(name)}\b", message.content, re.IGNORECASE):
                return True
        special = self.config["special_channel_id"]
        return bool(special and message.channel.id == special)

    def strip_bot_names(self, message: discord.Message) -> str:
        text = message.content
        if self.user:
            text = re.sub(rf"<@!?{self.user.id}>", " ", text)
            text = re.sub(rf"\b{re.escape(self.user.name)}\b", " ", text, flags=re.IGNORECASE)
        if message.guild and message.guild.me and len(message.guild.me.display_name) >= 3:
            text = re.sub(rf"\b{re.escape(message.guild.me.display_name)}\b", " ", text, flags=re.IGNORECASE)
        return text.strip()

    async def collect_images(self, message: discord.Message, limit: int = 4) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for attachment in message.attachments[:limit]:
            content_type = (attachment.content_type or "").lower()
            if not content_type.startswith("image/") or attachment.size > 4_000_000:
                continue
            try:
                data = await attachment.read()
            except discord.HTTPException:
                continue
            parts.append(
                {
                    "inlineData": {
                        "mimeType": content_type.split(";")[0],
                        "data": base64.b64encode(data).decode("ascii"),
                    }
                }
            )
        return parts

    # -- the actual conversation -------------------------------------------- #
    async def handle_chat(
        self,
        message: discord.Message,
        user_input: str,
        settings: dict[str, Any],
        images: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        key = (guild_key(message), message.author.id)
        history = self.memory.get(key, []) if settings["memory_turns"] else []

        parts: list[dict[str, Any]] = []
        if user_input:
            parts.append({"text": user_input})
        parts.extend(images or [])
        if not parts:
            return

        contents = list(history) + [{"role": "user", "parts": parts}]

        async with message.channel.typing():
            try:
                reply = await asyncio.wait_for(
                    self.ask(settings, contents),
                    timeout=GENERATE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                await message.reply(
                    "Gemini took too long to answer, so I stopped waiting. Try again?"
                )
                return
            except GeminiError as exc:
                await message.reply(f"I couldn't get an answer: {exc}")
                return
            except Exception as exc:  # pragma: no cover - unexpected
                traceback.print_exc()
                await message.reply(f"Something went wrong: {exc}")
                return

        for index, chunk in enumerate(split_text(reply)):
            if index == 0:
                await message.reply(chunk, mention_author=False)
            else:
                await message.channel.send(chunk)

        if settings["memory_turns"]:
            # Keep the text of the turn, but never the base64 image data: resending
            # every past screenshot would burn the context window in a few turns.
            remembered = [part for part in parts if "text" in part]
            if any("inlineData" in part for part in parts):
                remembered.append({"text": "[the user attached an image]"})
            history.extend(
                [
                    {"role": "user", "parts": remembered or [{"text": "[image]"}]},
                    {"role": "model", "parts": [{"text": reply}]},
                ]
            )
            max_entries = int(settings["memory_turns"]) * 2
            self.memory[key] = history[-max_entries:]

        player = self.players.get(message.guild.id) if message.guild else None
        if player is not None:
            player.enqueue(reply, settings, message.channel, str(message.author))

    async def close(self) -> None:
        for player in list(self.players.values()):
            player.cancel()
        for vc in list(self.voice_clients):
            try:
                await vc.disconnect(force=True)
            except Exception:
                pass
        seen: set[int] = set()
        for client in list(self.providers.values()) + [self.gemini]:
            if id(client) in seen:
                continue
            seen.add(id(client))
            try:
                await client.close()
            except Exception:
                pass
        await super().close()


# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #


def ensure_env_template() -> None:
    """Create .env so the user has somewhere to paste their keys."""
    if ENV_FILE.exists():
        return
    example = ENV_FILE.with_name(".env.example")
    template = example.read_text(encoding="utf-8") if example.exists() else (
        "# Fill these in, save the file, then start the bot again.\n"
        "DISCORD_TOKEN=\n"
        "GEMINI_API_KEY=\n"
    )
    try:
        ENV_FILE.write_text(template, encoding="utf-8")
    except OSError as exc:
        print(f"[setup] could not create {ENV_FILE}: {exc}")


def open_in_editor(path: Path) -> None:
    """Best effort: pop the file open so the keys can be pasted into it."""
    try:
        if os.name == "nt":
            subprocess.Popen(["notepad.exe", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-a", "TextEdit", str(path)])
        else:
            subprocess.Popen(
                ["xdg-open", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass


def print_credential_help(missing: list[str]) -> None:
    """Explain what to fill in, without asking for anything in the terminal."""
    ensure_env_template()
    print("\n" + "=" * 70)
    print("  Setup needed: " + ", ".join(missing))
    print("=" * 70)
    print("\nOpen this file in any text editor and paste each value after the '=':\n")
    print(f"    {ENV_FILE}\n")
    if "DISCORD_TOKEN" in missing:
        print("  DISCORD_TOKEN    https://discord.com/developers/applications")
        print("                   Your app -> Bot -> Reset Token.")
        print("                   Also enable the Message Content Intent.\n")
    if any(name != "DISCORD_TOKEN" for name in missing):
        print("  AI provider - add at least one, all have a free tier:")
        print("    GEMINI_API_KEY      https://aistudio.google.com/apikey")
        print("    OPENROUTER_API_KEY  https://openrouter.ai/keys")
        print("    GROQ_API_KEY        https://console.groq.com/keys\n")
    print("Save the file and run Run.bat again. Nothing is typed into this")
    print("window, so pasting in the editor is all you need to do.\n")
    open_in_editor(ENV_FILE)


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def print_voice_status(configured: str) -> None:
    """Report the loaded TTS libraries and the engine they add up to."""
    if edge_tts is not None:
        edge_state = f"edge-tts {_package_version('edge-tts')}"
    else:
        edge_state = "edge-tts MISSING" + (f" ({EDGE_IMPORT_ERROR})" if EDGE_IMPORT_ERROR else "")
    if gTTS is not None:
        gtts_state = f"gTTS {_package_version('gTTS')}"
    else:
        gtts_state = "gTTS MISSING" + (f" ({GTT_IMPORT_ERROR})" if GTT_IMPORT_ERROR else "")

    engine = resolve_engine(configured)
    print(f"[voice] {edge_state} | {gtts_state}")
    print(f"[voice] TTS_ENGINE={configured} resolves to '{engine}'")
    if engine == "gtts" and edge_tts is None:
        print("[voice] edge-tts is not installed, so the robotic voice is being used.")
        print("[voice] Fix: python -m pip install -r requirements.txt, then restart.")
    if engine == "none":
        print("[voice] no TTS library is installed - replies will be text only.")
    if not FFMPEG_PATH:
        print("[voice] ffmpeg was not found; voice playback needs it (install it or set FFMPEG_PATH).")


def print_provider_status(available: list[str], active: str, model: str) -> None:
    for name, info in PROVIDERS.items():
        state = "ready" if name in available else f"add {info['key_env']}"
        mark = "->" if name == active else "  "
        print(f"[ai] {mark} {name:<10} {state}")
    print(f"[ai] using {PROVIDERS[active]['label']} with {model}")
    limits = {
        "openrouter": "free models: 20 requests/min, 50/day without credit",
        "groq": "free tier: 30 requests/min, 1000/day per model",
        "gemini": "free tier limits depend on the model",
    }
    print(f"[ai] {limits.get(active, '')}")


def build_config(token: str, api_key: str) -> dict[str, Any]:
    return {
        "discord_token": token,
        "gemini_api_key": api_key,
        "owner_id": env_int("OWNER_ID", 0),
        "prefix": env_str("PREFIX", "!"),
        "special_channel_id": env_int("SPECIAL_CHANNEL_ID", 0),
        "allow_everyone_to_configure": env_bool("ALLOW_EVERYONE_TO_CONFIGURE", False),
        "idle_leave_seconds": env_int("IDLE_LEAVE_SECONDS", 300),
        "ffmpeg": FFMPEG_PATH,
    }


def build_defaults() -> dict[str, Any]:
    configured = env_str("PROVIDER", "auto").lower()
    if configured in PROVIDER_MODELS:
        model_default = env_str(f"{configured.upper()}_MODEL", default_model_for(configured))
    else:
        model_default = env_str("GEMINI_MODEL", DEFAULT_MODEL)
    return {
        "provider": configured,
        "model": model_default,
        "persona": env_str("PERSONA", DEFAULT_PERSONA),
        "temperature": env_float("TEMPERATURE", 0.8),
        "max_tokens": env_int("MAX_TOKENS", 512),
        "memory_turns": env_int("MEMORY_TURNS", 8),
        "tts_engine": env_str("TTS_ENGINE", "auto").lower(),
        "edge_voice": env_str("TTS_VOICE", "en-US-AvaNeural"),
        "gemini_voice": env_str("GEMINI_TTS_VOICE", "Kore"),
        "tts_rate": env_str("TTS_RATE", "+0%"),
        "tts_pitch": env_str("TTS_PITCH", "+0Hz"),
        "tts_style": env_str("TTS_STYLE", ""),
        "tts_lang": env_str("TTS_LANG", "en"),
        "idle_leave_seconds": env_int("IDLE_LEAVE_SECONDS", 300),
        "ffmpeg": FFMPEG_PATH or "ffmpeg",
    }


def main() -> None:
    load_env_file()
    # .env may set FFMPEG_PATH, so resolve it again now that the file is loaded.
    global FFMPEG_PATH
    FFMPEG_PATH = resolve_ffmpeg()
    gemini_key = env_str("GEMINI_API_KEY") or env_str("GOOGLE_API_KEY")
    config = build_config(env_str("DISCORD_TOKEN"), gemini_key)

    available = configured_providers()
    missing: list[str] = []
    if not config["discord_token"]:
        missing.append("DISCORD_TOKEN")
    if not available:
        missing.append("at least one of GEMINI_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY")
    if missing:
        print_credential_help(missing)
        sys.exit(1)

    defaults = build_defaults()
    active = resolve_provider(defaults["provider"], available)
    if not model_looks_like(defaults["model"], active):
        defaults["model"] = default_model_for(active)

    if defaults["tts_engine"] not in TTS_ENGINES:
        print(f"[config] unknown TTS_ENGINE '{defaults['tts_engine']}', using auto")
        defaults["tts_engine"] = "auto"

    print_provider_status(available, active, defaults["model"])
    print_voice_status(defaults["tts_engine"])

    gemini = GeminiClient(gemini_key)
    clients: dict[str, Any] = {}
    if gemini_key:
        clients["gemini"] = gemini
    for name in ("openrouter", "groq"):
        key = env_str(PROVIDERS[name]["key_env"])
        if key:
            clients[name] = OpenAICompatClient(name, key)

    store = SettingsStore(SETTINGS_FILE, defaults)
    bot = GeminiDiscordBot(config, store, gemini, clients)

    print(
        f"\nStarting with {PROVIDERS[active]['label']} / {defaults['model']} "
        f"and TTS engine {resolve_engine(defaults['tts_engine'])}..."
    )
    try:
        bot.run(config["discord_token"])
    except discord.LoginFailure:
        sys.exit("Discord rejected the token. Check DISCORD_TOKEN.")
    except KeyboardInterrupt:
        print("\nStopping.")


if __name__ == "__main__":
    main()
