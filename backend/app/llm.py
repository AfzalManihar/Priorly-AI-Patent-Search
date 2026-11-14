"""LLM stages: claim decomposition, HyDE expansion and grounded explanations.

Backends
  heuristic : rule-based, no network. Decomposition splits on "comprising ...;" lists, HyDE returns the
              element unchanged, explanations are templated from measured ranks and coverage.
  groq      : Groq's OpenAI-compatible chat API. Every call is cached in SQLite by input hash so
              evaluation reruns are deterministic and free (PRD FR-21).
"""
from __future__ import annotations

import hashlib
from email.utils import parsedate_to_datetime
import json
import re
import time

import httpx


class LLMError(RuntimeError):
    """The LLM could not produce a usable answer. Callers fall back and mark the run as degraded."""


_LIST_START = re.compile(r"\b(?:comprising|comprises|including|includes|consisting of|the steps of)\b\s*:?", re.I)


def heuristic_decompose(claim: str) -> list[str]:
    body = claim
    match = _LIST_START.search(claim)
    if match:
        body = claim[match.end():]
    elements: list[str] = []
    for part in re.split(r";\s*", body):
        part = re.sub(r"^\s*(?:and|or)\s+", "", part.strip(), flags=re.I).strip(" .,")
        if len(part) >= 8:
            elements.append(part[0].upper() + part[1:])
    if len(elements) < 2:
        return [claim.strip()]
    return elements[:8]


class HeuristicLLM:
    name = "heuristic"

    def decompose(self, claim: str) -> list[str]:
        return heuristic_decompose(claim)

    def hyde(self, element: str) -> str:
        return element

    def explain(self, elements, covered, passage: str, title: str) -> str | None:
        return None


class GroqLLM:
    name = "groq"

    DECOMPOSE_SYSTEM = (
        "You split a patent claim into 2 to 8 atomic, independently searchable elements. "
        "Keep the claim's own technical wording. Do not add information. "
        'Reply with JSON only: {"elements": ["...", "..."]}.'
    )
    HYDE_SYSTEM = (
        "Write 2 or 3 sentences exactly as they would appear in the description of an earlier patent that "
        "discloses the given claim element. Use concrete technical terms. Do not mention a claim."
    )
    EXPLAIN_SYSTEM = (
        "You explain, in at most 40 words, which claim elements a patent passage supports and which it does "
        "not. Use ONLY the passage text. If the passage does not support an element, say so plainly. "
        "Do not infer anything that is not written in the passage."
    )

    def __init__(self, api_key: str, model: str, base_url: str, store=None, transport=None, sleep=time.sleep):
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set")
        self.model = model
        self.store = store
        self.sleep = sleep
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
            transport=transport,
        )

    # -- plumbing ---------------------------------------------------------------
    def _cached(self, task: str, text: str, produce) -> str:
        key = hashlib.sha256("\x1f".join([task, self.model, text]).encode()).hexdigest()
        if self.store is not None:
            hit = self.store.cache_get(key)
            if hit is not None:
                return hit
        value = produce()
        if self.store is not None:
            self.store.cache_put(key, value)
        return value

    def _chat(self, system: str, user: str, json_mode: bool = False) -> str:
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        for attempt in range(3):
            try:
                response = self.client.post("/chat/completions", json=body)
            except httpx.HTTPError as exc:
                if attempt == 2:
                    raise LLMError("LLM request failed after 3 attempts") from exc
                delay = 1.0 * (2 ** attempt)
            else:
                if response.status_code == 200:
                    try:
                        return response.json()["choices"][0]["message"]["content"].strip()
                    except (KeyError, IndexError, TypeError, ValueError) as exc:
                        raise LLMError("LLM returned an unexpected response") from exc

                status = response.status_code
                detail = ""
                try:
                    detail = str(response.json().get("error", {}).get("message", ""))[:200]
                except (ValueError, AttributeError):
                    pass

                if status not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise LLMError(
                        f"LLM returned status {status}" + (f": {detail}" if detail else "")
                    )

                delay = 5.0 * (2 ** attempt) if status == 429 else 1.0 * (2 ** attempt)

                if status == 429:
                    header_delays = []
                    for name in ("retry-after", "x-ratelimit-reset-tokens",
                                 "x-ratelimit-reset-requests"):
                        raw = response.headers.get(name, "").strip().lower()
                        if not raw:
                            continue
                        try:
                            if raw.endswith("ms"):
                                seconds = float(raw[:-2]) / 1000
                            elif raw.endswith("s"):
                                seconds = float(raw[:-1])
                            elif raw.endswith("m"):
                                seconds = float(raw[:-1]) * 60
                            else:
                                seconds = float(raw)
                            if seconds >= 0:
                                header_delays.append(seconds)
                        except ValueError:
                            try:
                                reset_at = parsedate_to_datetime(raw).timestamp()
                                seconds = max(0.0, reset_at - time.time())
                                header_delays.append(seconds)
                            except (TypeError, ValueError, OverflowError):
                                continue
                    if header_delays:
                        delay = max(header_delays)

                    if delay > 60:
                        raise LLMError(
                            f"Groq rate limit requires waiting {delay:.1f}s; "
                            "retry stopped to avoid another rate-limit request"
                        )

            self.sleep(delay)

        raise LLMError("LLM request failed")  # pragma: no cover

    # -- stages -----------------------------------------------------------------
    def decompose(self, claim: str) -> list[str]:
        def produce() -> str:
            raw = self._chat(self.DECOMPOSE_SYSTEM, claim, json_mode=True)
            try:
                elements = json.loads(raw)["elements"]
            except (ValueError, KeyError, TypeError) as exc:
                raise LLMError("Decomposition was not valid JSON") from exc
            if (not isinstance(elements, list) or not 1 <= len(elements) <= 8
                    or not all(isinstance(e, str) and e.strip() for e in elements)):
                raise LLMError("Decomposition had an invalid shape")
            return json.dumps([e.strip() for e in elements])

        return json.loads(self._cached("decompose", claim, produce))

    def hyde(self, element: str) -> str:
        def produce() -> str:
            text = self._chat(self.HYDE_SYSTEM, element)
            if not text:
                raise LLMError("Empty HyDE passage")
            return text

        return self._cached("hyde", element, produce)

    def explain(self, elements, covered, passage: str, title: str) -> str | None:
        lines = "\n".join(f"E{n}: {text}" for n, text in elements)
        plain = re.sub(r"<[^>]+>", "", passage)
        user = f"Title: {title}\nClaim elements:\n{lines}\n\nPassage:\n{plain}"
        text = self._cached("explain", user, lambda: self._chat(self.EXPLAIN_SYSTEM, user))
        return text or None

    BATCH_EXPLAIN_SYSTEM = (
        "You explain patent passages against claim elements. For each item, explain in at most "
        "40 words which listed elements the passage supports. Use ONLY that item's passage; "
        "do not infer anything not written. Return JSON only in this format: "
        '{"explanations": {"R1": "explanation", "R2": "explanation"}}.'
    )

    def explain_batch(self, items: list[dict]) -> dict[str, str]:
        if not items:
            return {}

        payload = json.dumps({"items": items}, ensure_ascii=False, sort_keys=True)
        user = (
            "Explain each item independently. Keep each response concise and grounded only "
            "in its own passage. Preserve every supplied item ID.\n\n" + payload
        )

        def produce():
            raw = self._chat(self.BATCH_EXPLAIN_SYSTEM, user, json_mode=True)
            try:
                parsed = json.loads(raw)
                explanations = parsed["explanations"]
            except (ValueError, KeyError, TypeError) as exc:
                raise LLMError("LLM returned invalid batch explanations") from exc

            if not isinstance(explanations, dict):
                raise LLMError("LLM returned invalid batch explanations")

            cleaned = {
                str(key): value.strip()
                for key, value in explanations.items()
                if isinstance(value, str) and value.strip()
            }
            if not cleaned:
                raise LLMError("LLM returned no usable batch explanations")

            return json.dumps(cleaned, ensure_ascii=False, sort_keys=True)

        cached = self._cached("explain_batch", user, produce)
        try:
            result = json.loads(cached)
        except (ValueError, TypeError) as exc:
            raise LLMError("Cached batch explanations are invalid") from exc

        if not isinstance(result, dict):
            raise LLMError("Cached batch explanations are invalid")
        return result


def make_llm(settings, store=None):
    if settings.llm_backend == "heuristic":
        return HeuristicLLM()
    if settings.llm_backend == "groq":
        return GroqLLM(settings.groq_api_key, settings.groq_model, settings.groq_base_url, store=store)
    raise ValueError(f"Unknown LLM_BACKEND: {settings.llm_backend}")
