"""One real Groq call with the project's own client. Prints the exact error if it fails (the key is never printed).

  python -m eval.check_groq
"""
from __future__ import annotations

from app.config import Settings
from app.llm import GroqLLM, LLMError


def main() -> None:
    s = Settings.from_env()
    key = s.groq_api_key
    print(f"model={s.groq_model}  base_url={s.groq_base_url}")
    print(f"key set={bool(key)}  length={len(key)}  starts with 'gsk_'={key.startswith('gsk_')}")
    llm = GroqLLM(key, s.groq_model, s.groq_base_url)
    claim = "A wireless power system comprising a transmitter coil, a receiver coil, and a controller that adjusts frequency."
    try:
        print("decompose ->", llm.decompose(claim))
        print("hyde      ->", llm.hyde("a controller that adjusts frequency")[:200])
        print("OK: Groq works.")
    except LLMError as exc:
        print("FAILED:", exc)


if __name__ == "__main__":
    main()
