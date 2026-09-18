# translator.py
"""
Translates all non-English segments (and normalizes mixed-script English
segments) to clean English using Groq. Handles pure languages AND
code-switched speech (Manglish, Tanglish, Hinglish).

Same behavior as the original, but batched: instead of one Groq call per
segment (dozens of sequential round-trips even for a short meeting), every
segment that needs cleanup is sent in groups of up to BATCH_SIZE per call.
This is the single biggest speed win in the pipeline, and it also improves
translation quality because the model sees surrounding conversational
context instead of isolated lines.
"""

import os
import re
from dotenv import load_dotenv
from groq import Groq, RateLimitError
import time

load_dotenv()
_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

_LANG_NAMES = {
    "hi": "Hindi", "ta": "Tamil", "ml": "Malayalam", "mr": "Marathi",
    "gu": "Gujarati", "te": "Telugu", "bn": "Bengali", "ur": "Urdu",
    "fr": "French", "de": "German", "es": "Spanish", "zh": "Chinese",
    "ar": "Arabic", "ja": "Japanese", "ko": "Korean",
}

_MARKER = "|||SEG_{}|||"
_MARKER_RE = re.compile(r"\|\|\|SEG_(\d+)\|\|\|\s*(.*?)(?=\|\|\|SEG_\d+\|\|\||\Z)", re.S)

_SYSTEM_PROMPT = """You are a translation engine.

Each input line is tagged with a SEG marker and a source-language hint in
brackets, e.g. |||SEG_3||| [Hindi] <text>. Translate the text after each
marker into fluent English.

The input may be English, Malayalam, Hindi, Tamil, Marathi, Telugu,
Bengali, Urdu, French, German, Spanish, or any mixture of these with
English (for example, Manglish, Tanglish, Hinglish).

Rules:
- Return exactly one output line per input line, in the same order.
- Keep each SEG marker exactly as-is at the start of its line.
- Return ONLY the English translation after each marker.
- Preserve the original meaning. Do NOT explain, answer, or add notes.
- If a line is unintelligible, return exactly: [Unclear speech]
"""


class Translator:
    MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was decommissioned by Groq on 2026-08-16
    BATCH_SIZE = 40   # segments per call — keeps prompts within token limits
                       # even for long meetings, while still turning a
                       # 200-segment transcript into ~5 calls instead of 200

    def __init__(self):
        self._client = Groq(api_key=_GROQ_KEY) if _GROQ_KEY else None
        if not self._client:
            print("  WARNING: GROQ_API_KEY not set. Translation will be skipped.")

    def translate_segments(self, segments: list[dict]) -> list[dict]:
        cleaned = [dict(s, text=s["text"].strip()) for s in segments if s["text"].strip()]
        if not self._client:
            return [{**s, "original_text": s["text"]} for s in cleaned]

        needs_work = [
            (i, s) for i, s in enumerate(cleaned)
            if not (s.get("lang", "en") in ("en",) and not self._has_mixed_script(s["text"]))
        ]

        translated_map: dict[int, str] = {}
        for batch_start in range(0, len(needs_work), self.BATCH_SIZE):
            batch = needs_work[batch_start: batch_start + self.BATCH_SIZE]
            translated_map.update(self._translate_batch(batch))

        result = []
        for i, seg in enumerate(cleaned):
            if i in translated_map:
                result.append({
                    **seg,
                    "text": translated_map[i],
                    "original_text": seg["text"],
                    "translated": True,
                })
            else:
                result.append({**seg, "original_text": seg["text"]})
        return result

    #  Internal helpers

    def _translate_batch(self, indexed_segments: list[tuple[int, dict]]) -> dict[int, str]:
        lines = []
        for i, seg in indexed_segments:
            lang_name = _LANG_NAMES.get(seg.get("lang", "en"), seg.get("lang", "mixed"))
            lines.append(f"{_MARKER.format(i)} [{lang_name}] {seg['text']}")
        prompt = "Translate these lines:\n\n" + "\n".join(lines)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                )
                return self._parse_response(response.choices[0].message.content)
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait = 5 * (2 ** attempt)
                    print(f"      Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"      Translation error: {e}")
                    return {}
            except Exception as e:
                print(f"      Translation error: {e}")
                return {}
        return {}

    def _parse_response(self, text: str) -> dict[int, str]:
        out = {}
        for match in _MARKER_RE.finditer(text):
            idx, translated = int(match.group(1)), match.group(2).strip()
            if translated:
                out[idx] = translated
        return out

    def _has_mixed_script(self, text: str) -> bool:
        """Detect non-ASCII characters in text."""
        return sum(1 for c in text if ord(c) > 127) > 3
