# mom_generator.py
import os
import textwrap
import time
from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()

_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

# Language display-name -> hint used in prompts. Add more as needed.
SUPPORTED_OUTPUT_LANGUAGES = [
    "English", "Hindi", "Tamil", "Malayalam", "Marathi",
    "Gujarati", "Telugu", "Bengali", "Urdu", "French",
    "German", "Spanish",
]


class MoMGenerator:
    MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was decommissioned by Groq on 2026-08-16

    def __init__(self):
        if not _GROQ_KEY:
            print("  WARNING: GROQ_API_KEY not set in .env file.")
            self._client = None
        else:
            self._client = Groq(api_key=_GROQ_KEY)
            print("  Groq client loaded successfully.")

    def generate(self, topics, title, date, participants, target_language="English"):
        """
        target_language: display name from SUPPORTED_OUTPUT_LANGUAGES.
        Header (title/date/participants) stays as literal user input.
        Body (summary/discussions/decisions/actions) is generated directly
        in the target language.
        """
        full_transcript = " ".join(t["text"] for t in topics).strip()

        header = self._build_header(title, date, participants, target_language)
        print("\n========== TRANSCRIPT SENT TO LLM ==========\n")
        print(full_transcript)
        print("\n===========================================\n")

        if self._client:
            body = self._groq_extract_full(full_transcript, target_language)
        else:
            body = self._fallback_extract(full_transcript)

        footer = "\n" + "=" * 60 + "\nEnd of Minutes\n"
        return header + body + footer

    def _groq_extract_full(self, transcript: str, target_language: str) -> str:
        """
        Single Groq call on the full (English) transcript.
        Model writes the structured MoM directly in target_language,
        but still uses the fixed English section labels below as anchors
        so downstream parsing (evaluator, PDF exporter) keeps working.
        """
        words = transcript.split()
        if len(words) > 6000:
            transcript = " ".join(words[:6000])
            print("      Transcript trimmed to 6000 words for token limit.")

        if target_language == "English":
            lang_instruction = "Write the entire document in English."
        else:
            lang_instruction = (
                f"Write the SUMMARY, DISCUSSION POINTS, DECISIONS, and ACTION ITEMS "
                f"content entirely in {target_language}. "
                f"Keep the four section labels themselves in English exactly as shown "
                f"(MEETING SUMMARY:, KEY DISCUSSIONS:, DECISIONS TAKEN:, ACTION ITEMS:) "
                f"so the document structure stays machine-parseable, but everything "
                f"after each label must be in {target_language}."
            )

        prompt = textwrap.dedent(f"""
            You are a professional meeting minutes writer.
            Below is the full transcript of a meeting, already translated to English.

            Generate a single structured Minutes of Meeting document with:

            1. MEETING SUMMARY - 3-5 sentences summarizing the overall meeting
            2. KEY DISCUSSIONS - 5-8 bullet points of main topics discussed
            3. DECISIONS TAKEN - bullet points of all decisions made (write the
               equivalent of "None" in the target language if there are none)
            4. ACTION ITEMS
            - List every action item explicitly mentioned.
            - Mention the owner ONLY if explicitly stated in the transcript.
            - Never guess or invent an owner.
            - Do NOT write "Owner: Not specified".
            - If no owner is mentioned, write only the action item.
            - If there are no action items, write the equivalent of "None" in the target language.

            {lang_instruction}

            Format EXACTLY like this:
            MEETING SUMMARY:
            <summary paragraph>

            KEY DISCUSSIONS:
            - <point>
            - <point>

            DECISIONS TAKEN:
            - <decision>

            ACTION ITEMS:
            - <action>
            - <action> (Owner: <name>)
            Full Transcript:
            {transcript}
        """).strip()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional multilingual meeting minutes "
                                "writer. Be concise, accurate, and professional. "
                                "Only include facts explicitly mentioned in the transcript. "
                                "Do NOT invent or assume any information. "
                                "Do NOT invent participants, decisions, action items, owners, "
                                "deadlines, meeting outcomes, or other details. "
                                "If information is not explicitly mentioned, state that none "
                                "were mentioned instead of guessing. "
                                "Follow the requested output language instruction exactly."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2048,
                )
                return response.choices[0].message.content.strip()

            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait = 5 * (2 ** attempt)  # 5s, 10s, 20s — was 30/60/90s
                    print(f"      Rate limited. Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    print(f"      Groq rate limit exceeded after retries: {e}")
                    return self._fallback_extract(transcript)
            except Exception as e:
                print(f"      Groq error: {e}")
                return self._fallback_extract(transcript)

    def _fallback_extract(self, transcript: str) -> str:
        preview = transcript[:800] + ("..." if len(transcript) > 800 else "")
        return (
            f"MEETING SUMMARY:\n"
            f"- {preview}\n\n"
            f"KEY DISCUSSIONS:\n- See transcript above\n\n"
            f"DECISIONS TAKEN:\n- None extracted\n\n"
            f"ACTION ITEMS:\n- None extracted\n"
        )

    def _build_header(self, title, date, participants, target_language="English"):
        line = "=" * 60
        return (
            f"{line}\n"
            f"MINUTES OF MEETING\n"
            f"{line}\n"
            f"Title        : {title}\n"
            f"Date         : {date}\n"
            f"Participants : {participants}\n"
            f"Language     : {target_language}\n"
            f"{line}\n\n"
        )

    @staticmethod
    def _fmt_time(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
