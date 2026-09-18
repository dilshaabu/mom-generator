"""
app.py — Streamlit demo UI for the Multilingual MoM Generator.

Supports two input modes:
  1. Audio recording  -> ASR -> translate -> segment -> generate -> export
  2. Transcript file (.txt / .docx / .pdf)
                      -> language-tag chunks -> translate -> segment -> generate -> export

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
  - Add GROQ_API_KEY under the app's Settings -> Secrets, e.g.:
        GROQ_API_KEY = "your-key-here"
  - Commit your fonts/ folder to the repo so PDF export works out of the box
    for anyone using the deployed app (no manual upload needed there).
"""

import os
import time
import datetime
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Prefer a local .env for dev; fall back to Streamlit Cloud's Secrets Manager
# so a deployed public app never needs the visitor to supply a key.
if not os.environ.get("GROQ_API_KEY"):
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

from asr import ASR
from translator import Translator
from topic_segmenter import TopicSegmenter
from mom_generator import MoMGenerator
from exporter import export_txt, export_pdf

st.set_page_config(page_title="Multilingual MoM Generator", page_icon="📝", layout="centered")
st.title("📝 Multilingual Meeting Minutes Generator")
st.caption("Upload a meeting recording or an existing transcript — get structured, multilingual minutes.")

# ---------------------------------------------------------------------------
# Sidebar — meeting details
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Meeting details")
    title = st.text_input("Title", value="Team Meeting")
    date = st.text_input("Date", value=datetime.date.today().isoformat())
    participants = st.text_input("Participants", placeholder="Alice, Bob, Carol")
    output_lang = st.selectbox(
        "Output language",
        ["English", "Hindi", "Tamil", "Malayalam", "Marathi", "Gujarati",
         "Telugu", "Bengali", "Urdu", "French", "German", "Spanish"],
    )

# ---------------------------------------------------------------------------
# Input mode
# ---------------------------------------------------------------------------
mode = st.radio("Input type", ["Audio recording", "Transcript file"], horizontal=True)

audio_file = None
transcript_file = None
whisper_size = "small"

if mode == "Audio recording":
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg", "flac"])
    whisper_size = st.selectbox(
        "Whisper model size", ["small", "medium", "large-v3"], index=0,
        help="Larger = more accurate but slower. 'small' keeps a public CPU demo responsive.",
    )
else:
    transcript_file = st.file_uploader(
        "Upload transcript", type=["txt", "docx", "pdf"],
        help="Plain text, Word, or PDF — one paragraph or line per speaker turn works best.",
    )

generate_clicked = st.button("Generate Minutes", type="primary")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_transcript_text(uploaded_file) -> str:
    """Extract raw text from an uploaded .txt / .docx / .pdf transcript."""
    suffix = uploaded_file.name.lower().rsplit(".", 1)[-1]

    if suffix == "txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if suffix == "docx":
        import docx
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            doc = docx.Document(tmp_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        finally:
            os.unlink(tmp_path)

    if suffix == "pdf":
        from PyPDF2 import PdfReader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            reader = PdfReader(tmp_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        finally:
            os.unlink(tmp_path)

    raise ValueError(f"Unsupported transcript format: .{suffix}")


def text_to_segments(text: str) -> list[dict]:
    """
    Turn a plain transcript into segment dicts shaped like ASR.transcribe()'s
    output ({start, end, text, lang}), so the rest of the pipeline
    (translator -> segmenter -> generator -> exporter) needs no changes.

    Each paragraph (or line, if the transcript has no blank-line breaks) is
    language-detected independently, so a transcript that switches languages
    partway through still gets routed to translation correctly.
    """
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # deterministic detection

    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    if len(chunks) <= 1:
        chunks = [c.strip() for c in text.split("\n") if c.strip()]

    segments = []
    t = 0.0
    for chunk in chunks:
        try:
            lang = detect(chunk)
        except Exception:
            lang = "en"
        segments.append({"start": t, "end": t + 1.0, "text": chunk, "lang": lang})
        t += 1.0
    return segments


def run_pipeline(segments: list[dict]):
    status = st.empty()
    t0 = time.time()

    status.info("Translating...")
    translator = Translator()
    english_segments = translator.translate_segments(segments)

    status.info("Segmenting into topics...")
    segmenter = TopicSegmenter()
    topics = segmenter.segment(english_segments)

    status.info("Generating minutes...")
    generator = MoMGenerator()
    mom_document = generator.generate(
        topics=topics, title=title, date=date,
        participants=participants or "N/A",
        target_language=output_lang,
    )

    os.makedirs("outputs", exist_ok=True)
    txt_path = "outputs/minutes.txt"
    pdf_path = "outputs/minutes.pdf"
    export_txt(mom_document, txt_path)
    export_pdf(mom_document, pdf_path, language=output_lang)

    status.success(f"Done in {time.time() - t0:.1f}s")
    return mom_document, txt_path, pdf_path


# ---------------------------------------------------------------------------
# Main action
# ---------------------------------------------------------------------------
if generate_clicked:
    if mode == "Audio recording" and not audio_file:
        st.warning("Please upload an audio file first.")
    elif mode == "Transcript file" and not transcript_file:
        st.warning("Please upload a transcript file first.")
    elif not os.environ.get("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY is not set. Add it to a local .env file, "
            "or under Settings → Secrets if this app is deployed on Streamlit Cloud."
        )
    else:
        try:
            if mode == "Audio recording":
                suffix = os.path.splitext(audio_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_file.read())
                    audio_path = tmp.name
                try:
                    with st.spinner("Transcribing audio..."):
                        asr = ASR(model_size=whisper_size)
                        segments = asr.transcribe(audio_path)
                finally:
                    os.unlink(audio_path)
            else:
                with st.spinner("Reading transcript..."):
                    text = read_transcript_text(transcript_file)
                    if not text.strip():
                        st.error("No readable text found in that file.")
                        st.stop()
                    segments = text_to_segments(text)

            mom_document, txt_path, pdf_path = run_pipeline(segments)

            st.subheader("Minutes of Meeting")
            st.markdown(mom_document)

            col1, col2 = st.columns(2)
            with col1:
                with open(txt_path, "rb") as f:
                    st.download_button("Download TXT", f, file_name="minutes.txt")
            with col2:
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF", f, file_name="minutes.pdf")

        except Exception as e:
            st.error(f"Error: {e}")
