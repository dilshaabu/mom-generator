# asr.py
from faster_whisper import WhisperModel
import numpy as np
import librosa


class ASR:
    """
    Transcribes audio using faster-whisper (CTranslate2 backend).
    Auto-detects GPU (float16) vs CPU (int8) so the same code runs well on
    a CPU-only laptop or on a Colab GPU runtime.

    Returns segments with text, timestamps, and a detected language per
    chunk, so code-switched meetings (Hinglish/Tanglish/Manglish) still get
    per-chunk language tags for downstream translation.
    """

    CHUNK_DURATION = 60        # seconds per language-detection window
    SAMPLE_RATE    = 16_000    # Whisper requires 16 kHz

    def __init__(self, model_size: str = "small", device: str = "auto",
                 compute_type: str = "auto"):
        if device == "auto":
            device = self._detect_device()
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        print(f"  Loading faster-whisper '{model_size}' "
              f"({compute_type} on {device})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    #  Public API

    def transcribe(self, audio_path: str) -> list[dict]:
        """
        Returns a list of segment dicts:
            {
                "start": float,   # seconds
                "end":   float,
                "text":  str,
                "lang":  str      # ISO-639-1 code, e.g. "hi", "en", "ta"
            }
        """
        audio = self._load_audio(audio_path)
        return self._transcribe_long(audio)

    #  Internal helpers

    @staticmethod
    def _detect_device() -> str:
        try:
            import ctranslate2
            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            return "cpu"

    def _load_audio(self, path: str) -> np.ndarray:
        """Load any audio format, resample to 16 kHz mono float32."""
        audio, _ = librosa.load(path, sr=self.SAMPLE_RATE, mono=True)
        return audio.astype(np.float32)

    def _transcribe_long(self, audio: np.ndarray) -> list[dict]:
        """
        Split audio into fixed windows purely for per-window language
        tagging (useful for meetings that switch languages by topic).
        Each window is transcribed with a single forward pass — no
        redundant separate detect_language() call — and silence is
        skipped by faster-whisper's built-in VAD filter rather than a
        crude amplitude threshold.
        """
        chunk_samples = self.CHUNK_DURATION * self.SAMPLE_RATE
        total_samples = len(audio)
        all_segments  = []
        time_offset   = 0.0
        chunk_idx     = 0

        for start in range(0, total_samples, chunk_samples):
            chunk = audio[start : start + chunk_samples]
            chunk_idx += 1

            segments, info = self.model.transcribe(
                chunk,
                task="transcribe",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=False,
                beam_size=5,
            )

            chunk_lang = info.language
            seg_count = 0
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                all_segments.append({
                    "start": round(seg.start + time_offset, 2),
                    "end":   round(seg.end   + time_offset, 2),
                    "text":  text,
                    "lang":  chunk_lang,
                })
                seg_count += 1

            time_offset += self.CHUNK_DURATION
            print(f"      Chunk {chunk_idx}: lang={chunk_lang}, segments={seg_count}")

        return all_segments
