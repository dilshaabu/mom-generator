# topic_segmenter.py
"""
Splits a list of translated segments into topic-coherent blocks.
"""

import re
from nltk.tokenize import TextTilingTokenizer
import nltk

# Download required NLTK data silently
for pkg in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)


class TopicSegmenter:
    """
    Splits translated segments into topic blocks.
    Falls back to simple time-window splitting if TextTiling fails
    (e.g., for very short transcripts).
    """

    MIN_WORDS_FOR_TILING = 200   # TextTiling needs reasonable text length
    FALLBACK_WINDOW_MINS = 5     # minutes per fallback chunk

    def segment(self, segments: list[dict]) -> list[dict]:
        """
        Input : list of translated segment dicts {start, end, text, lang, ...}
        Output: list of topic dicts:
            {
                "topic_index": int,
                "start": float,
                "end":   float,
                "text":  str,           # full topic text
                "segment_count": int,
            }
        """
        if not segments:
            return []

        full_text  = " ".join(s["text"] for s in segments)
        word_count = len(full_text.split())

        if word_count >= self.MIN_WORDS_FOR_TILING:
            return self._tile_topics(segments, full_text)
        else:
            # Short meeting — treat as single topic
            return [self._make_topic(0, segments)]

    #  TextTiling-based segmentation

    def _tile_topics(self, segments: list[dict], full_text: str) -> list[dict]:
        try:
            tt = TextTilingTokenizer(w=20, k=10)

            # TextTiling needs double-newlines between pseudo-sentences
            tiled_text = self._prepare_for_tiling(full_text)
            tiles      = tt.tokenize(tiled_text)

            return self._map_tiles_to_segments(tiles, segments)

        except Exception as e:
            print(f"      TextTiling failed ({e}), using fallback chunking.")
            return self._fallback_chunk(segments)

    def _prepare_for_tiling(self, text: str) -> str:
        """
        TextTiling expects text split into pseudo-sentences separated by \n\n.
        We split on sentence boundaries.
        """
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return "\n\n".join(sentences)

    def _map_tiles_to_segments(
        self,
        tiles: list[str],
        segments: list[dict]
    ) -> list[dict]:
        """
        Map TextTiling output back to original segments by text matching.
        """
        topics     = []
        seg_idx    = 0
        all_segs   = segments[:]

        for i, tile in enumerate(tiles):
            tile_words   = tile.split()
            tile_len     = len(tile_words)
            covered      = 0
            topic_segs   = []

            while seg_idx < len(all_segs) and covered < tile_len:
                seg = all_segs[seg_idx]
                topic_segs.append(seg)
                covered  += len(seg["text"].split())
                seg_idx  += 1

            if topic_segs:
                topics.append(self._make_topic(i, topic_segs))

        # Catch any remaining segments in a final topic
        if seg_idx < len(all_segs):
            remaining = all_segs[seg_idx:]
            last_idx  = len(topics)
            if topics:
                # Merge into last topic
                last = topics[-1]
                last["text"]          += " " + " ".join(s["text"] for s in remaining)
                last["end"]            = remaining[-1]["end"]
                last["segment_count"] += len(remaining)
            else:
                topics.append(self._make_topic(last_idx, remaining))

        return topics if topics else [self._make_topic(0, segments)]


    #  Fallback: time-window chunking


    def _fallback_chunk(self, segments: list[dict]) -> list[dict]:
        """Split by fixed time windows when TextTiling can't run."""
        window_secs = self.FALLBACK_WINDOW_MINS * 60
        topics      = []
        bucket      = []
        topic_idx   = 0
        bucket_start = segments[0]["start"] if segments else 0

        for seg in segments:
            if seg["start"] - bucket_start >= window_secs and bucket:
                topics.append(self._make_topic(topic_idx, bucket))
                topic_idx   += 1
                bucket       = []
                bucket_start = seg["start"]
            bucket.append(seg)

        if bucket:
            topics.append(self._make_topic(topic_idx, bucket))

        return topics
    #  Helpers

    def _make_topic(self, idx: int, segs: list[dict]) -> dict:
        return {
            "topic_index":   idx,
            "start":         segs[0]["start"],
            "end":           segs[-1]["end"],
            "text":          " ".join(s["text"] for s in segs),
            "segment_count": len(segs),
        }

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"
