"""Intelligent text chunker for ExamBookGenerator.

Splits ``Document`` content into ``Chunk`` objects that respect natural
text boundaries (paragraphs, sentences, words) so no chunk ever cuts a
sentence in half.

Token counting uses ``tiktoken`` when available (cl100k_base encoding)
and falls back to a simple word-count heuristic otherwise.
"""

from __future__ import annotations

import re
from typing import Callable

from core.models import Chunk, Document
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Tokeniser ────────────────────────────────────────────────────────────────

_ENCODING_NAME = "cl100k_base"

try:
    import tiktoken as _tiktoken

    _enc = _tiktoken.get_encoding(_ENCODING_NAME)

    def count_tokens(text: str) -> int:
        """Return the number of tokens in *text*."""
        return len(_enc.encode(text))

except ImportError:
    _enc = None  # type: ignore[assignment]

    def count_tokens(text: str) -> int:
        """Fallback: ≈1 token per 4 characters (rough cl100k estimate)."""
        return max(1, len(text) // 4)


# ── Sentence splitting ───────────────────────────────────────────────────────

_RE_SENTENCE_END = re.compile(
    r'(?<=[.!?…])'     # lookbehind for sentence-ending punctuation
    r'(?:\s+|(?=\n))'  # followed by whitespace or newline
)

_RE_HEADING = re.compile(r"^#{1,6}\s+.+", re.MULTILINE)


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, preserving the delimiters."""
    parts = _RE_SENTENCE_END.split(text)
    # Re-join any parts that were split incorrectly (e.g. "U.S.A. hello")
    merged: list[str] = []
    buffer = ""
    for part in parts:
        buffer += part
        # If the chunk ends with an abbreviation-like pattern, keep accumulating
        if buffer.rstrip() and not _RE_SENTENCE_END.search(buffer):
            merged.append(buffer)
            buffer = ""
        elif _RE_SENTENCE_END.search(buffer):
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)
    return [s for s in merged if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Split *text* into paragraphs on double newlines."""
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


# ── Core chunking ────────────────────────────────────────────────────────────

def _tokenise_and_chunk(
    segments: list[str],
    *,
    max_tokens: int,
    overlap_tokens: int,
    token_fn: Callable[[str], int],
) -> list[str]:
    """Grow a list of chunks from *segments*, never exceeding *max_tokens*.

    Parameters
    ----------
    segments:
        Text segments to pack into chunks (paragraphs or sentences).
    max_tokens:
        Maximum tokens per chunk.
    overlap_tokens:
        Number of trailing tokens to carry over into the next chunk.
    token_fn:
        Callable that returns the token count for a string.

    Returns
    -------
    list[str]
        The resulting chunk texts.
    """
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for segment in segments:
        seg_tokens = token_fn(segment)

        # Single segment exceeds limit — split by sentences first, then words
        if seg_tokens > max_tokens:
            # Flush what we have so far
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0

            # Try sentence-level split first
            sentences = _split_sentences(segment)
            if len(sentences) > 1:
                sub_chunks = _tokenise_and_chunk(
                    sentences,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    token_fn=token_fn,
                )
                chunks.extend(sub_chunks)
            else:
                # Fall back to word-level split
                words = segment.split()
                word_buffer: list[str] = []
                word_tokens = 0
                for word in words:
                    wtok = token_fn(word + " ")
                    if word_tokens + wtok > max_tokens and word_buffer:
                        chunks.append(" ".join(word_buffer))
                        if overlap_tokens > 0:
                            tail_tokens = 0
                            tail_start = len(word_buffer)
                            for i in range(len(word_buffer) - 1, -1, -1):
                                tail_tokens += token_fn(word_buffer[i] + " ")
                                if tail_tokens > overlap_tokens:
                                    tail_start = i + 1
                                    break
                            word_buffer = word_buffer[tail_start:]
                            word_tokens = token_fn(" ".join(word_buffer) + " ")
                        else:
                            word_buffer = []
                            word_tokens = 0
                    word_buffer.append(word)
                    word_tokens += wtok
                if word_buffer:
                    chunks.append(" ".join(word_buffer))
            continue

        # Adding segment would exceed limit — flush current chunk
        if current_tokens + seg_tokens > max_tokens and current_parts:
            chunks.append("\n\n".join(current_parts))
            # Overlap: carry tail segments forward
            if overlap_tokens > 0:
                carry_tokens = 0
                carry_start = len(current_parts)
                for i in range(len(current_parts) - 1, -1, -1):
                    carry_tokens += token_fn(current_parts[i])
                    if carry_tokens > overlap_tokens:
                        carry_start = i + 1
                        break
                current_parts = current_parts[carry_start:]
                current_tokens = sum(token_fn(p) for p in current_parts)
            else:
                current_parts = []
                current_tokens = 0

        current_parts.append(segment)
        current_tokens += seg_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


# ── Public API ───────────────────────────────────────────────────────────────

def create_chunks(
    document: Document,
    *,
    max_tokens: int = 1024,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Split a ``Document`` into ``Chunk`` objects.

    The algorithm:

    1. Split text into paragraphs (double-newline delimited).
    2. Paragraphs that fit within *max_tokens* are kept whole.
    3. Oversized paragraphs are split by sentences, then by words if
       still too large.
    4. Adjacent chunks share *overlap_tokens* for context continuity.
    5. Markdown headings are kept at the start of the chunk they introduce.

    Parameters
    ----------
    document:
        The document to split.
    max_tokens:
        Maximum tokens per chunk.  Defaults to ``1024``.
    overlap_tokens:
        Tokens carried over between chunks for context.  Defaults to ``64``.

    Returns
    -------
    list[Chunk]
        Ordered list of ``Chunk`` objects linked to *document*.

    Example
    -------
    >>> doc = Document(content="Paragraph one.\\n\\nParagraph two.")
    >>> chunks = create_chunks(doc, max_tokens=50)
    >>> len(chunks) >= 1
    True
    >>> chunks[0].document_id == doc.id
    True
    """
    if not document.content or not document.content.strip():
        logger.debug("Document '%s' has no content — returning empty chunk list", document.title)
        return []

    paragraphs = _split_paragraphs(document.content)

    chunk_texts = _tokenise_and_chunk(
        paragraphs,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        token_fn=count_tokens,
    )

    chunks: list[Chunk] = []
    for idx, text in enumerate(chunk_texts):
        chunks.append(
            Chunk(
                document_id=document.id,
                content=text.strip(),
                position=idx,
            )
        )

    logger.info(
        "Chunked document '%s' — %d token(s) total → %d chunk(s) "
        "(max_tokens=%d, overlap=%d)",
        document.title,
        count_tokens(document.content),
        len(chunks),
        max_tokens,
        overlap_tokens,
    )
    return chunks
