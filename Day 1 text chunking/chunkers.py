import re
from typing import List, Callable


def fixed_size_chunk(text: str, chunk_size: int) -> List[str]:
    """
    Splits the text into fixed-size character chunks.
    Useful as a baseline or when the structure of the text doesn't matter.
    """
    chunks = []
    start = 0

    # Walk through the text in uniform steps
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end

    return chunks


def sentence_chunk(text: str, max_chars: int) -> List[str]:
    """
    Groups sentences together until the chunk reaches the target size.
    Helps keep thoughts intact while still controlling chunk length.
    """
    # Basic sentence splitting using punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""

    for s in sentences:
        # If adding the next sentence pushes us over the limit,
        # store the current chunk and start a new one.
        if len(current) + len(s) > max_chars:
            if current:
                chunks.append(current.strip())
            current = s
        else:
            current += " " + s if current else s

    # Capture any leftover text
    if current:
        chunks.append(current.strip())

    return chunks


def paragraph_chunk(text: str) -> List[str]:
    """
    Splits text by paragraph breaks.
    Works best when the source has clean formatting (docs, articles, etc.).
    """
    # Split on double newlines and drop empty entries
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def overlapping_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Creates overlapping chunks.
    Useful when the boundaries between chunks contain important context.
    """
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())

        # Move forward but keep a portion of the previous chunk
        start = end - overlap

    return chunks


def token_aware_chunk(text: str, tokenizer: Callable, max_tokens: int) -> List[str]:
    """
    Splits text by token count rather than character count.
    Necessary when embedding models impose strict token limits.
    
    `tokenizer` should be a function that:
        - encodes text into token IDs
        - decodes token IDs back into text
    """
    token_ids = tokenizer(text)
    chunks = []
    start = 0

    # Walk through tokens in continuous windows
    while start < len(token_ids):
        end = start + max_tokens
        chunk_tokens = token_ids[start:end]

        # Convert token IDs back to readable text
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text.strip())

        start = end

    return chunks


def recursive_chunk(text: str, target_size: int) -> List[str]:
    """
    Attempts to chunk text using natural boundaries first (paragraphs, sentences).
    Falls back to fixed-size chunks if needed.
    This tends to handle messy or uneven documents better than any single strategy.
    """
    paragraphs = paragraph_chunk(text)
    intermediate = []

    for para in paragraphs:
        # If the paragraph already fits, keep it as-is
        if len(para) <= target_size:
            intermediate.append(para)
            continue

        # Otherwise, break the paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+', para)
        buffer = ""

        for s in sentences:
            # Start a new chunk when the buffer crosses the size limit
            if len(buffer) + len(s) > target_size:
                if buffer:
                    intermediate.append(buffer.strip())
                buffer = s
            else:
                buffer += " " + s if buffer else s

        if buffer:
            intermediate.append(buffer.strip())

    final_chunks = []

    # After sentence-level splitting, some chunks may still be too large.
    # If so, fall back to fixed-size splitting.
    for chunk in intermediate:
        if len(chunk) > target_size * 1.5:
            final_chunks.extend(fixed_size_chunk(chunk, target_size))
        else:
            final_chunks.append(chunk)

    return final_chunks
