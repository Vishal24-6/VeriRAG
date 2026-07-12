'''Sliding-window text chunker with paragraph-aware boundaries and configurable overlap.
Used to split documents into retrieval-friendly passages.'''

import re
import config


def chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):

    # Normalize spacing
    text = re.sub(r"\n\s*\n", "\n\n", text.strip())

    # Split into paragraphs
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # If adding this paragraph keeps chunk under size
        if len(current_chunk) + len(paragraph) <= chunk_size:

            current_chunk += paragraph + "\n\n"

        else:

            # Save current chunk
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Start new chunk
            current_chunk = paragraph + "\n\n"

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add overlap for better retrieval continuity
    final_chunks = []

    for i in range(len(chunks)):

        chunk = chunks[i]

        if i > 0:
            prev_chunk = chunks[i - 1]
            overlap_text = prev_chunk[-overlap:]
            chunk = overlap_text + chunk

        final_chunks.append(chunk)

    return final_chunks


def get_chunk_stats(chunks):
    """Return basic statistics about a list of chunks for evaluation and debugging."""
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_length": 0,
            "min_chunk_length": 0,
            "max_chunk_length": 0,
        }
    lengths = [len(c) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_chunk_length": sum(lengths) / len(lengths),
        "min_chunk_length": min(lengths),
        "max_chunk_length": max(lengths),
    }