from chunkers import (
    fixed_size_chunk,
    sentence_chunk,
    paragraph_chunk,
    overlapping_chunk,
    recursive_chunk,
)


def load_sample():
    return """Artificial intelligence is transforming how people interact with software.
Machine learning models can now analyze language, generate text, and understand complex queries.

Large Language Models provide powerful reasoning abilities but are limited by context windows.
Retrieval-Augmented Generation aims to solve this by grounding responses in external documents.

Chunking quality plays a major role in retrieval accuracy. Poor chunking leads to missing context,
weak embeddings, and degraded answers.
"""


def main():
    text = load_sample()

    print("Original length:", len(text))

    fs = fixed_size_chunk(text, 150)
    print("\nFixed-size:", len(fs))

    sc = sentence_chunk(text, 150)
    print("Sentence-based:", len(sc))

    pc = paragraph_chunk(text)
    print("Paragraph-based:", len(pc))

    oc = overlapping_chunk(text, chunk_size=150, overlap=40)
    print("Overlapping:", len(oc))

    rc = recursive_chunk(text, target_size=150)
    print("Recursive:", len(rc))


if __name__ == "__main__":
    main()
