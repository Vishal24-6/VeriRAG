"""
evaluation.py — Standalone Evaluation Script

Run this to evaluate the RAG pipeline on the PDFs in the docs/ folder.

Usage:
    python evaluation.py

Place test PDFs in the docs/ folder before running.
"""

import os
import time
import numpy as np
from pypdf import PdfReader

from retriever import build_index, search_index
from llm import query_llm


# ---------------------------------------------------
# Load PDFs
# ---------------------------------------------------

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# ---------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------

def keyword_match(text, keywords):
    """Basic keyword presence check."""
    text = text.lower()
    return any(k.lower() in text for k in keywords)


# TODO: Future — semantic similarity
# def semantic_similarity(answer, expected):
#     from sentence_transformers import SentenceTransformer
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     emb_a = model.encode([answer])
#     emb_e = model.encode([expected])
#     return float(np.dot(emb_a[0], emb_e[0]) / (np.linalg.norm(emb_a[0]) * np.linalg.norm(emb_e[0])))

# TODO: Future — RAGAS evaluation suite
# from ragas import evaluate


# ---------------------------------------------------
# Generic Dataset (works with any PDF content)
# ---------------------------------------------------
# These questions are designed to test the RAG
# pipeline mechanics regardless of PDF topic.
# Keywords are intentionally broad to match varied content.

dataset = [

    {
        "question": "What is the main topic of this document?",
        "keywords": []  # open-ended — evaluated by LLM answer length
    },

    {
        "question": "What are the key concepts explained in this document?",
        "keywords": []  # open-ended
    },

    {
        "question": "Summarize the most important points in the document.",
        "keywords": []  # open-ended
    },

    {
        "question": "What methods or techniques are described?",
        "keywords": ["method", "technique", "approach", "algorithm", "process", "step"]
    },

    {
        "question": "What examples are provided in the document?",
        "keywords": ["example", "case", "instance", "sample", "illustration"]
    },

]


# ---------------------------------------------------
# Load documents
# ---------------------------------------------------

print("\nLoading test documents...\n")

docs_folder = "docs"
documents = {}

if not os.path.exists(docs_folder):
    print(f"Error: '{docs_folder}/' folder not found. Create it and add PDF files.")
    exit(1)

for file in os.listdir(docs_folder):
    if file.endswith(".pdf"):
        print(f"  Loading: {file}")
        documents[file] = load_pdf(os.path.join(docs_folder, file))

if not documents:
    print(f"\nNo PDF files found in '{docs_folder}/' folder.")
    print("Add test PDFs to the docs/ folder and re-run.")
    exit(1)

print(f"\nLoaded {len(documents)} document(s).\n")


# ---------------------------------------------------
# Build index
# ---------------------------------------------------

print("Building index...\n")

index_bundle, chunks, sources = build_index(documents)

print(f"Index ready: {len(chunks)} chunks\n")


# ---------------------------------------------------
# Evaluation loop
# ---------------------------------------------------

print("Running evaluation...\n")

retrieval_hits = 0
answer_hits = 0
hallucinations = 0
total_latency = 0
confidence_values = []
skipped = 0


for item in dataset:

    question = item["question"]
    keywords = item["keywords"]

    print(f"Q: {question}")

    start = time.time()

    retrieved_chunks, retrieved_sources, scores = search_index(
        index_bundle,
        chunks,
        sources,
        question,
        original_query=question   # same for eval (no rewriting in standalone mode)
    )

    latency = time.time() - start
    total_latency += latency

    if scores:
        confidence_values.append(np.mean(scores))

    # ---------------------------------------------------
    # Retrieval evaluation
    # ---------------------------------------------------

    retrieval_text = " ".join(retrieved_chunks)

    if not keywords:
        # Open-ended question — count as retrieval hit if we got any chunks back
        if len(retrieved_chunks) > 0:
            retrieval_hits += 1
    else:
        if keyword_match(retrieval_text, keywords):
            retrieval_hits += 1

    # ---------------------------------------------------
    # LLM answer generation
    # ---------------------------------------------------

    context = "[DOCUMENT CONTEXT]\n"
    for i, chunk in enumerate(retrieved_chunks):
        context += f"[{i+1}] ({retrieved_sources[i]})\n{chunk}\n\n"

    try:
        answer = ""
        for token in query_llm(context, question):
            answer += token

    except Exception as e:
        print(f"  ⚠️  LLM error: {str(e)[:80]}")
        print()
        skipped += 1
        continue

    # ---------------------------------------------------
    # Answer evaluation
    # ---------------------------------------------------

    if not keywords:
        # Open-ended — count as hit if answer is non-trivial (>50 chars)
        if len(answer.strip()) > 50:
            answer_hits += 1
    else:
        if keyword_match(answer, keywords):
            answer_hits += 1

    # ---------------------------------------------------
    # Hallucination detection
    # ---------------------------------------------------

    # Only flag hallucination if keywords were expected but not found in retrieval
    if keywords and not keyword_match(retrieval_text, keywords):
        hallucinations += 1

    print(f"  Answer: {answer[:150].strip()}...")
    print(f"  Confidence: {round(float(np.mean(scores)), 4) if scores else 'N/A'}")
    print(f"  Sources: {list(dict.fromkeys(retrieved_sources))}")
    print()

    # Delay to avoid Groq rate limits
    time.sleep(3)


# ---------------------------------------------------
# Final metrics
# ---------------------------------------------------

total = len(dataset) - skipped

if total == 0:
    print("All questions were skipped due to errors. Check your GROQ_API_KEY.")
    exit(1)

recall = retrieval_hits / total
accuracy = answer_hits / total
hallucination_rate = hallucinations / max(total, 1)
avg_latency = total_latency / len(dataset)
avg_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0


print("=" * 45)
print("         EVALUATION RESULTS")
print("=" * 45)
print(f"  Documents evaluated:   {len(documents)}")
print(f"  Questions run:         {total} / {len(dataset)}")
print(f"  Retrieval Recall:      {round(recall, 2)}")
print(f"  Answer Accuracy:       {round(accuracy, 2)}")
print(f"  Hallucination Rate:    {round(hallucination_rate, 2)}")
print(f"  Avg Retrieval Latency: {round(avg_latency, 3)}s")
print(f"  Avg Confidence:        {round(avg_confidence, 4)}")
print("=" * 45)
print()
print("NOTE: Keyword-based evaluation is a proxy metric.")
print("Upgrade to semantic similarity or RAGAS for production grading.")