"""
benchmark.py — Standard Benchmark Evaluation for VeriRAG

Evaluates the retrieval pipeline against a curated QA benchmark with
ground-truth passages and answers. No external downloads needed — the
benchmark dataset is embedded directly.

Metrics computed:
  - Recall@K        : Was the correct passage retrieved in top-K?
  - MRR             : Mean Reciprocal Rank of the correct passage
  - Answer F1       : Token-level F1 between generated and expected answer
  - Answer Accuracy : Exact keyword match rate
  - Hallucination   : Answer contains claims not in retrieved context
  - Avg Latency     : Mean retrieval + generation time

Usage:
    python benchmark.py
"""

import time
import re
import logging
import numpy as np
from collections import Counter

import config
from retriever import build_index, search_index
from llm import query_llm


# ──────────────────────────────────────────────────────────
# Embedded Benchmark Dataset
# ──────────────────────────────────────────────────────────
# Each entry has:
#   - passage: source text (simulates a document chunk)
#   - question: query to ask
#   - answer: expected ground-truth answer
#   - keywords: key terms that must appear in a correct answer

BENCHMARK_PASSAGES = {
    "attention_mechanisms": """
The attention mechanism, first introduced for neural machine translation by Bahdanau et al.
in 2014, allows models to focus on specific parts of the input when producing each part of
the output. In the original sequence-to-sequence architecture, the encoder compressed the
entire input into a single fixed-length context vector, which created an information
bottleneck for long sequences. Attention solved this by computing a weighted sum over all
encoder hidden states, where the weights (attention scores) are learned based on the
relevance of each source position to the current decoder position.

Self-attention, introduced in the Transformer architecture by Vaswani et al. (2017), extends
this concept by allowing each position in a sequence to attend to all other positions within
the same sequence. The Transformer computes attention using three learned projections: Query
(Q), Key (K), and Value (V) matrices. The attention output is computed as:
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V, where d_k is the dimensionality of the
key vectors. The division by sqrt(d_k) prevents the dot products from growing too large,
which would push the softmax into regions with extremely small gradients.

Multi-head attention runs h parallel attention functions (typically h=8 or h=12), each with
different learned projections, allowing the model to attend to information from different
representation subspaces. The outputs are concatenated and linearly projected. This is
crucial because a single attention head can only focus on one type of relationship at a
time — for instance, one head might capture syntactic dependencies while another captures
semantic relationships. The computational complexity of self-attention is O(n^2 * d) where
n is the sequence length, making it expensive for very long sequences. This quadratic
scaling led to research on efficient attention variants like Linformer (O(n)), Performer
(O(n)), and Flash Attention which optimizes memory access patterns.

Cross-attention is used in encoder-decoder architectures where the queries come from the
decoder and the keys/values come from the encoder. This allows the decoder to selectively
focus on relevant parts of the input sequence when generating each output token. In modern
vision-language models like CLIP and Flamingo, cross-attention bridges visual and textual
representations by allowing text tokens to attend to image patch embeddings.
""",

    "retrieval_augmented_generation": """
Retrieval-Augmented Generation (RAG) was formalized by Lewis et al. (2020) at Facebook AI
Research as a method to ground language model outputs in external knowledge. The core
motivation was that parametric-only models (like GPT) store knowledge implicitly in their
weights, leading to hallucination, inability to update knowledge without retraining, and
poor performance on knowledge-intensive tasks. RAG addresses this by combining a non-
parametric retrieval component with a parametric generator.

The original RAG architecture has two variants: RAG-Sequence, which uses the same retrieved
document to generate the entire output sequence, and RAG-Token, which can use different
retrieved documents for each output token. Both variants use a retriever based on Dense
Passage Retrieval (DPR) with FAISS indexing over Wikipedia passages. The retriever uses a
bi-encoder architecture where queries and passages are independently encoded into dense
vectors, and retrieval is performed via Maximum Inner Product Search (MIPS).

A critical challenge in RAG systems is the "lost in the middle" problem, identified by Liu
et al. (2023). When many passages are retrieved and placed in the context, language models
tend to focus on information at the beginning and end of the context window, while largely
ignoring information in the middle positions. This has significant implications for how
retrieved passages should be ordered — placing the most relevant passage first or last
yields better results than placing it in the middle.

Modern RAG pipelines have evolved beyond simple retrieve-then-generate. Advanced techniques
include: (1) Query decomposition — breaking complex questions into sub-queries for multi-hop
reasoning; (2) Iterative retrieval — multiple rounds of retrieval where each round is
informed by previously retrieved context; (3) Reranking — using cross-encoders to rescore
retrieved passages for higher precision; (4) Adaptive retrieval — deciding whether retrieval
is needed at all based on the query type; (5) Self-RAG (Asai et al., 2023) — where the
model learns to retrieve, critique, and generate in a self-reflective manner, outputting
special reflection tokens that indicate whether retrieval is needed and whether the
generated output is supported by the retrieved evidence.

The RAGAS framework (Retrieval Augmented Generation Assessment) provides standardized
metrics for evaluating RAG systems: Faithfulness (is the answer grounded in context?),
Answer Relevancy (does the answer address the question?), Context Precision (are retrieved
passages relevant?), and Context Recall (are all relevant passages retrieved?).
""",

    "vector_databases_and_indexing": """
Vector databases are specialized storage systems optimized for storing, indexing, and
querying high-dimensional vector embeddings. Unlike traditional databases that use B-tree
or hash indexes for exact matching, vector databases use Approximate Nearest Neighbor (ANN)
algorithms to efficiently find vectors similar to a query vector in high-dimensional space.

FAISS (Facebook AI Similarity Search), developed by Meta AI Research, is the most widely
used vector search library. FAISS provides several index types with different trade-offs:
IndexFlatL2 performs exact brute-force search with O(n*d) complexity — accurate but slow
for large collections. IndexIVFFlat uses Inverted File indexing, partitioning vectors into
Voronoi cells using k-means clustering, then only searching the nearest nprobe cells —
reducing search time to O(nprobe * n/k * d) at the cost of some recall. IndexHNSW
(Hierarchical Navigable Small World) builds a multi-layer graph where each node connects
to its nearest neighbors, enabling O(log n) approximate search with very high recall.
Product Quantization (PQ) compresses vectors by splitting them into subvectors and
quantizing each independently, dramatically reducing memory requirements from O(n*d) to
O(n*m*log(k)) where m is the number of subvectors.

ChromaDB is an open-source embedding database designed for AI applications. Unlike FAISS,
which is a library, ChromaDB provides persistence, metadata filtering, and a client-server
architecture. It stores embeddings alongside their source documents and metadata, enabling
hybrid queries that combine vector similarity with metadata filters. Pinecone is a fully
managed cloud vector database that handles scaling, sharding, and replication automatically.

The choice of distance metric significantly affects retrieval quality. L2 (Euclidean)
distance measures the straight-line distance between points and is sensitive to vector
magnitude. Cosine similarity measures the angle between vectors, making it invariant to
magnitude — two vectors pointing in the same direction have cosine similarity 1.0 regardless
of their lengths. Inner product (dot product) combines magnitude and direction information.
For normalized vectors, cosine similarity and inner product are equivalent. Most sentence
embedding models produce normalized vectors, making cosine similarity the standard choice.

Dimensionality reduction techniques like PCA can reduce embedding dimensions (e.g., from
768 to 256) to speed up search at the cost of some information loss. Matryoshka
Representation Learning (MRL) trains models to produce useful embeddings at multiple
dimensionalities simultaneously, allowing users to truncate embeddings without retraining.
""",

    "cross_encoder_vs_bi_encoder": """
The distinction between bi-encoder and cross-encoder architectures is fundamental to
modern information retrieval systems. Bi-encoders (also called dual-encoders or two-tower
models) process the query and document independently through separate encoder passes,
producing fixed-size embedding vectors that can be compared using simple distance metrics.
This architecture enables pre-computation of document embeddings, making retrieval over
millions of documents feasible in milliseconds using ANN indexes.

Cross-encoders, by contrast, process the query and document simultaneously as a single
concatenated input: [CLS] query [SEP] document [SEP]. This allows full cross-attention
between query and document tokens, enabling the model to capture fine-grained lexical and
semantic interactions that bi-encoders miss. For example, the query "apple stock price"
and a document about "Apple Inc. quarterly earnings" would have strong cross-attention
between "stock" and "earnings" — a relationship a bi-encoder might miss because these
words have very different embeddings when encoded independently.

The performance gap between bi-encoders and cross-encoders is well-documented. On the MS
MARCO passage ranking benchmark, cross-encoders consistently outperform bi-encoders by
5-10% in MRR@10. However, cross-encoders are approximately 1000x slower because they
require a forward pass for every (query, document) pair. For a collection of 1 million
documents, a bi-encoder needs 1 forward pass (for the query) plus an ANN search, while a
cross-encoder would need 1 million forward passes.

This leads to the standard two-stage retrieval-reranking pipeline: Stage 1 uses a bi-encoder
to retrieve the top-k candidates (typically k=100 or k=1000) from the full collection.
Stage 2 uses a cross-encoder to rerank these candidates, selecting the final top-n results
(typically n=5 or n=10). This cascade achieves near cross-encoder quality at near bi-encoder
speed.

Knowledge distillation can bridge the gap further. The cross-encoder serves as a "teacher"
that generates soft labels (relevance scores) for query-document pairs. These scores are
used to train a "student" bi-encoder, transferring some of the cross-encoder's fine-grained
understanding into the more efficient architecture. Models like ColBERT take a middle
approach with late interaction — encoding query and document independently but using a
MaxSim operation over all token-level embeddings rather than a single vector comparison,
achieving a balance between quality and efficiency.

Sentence-BERT (SBERT) by Reimers and Gurevych (2019) demonstrated that vanilla BERT
embeddings (using [CLS] token or mean pooling) perform poorly for semantic similarity tasks.
SBERT fine-tunes BERT using siamese and triplet networks to produce embeddings where
semantically similar sentences have high cosine similarity. The all-MiniLM-L6-v2 model is
a popular SBERT variant with 6 layers and 384-dimensional output, offering a good balance
between speed and quality for embedding generation.
""",

    "tokenization_and_text_processing": """
Tokenization — the process of converting raw text into discrete tokens — is the critical
first step in any NLP pipeline and significantly impacts model performance. Three main
approaches exist, each with distinct trade-offs.

Word-level tokenization splits text on whitespace and punctuation. While intuitive, it
creates massive vocabularies (English has 170,000+ words) and cannot handle out-of-
vocabulary (OOV) words — misspellings, neologisms, or technical terms not seen during
training are mapped to a generic [UNK] token, losing all information. Morphological
variants like "run", "running", "runs", "runner" each occupy separate vocabulary entries
despite sharing the same root.

Character-level tokenization uses individual characters as tokens, requiring only a small
vocabulary (~256 for ASCII, ~65K for Unicode). While it eliminates OOV issues, sequences
become very long (a 10-word sentence becomes ~50 characters), making self-attention O(n^2)
cost prohibitive, and the model must learn to compose characters into meaningful units from
scratch without any prior knowledge of word boundaries.

Subword tokenization algorithms like Byte-Pair Encoding (BPE), WordPiece, and SentencePiece
strike a balance. BPE starts with a character-level vocabulary and iteratively merges the
most frequent adjacent pairs. For example, "l" + "o" -> "lo", then "lo" + "w" -> "low".
Common words like "the" remain single tokens, while rare words like "transformerization"
are split into "transform" + "er" + "ization". WordPiece (used by BERT) is similar but
selects merges based on likelihood maximization rather than frequency. SentencePiece
(used by T5, LLaMA) operates directly on raw text without pre-tokenization, treating the
input as a raw byte stream, which makes it language-agnostic.

The vocabulary size is a critical hyperparameter. BERT uses 30,522 WordPiece tokens.
GPT-2 uses 50,257 BPE tokens. LLaMA 2 uses 32,000 SentencePiece tokens. Larger
vocabularies mean shorter sequences (faster processing) but larger embedding matrices
(more parameters). The tokenizer's behavior directly affects token-to-word ratios: English
averages about 1.3 tokens per word with modern tokenizers, but code or non-English text
can reach 3-5 tokens per word, effectively reducing the model's context window.

For retrieval systems, tokenization choices affect BM25 performance. BM25 operates on
word-level tokens and is sensitive to stemming (reducing words to roots), stop word removal,
and case normalization. A query for "running shoes" would miss a document about "jogging
footwear" unless synonyms are handled. This is precisely why hybrid retrieval systems
combine BM25 (which excels at exact matching) with dense retrieval (which captures
semantic similarity through learned embeddings that are invariant to surface-form
differences).
""",

    "evaluation_metrics_for_ir": """
Evaluating information retrieval and question answering systems requires a carefully
chosen set of metrics that capture different aspects of system quality.

Precision@K measures the fraction of retrieved documents in the top-K results that are
relevant: P@K = |relevant docs in top K| / K. For example, if 3 out of 5 retrieved
documents are relevant, P@5 = 0.6. Recall@K measures the fraction of ALL relevant documents
that appear in the top-K: R@K = |relevant docs in top K| / |total relevant docs|. High
precision means few false positives; high recall means few false negatives. There is
typically a precision-recall trade-off: retrieving more documents increases recall but
may decrease precision.

Mean Reciprocal Rank (MRR) evaluates ranking quality by averaging 1/rank of the first
relevant result across queries. If the first relevant result is at position 1, the
reciprocal rank is 1.0; at position 3, it is 0.333. MRR rewards systems that place relevant
results higher. Normalized Discounted Cumulative Gain (NDCG) extends this by considering
graded relevance (not just binary) and applying a logarithmic discount: DCG@K =
sum(rel_i / log2(i+1)) for i=1..K. NDCG normalizes this by the ideal DCG (documents sorted
by true relevance).

For question answering, the standard metrics are Exact Match (EM) and F1 Score. EM gives
credit only when the predicted answer exactly matches the ground truth (after normalization).
Token-level F1 computes precision and recall over individual tokens: precision = |common
tokens| / |predicted tokens|, recall = |common tokens| / |ground truth tokens|, F1 =
2*P*R/(P+R). F1 is more forgiving than EM — a predicted answer "the backpropagation
algorithm computes gradients" would score 0.0 EM but high F1 against the ground truth
"backpropagation algorithm".

BLEU (Bilingual Evaluation Understudy) and ROUGE (Recall-Oriented Understudy for Gisting
Evaluation) are n-gram based metrics. BLEU measures precision of n-grams in the prediction
against references, while ROUGE measures recall. ROUGE-L uses the longest common
subsequence. BERTScore uses BERT embeddings to compute semantic similarity between predicted
and reference tokens, capturing paraphrases that n-gram metrics miss.

For RAG systems specifically, the RAGAS framework defines four key metrics: (1) Faithfulness
measures whether the answer is grounded in the retrieved context — it decomposes the answer
into atomic statements and checks each against the context. (2) Answer Relevancy measures
whether the answer addresses the question using an LLM to generate questions from the answer
and comparing them to the original. (3) Context Precision measures what fraction of
retrieved passages are actually relevant using an LLM judge. (4) Context Recall checks
whether the retrieved context contains all the information needed to answer correctly.

Hallucination detection in RAG is typically measured by checking whether claims in the
generated answer are supported by the retrieved context. A claim is hallucinated if it
contains information not present in any retrieved passage. The hallucination rate = number
of hallucinated answers / total answers.
""",

    "bm25_and_sparse_retrieval": """
BM25 (Best Matching 25) is the most widely used lexical retrieval function, developed by
Robertson et al. in the 1990s as part of the Okapi information retrieval system at City
University London. It is a bag-of-words model that scores documents based on query term
frequency within documents, inverse document frequency across the collection, and document
length normalization.

The BM25 scoring formula for a query Q containing terms q1, ..., qn against a document D is:
score(D, Q) = sum_i [ IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1*(1 - b + b*|D|/avgdl)) ]
where f(qi,D) is the term frequency of qi in D, |D| is the document length, avgdl is the
average document length, k1 controls term frequency saturation (typically 1.2-2.0), and b
controls length normalization (typically 0.75). IDF is computed as:
IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1) where N is the total number of
documents and n(qi) is the number of documents containing qi.

The parameter k1 controls how quickly the term frequency contribution saturates. With k1=0,
BM25 becomes a binary model (only presence/absence matters). With k1=infinity, term
frequency contribution grows linearly without bound. The typical value k1=1.2 provides
diminishing returns — the first occurrence of a term contributes significantly, but
additional occurrences have progressively less impact. The parameter b controls document
length normalization: b=0 means no normalization (long documents aren't penalized), b=1
means full normalization (scoring is relative to document length). The typical value b=0.75
partially normalizes for length.

BM25 has several known limitations. It cannot capture synonyms or paraphrases — "automobile"
and "car" are treated as completely different terms. It ignores word order — "dog bites man"
and "man bites dog" receive identical scores. It treats each query term independently,
missing multi-word expressions. Term frequency statistics are computed at the document level
only, without considering the position of matches within the document.

Despite these limitations, BM25 remains remarkably competitive. The BEIR benchmark (Thakur
et al., 2021) showed that BM25 outperforms many learned dense retrieval models on
out-of-domain datasets, suggesting that lexical matching provides a robust signal that is
complementary to semantic matching. This is why hybrid retrieval — combining BM25 with
dense retrieval using Reciprocal Rank Fusion — has become the standard approach in modern
retrieval systems, consistently outperforming either method alone.
""",

    "llm_prompting_and_generation": """
Large Language Models generate text through autoregressive next-token prediction: given a
sequence of tokens, the model predicts a probability distribution over the vocabulary for
the next token, samples from this distribution, appends the sampled token, and repeats.
The quality and behavior of generation are controlled by several decoding parameters.

Temperature (T) controls the sharpness of the probability distribution. The logits z are
transformed as: p_i = exp(z_i/T) / sum(exp(z_j/T)). Temperature=0 is greedy decoding
(always picks the highest-probability token), producing deterministic but potentially
repetitive output. Temperature=1.0 samples from the unmodified distribution.
Temperature>1.0 flattens the distribution (more random), Temperature<1.0 sharpens it
(more focused). For factual QA tasks, low temperature (0.1-0.3) is preferred to minimize
hallucination and maximize consistency.

Top-K sampling restricts the next token to the K most probable tokens, setting all other
probabilities to zero and renormalizing. Top-K=1 is equivalent to greedy decoding. Top-K=50
(GPT-2 default) allows moderate diversity. However, Top-K is fragile: K=50 works well when
the distribution is peaked but may still include very unlikely tokens when many tokens have
similar probabilities.

Top-P (nucleus sampling, Holtzman et al. 2019) dynamically determines the cutoff by
selecting the smallest set of tokens whose cumulative probability exceeds P. For example,
Top-P=0.9 includes tokens until their combined probability reaches 90%. This adapts to
the distribution shape: when the model is confident, only 2-3 tokens might be included;
when uncertain, dozens might be.

Prompting strategies significantly affect LLM performance. Zero-shot prompting provides only
the task description. Few-shot prompting includes example input-output pairs, enabling
in-context learning without weight updates. Chain-of-Thought (CoT) prompting, introduced by
Wei et al. (2022), instructs the model to "think step by step", dramatically improving
performance on reasoning tasks. For a math problem, instead of directly outputting "42", the
model produces intermediate reasoning steps.

System prompts establish persistent behavioral constraints for the model. In RAG systems,
the system prompt typically instructs the model to: (1) answer only from provided context,
(2) cite sources using [1], [2] notation, (3) acknowledge when the context is insufficient
rather than fabricating information, and (4) maintain a specific tone or persona. The
effectiveness of system prompts varies by model — some models (like GPT-4, Claude) strongly
follow system instructions, while others may drift from instructions during long
conversations, a phenomenon called "instruction forgetting".

Structured output generation using JSON mode or function calling ensures the model output
conforms to a specific schema, which is critical for downstream processing in pipelines.
""",
}


BENCHMARK_QA = [
    # --- Multi-hop reasoning ---
    {
        "question": "Why does the original sequence-to-sequence model struggle with long sequences, and how does self-attention solve this?",
        "answer": "The original seq2seq model compresses the entire input into a single fixed-length context vector creating an information bottleneck. Self-attention solves this by allowing each position to attend to all other positions via Query Key Value projections with scaled dot product attention.",
        "keywords": ["bottleneck", "fixed-length", "query", "key", "value"],
        "relevant_passage": "attention_mechanisms",
    },
    {
        "question": "What is the computational complexity of self-attention and what alternatives exist to address it?",
        "answer": "Self-attention has O(n^2 * d) complexity where n is sequence length. Efficient alternatives include Linformer and Performer with O(n) complexity, and Flash Attention which optimizes memory access patterns.",
        "keywords": ["n^2", "linformer", "performer"],
        "relevant_passage": "attention_mechanisms",
    },
    # --- Comparison / nuanced ---
    {
        "question": "How does RAG-Sequence differ from RAG-Token in the original RAG architecture?",
        "answer": "RAG-Sequence uses the same retrieved document to generate the entire output sequence, while RAG-Token can use different retrieved documents for each output token.",
        "keywords": ["rag-sequence", "rag-token", "entire", "each"],
        "relevant_passage": "retrieval_augmented_generation",
    },
    {
        "question": "What is the lost in the middle problem in RAG systems and what does it imply?",
        "answer": "Identified by Liu et al. 2023, language models focus on information at the beginning and end of the context window while ignoring the middle. The most relevant passage should be placed first or last for best results.",
        "keywords": ["middle", "beginning", "end", "ignore"],
        "relevant_passage": "retrieval_augmented_generation",
    },
    {
        "question": "What is Self-RAG and how does it improve upon standard RAG?",
        "answer": "Self-RAG by Asai et al. 2023 is where the model learns to retrieve, critique, and generate in a self-reflective manner, outputting special reflection tokens that indicate whether retrieval is needed and whether the generated output is supported by evidence.",
        "keywords": ["self-reflective", "reflection", "critique"],
        "relevant_passage": "retrieval_augmented_generation",
    },
    # --- Technical depth ---
    {
        "question": "Explain the difference between IndexFlatL2, IndexIVFFlat, and IndexHNSW in FAISS",
        "answer": "IndexFlatL2 performs exact brute-force search with O(n*d) complexity. IndexIVFFlat uses Inverted File indexing with Voronoi cells via k-means to reduce search space. IndexHNSW builds a multi-layer navigable small world graph enabling O(log n) approximate search with high recall.",
        "keywords": ["brute-force", "voronoi", "graph", "log n"],
        "relevant_passage": "vector_databases_and_indexing",
    },
    {
        "question": "What is Product Quantization and how does it reduce memory in vector search?",
        "answer": "Product Quantization compresses vectors by splitting them into subvectors and quantizing each independently, reducing memory from O(n*d) to O(n*m*log(k)) where m is subvectors count.",
        "keywords": ["subvector", "quantiz", "compress"],
        "relevant_passage": "vector_databases_and_indexing",
    },
    {
        "question": "Why do bi-encoders miss fine-grained interactions that cross-encoders capture?",
        "answer": "Bi-encoders encode query and document independently so they cannot model cross-attention between query and document tokens. For example, the relationship between 'stock' in a query and 'earnings' in a document requires reading them together, which only a cross-encoder with its concatenated input and full cross-attention can capture.",
        "keywords": ["independently", "cross-attention", "concatenat"],
        "relevant_passage": "cross_encoder_vs_bi_encoder",
    },
    {
        "question": "What is knowledge distillation in the context of retrieval, and how does ColBERT differ from standard bi-encoders?",
        "answer": "Knowledge distillation uses a cross-encoder as teacher to generate soft labels for training a student bi-encoder, transferring fine-grained understanding. ColBERT uses late interaction, encoding query and document independently but comparing all token-level embeddings via MaxSim rather than a single vector.",
        "keywords": ["teacher", "student", "colbert", "maxsim", "late interaction"],
        "relevant_passage": "cross_encoder_vs_bi_encoder",
    },
    # --- NLP fundamentals with depth ---
    {
        "question": "How does BPE tokenization handle rare words, and what advantage does SentencePiece have over BPE?",
        "answer": "BPE starts with character-level vocabulary and iteratively merges frequent adjacent pairs so rare words like transformerization are split into transform+er+ization. SentencePiece operates directly on raw byte stream without pre-tokenization, making it language-agnostic unlike BPE which requires pre-tokenization.",
        "keywords": ["merges", "character", "byte", "language-agnostic"],
        "relevant_passage": "tokenization_and_text_processing",
    },
    {
        "question": "How does tokenizer vocabulary size affect the model's context window in practice?",
        "answer": "Larger vocabularies produce shorter token sequences allowing more text in the context window. English averages 1.3 tokens per word but code or non-English text can reach 3-5 tokens per word, effectively reducing the usable context window for those inputs.",
        "keywords": ["1.3", "3-5 tokens", "context window"],
        "relevant_passage": "tokenization_and_text_processing",
    },
    # --- Evaluation knowledge ---
    {
        "question": "What is the difference between NDCG and MRR, and when would you prefer one over the other?",
        "answer": "MRR considers only the rank of the first relevant result and uses binary relevance. NDCG considers graded relevance of all results with logarithmic discount. Use MRR when only one result matters, NDCG when you need to evaluate the full ranked list quality with varying relevance levels.",
        "keywords": ["graded", "logarithmic", "first relevant", "binary"],
        "relevant_passage": "evaluation_metrics_for_ir",
    },
    {
        "question": "What are the four metrics in the RAGAS evaluation framework for RAG systems?",
        "answer": "Faithfulness measures whether the answer is grounded in retrieved context. Answer Relevancy measures whether the answer addresses the question. Context Precision measures what fraction of retrieved passages are relevant. Context Recall checks whether context contains all information needed to answer correctly.",
        "keywords": ["faithfulness", "relevancy", "context precision", "context recall"],
        "relevant_passage": "evaluation_metrics_for_ir",
    },
    {
        "question": "Why does F1 score give a more useful signal than Exact Match for evaluating question answering systems?",
        "answer": "Exact Match gives credit only when predicted answer exactly matches ground truth after normalization. F1 computes token-level precision and recall so a verbose but correct answer scores high on F1 but zero on EM. F1 is more forgiving of paraphrasing and additional context.",
        "keywords": ["token-level", "forgiving", "paraphras"],
        "relevant_passage": "evaluation_metrics_for_ir",
    },
    # --- BM25 deep dive ---
    {
        "question": "What do the BM25 parameters k1 and b control, and what happens at their extreme values?",
        "answer": "k1 controls term frequency saturation: k1=0 makes it binary (only presence matters), k1=infinity gives linear TF growth, typical value 1.2 gives diminishing returns. b controls document length normalization: b=0 means no normalization, b=1 means full normalization, typical 0.75 partially normalizes.",
        "keywords": ["saturation", "k1", "b=0", "b=1", "diminishing"],
        "relevant_passage": "bm25_and_sparse_retrieval",
    },
    {
        "question": "Despite its limitations, why does BM25 remain competitive against learned dense retrieval models?",
        "answer": "The BEIR benchmark by Thakur et al. showed BM25 outperforms many dense retrieval models on out-of-domain datasets because lexical matching provides a robust signal complementary to semantic matching. This is why hybrid retrieval combining BM25 with dense retrieval via RRF consistently outperforms either method alone.",
        "keywords": ["beir", "out-of-domain", "complementary", "hybrid"],
        "relevant_passage": "bm25_and_sparse_retrieval",
    },
    # --- LLM generation ---
    {
        "question": "How does temperature affect LLM text generation, and what value is recommended for factual QA?",
        "answer": "Temperature controls probability distribution sharpness. Temperature=0 is greedy decoding producing deterministic output. Temperature>1 flattens distribution making output more random. Temperature<1 sharpens it for more focused output. For factual QA, low temperature 0.1-0.3 is preferred to minimize hallucination.",
        "keywords": ["greedy", "flatten", "sharpen", "0.1", "hallucination"],
        "relevant_passage": "llm_prompting_and_generation",
    },
    {
        "question": "What is the difference between Top-K and Top-P (nucleus) sampling?",
        "answer": "Top-K restricts sampling to the K most probable tokens regardless of distribution shape. Top-P dynamically selects the smallest set of tokens whose cumulative probability exceeds P, adapting to the distribution: including few tokens when confident and many when uncertain.",
        "keywords": ["cumulative", "dynamic", "adapts", "confident", "uncertain"],
        "relevant_passage": "llm_prompting_and_generation",
    },
    {
        "question": "What is Chain-of-Thought prompting and why does it improve reasoning performance?",
        "answer": "Chain-of-Thought prompting introduced by Wei et al. 2022 instructs the model to think step by step, producing intermediate reasoning steps instead of directly outputting the final answer. This dramatically improves performance on reasoning tasks by decomposing complex problems.",
        "keywords": ["step by step", "intermediate", "reasoning", "wei"],
        "relevant_passage": "llm_prompting_and_generation",
    },
    {
        "question": "What is instruction forgetting in LLMs and why is it relevant for RAG system prompts?",
        "answer": "Instruction forgetting is when models drift from system prompt instructions during long conversations. In RAG systems the system prompt instructs the model to answer only from context and cite sources, so instruction forgetting can cause hallucination as the model reverts to generating from its parametric knowledge.",
        "keywords": ["drift", "long conversation", "system prompt", "parametric"],
        "relevant_passage": "llm_prompting_and_generation",
    },
]


# ──────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────

def _normalize(text):
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_f1(prediction, ground_truth):
    """Token-level F1 score between prediction and ground truth."""
    pred_tokens = _normalize(prediction).split()
    truth_tokens = _normalize(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)


def keyword_match(text, keywords):
    """Check if any keywords appear in text."""
    text = text.lower()
    return any(k.lower() in text for k in keywords)


# ──────────────────────────────────────────────────────────
# Run Benchmark
# ──────────────────────────────────────────────────────────

def run_benchmark():
    # Suppress noisy HTTP debug logs from httpx/httpcore
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("retriever").setLevel(logging.WARNING)
    logging.getLogger("models").setLevel(logging.WARNING)

    print("\n" + "=" * 55)
    print("   VeriRAG BENCHMARK EVALUATION")
    print("   Dataset: 8 passages, 20 questions (embedded)")
    print("=" * 55)

    # Build index from benchmark passages
    print("\nBuilding index from benchmark passages...")
    index_bundle, chunks, sources = build_index(BENCHMARK_PASSAGES)
    print(f"Index ready: {len(chunks)} chunks from {len(BENCHMARK_PASSAGES)} passages\n")

    # Metrics accumulators
    recall_hits = 0
    reciprocal_ranks = []
    f1_scores = []
    keyword_hits = 0
    hallucinations = 0
    latencies = []
    skipped = 0

    for i, qa in enumerate(BENCHMARK_QA):
        question = qa["question"]
        expected = qa["answer"]
        keywords = qa["keywords"]
        relevant = qa["relevant_passage"]

        print(f"[{i+1}/{len(BENCHMARK_QA)}] Q: {question}")

        start = time.perf_counter()

        # Retrieve
        retrieved_chunks, retrieved_sources, scores = search_index(
            index_bundle, chunks, sources, question, original_query=question
        )

        retrieval_time = time.perf_counter() - start

        # --- Recall@K: Check if the relevant passage was retrieved ---
        relevant_text = BENCHMARK_PASSAGES[relevant].strip().lower()
        found_rank = None

        for rank, chunk in enumerate(retrieved_chunks):
            # Check if the chunk overlaps significantly with the relevant passage
            overlap = sum(1 for word in chunk.lower().split()
                         if word in relevant_text.split())
            if overlap > 10:  # At least 10 words overlap
                found_rank = rank + 1
                break

        if found_rank is not None:
            recall_hits += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

        # --- Generate answer ---
        context = "\n".join(f"[{j+1}] ({retrieved_sources[j]})\n{c}"
                           for j, c in enumerate(retrieved_chunks))

        try:
            answer = "".join(query_llm(context, question))
        except Exception as e:
            print(f"  [!] LLM error: {str(e)[:80]}")
            skipped += 1
            print()
            time.sleep(3)
            continue

        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        # --- F1 Score ---
        f1 = compute_f1(answer, expected)
        f1_scores.append(f1)

        # --- Keyword accuracy ---
        if keyword_match(answer, keywords):
            keyword_hits += 1

        # --- Hallucination check ---
        retrieval_text = " ".join(retrieved_chunks).lower()
        if keywords and not keyword_match(retrieval_text, keywords):
            if keyword_match(answer, keywords):
                hallucinations += 1

        # --- Print ---
        confidence = round(float(max(scores)), 3) if scores else 0.0
        print(f"  Answer:     {answer[:120].strip()}...")
        print(f"  F1 Score:   {f1:.3f}")
        print(f"  Confidence: {confidence}")
        retrieved_status = "[OK] Correct passage" if found_rank else "[MISS] Not found"
        rank_info = f" (rank {found_rank})" if found_rank else ""
        print(f"  Retrieved:  {retrieved_status}{rank_info}")
        print(f"  Latency:    {elapsed:.2f}s")
        print()

        # Rate limit delay
        time.sleep(3)

    # ──────────────────────────────────────────────────────
    # Final Results
    # ──────────────────────────────────────────────────────
    total = len(BENCHMARK_QA) - skipped

    if total == 0:
        print("All questions skipped. Check your GROQ_API_KEY.")
        return

    recall = recall_hits / total
    mrr = np.mean(reciprocal_ranks[:total]) if reciprocal_ranks else 0.0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    accuracy = keyword_hits / total
    hall_rate = hallucinations / max(total, 1)
    avg_latency = np.mean(latencies) if latencies else 0.0

    print("=" * 55)
    print("         BENCHMARK RESULTS")
    print("=" * 55)
    print(f"  Passages:             {len(BENCHMARK_PASSAGES)}")
    print(f"  Questions:            {total} / {len(BENCHMARK_QA)}")
    print(f"  -----------------------------------")
    print(f"  Recall@5:             {recall:.2%}")
    print(f"  MRR (Mean Recip Rank):{mrr:.4f}")
    print(f"  Answer F1 Score:      {avg_f1:.4f}")
    print(f"  Keyword Accuracy:     {accuracy:.2%}")
    print(f"  Hallucination Rate:   {hall_rate:.2%}")
    print(f"  Avg Latency:          {avg_latency:.2f}s")
    print("=" * 55)
    print()
    print("Benchmark dataset: 8 technical passages, 20 complex QA pairs")
    print("Question types: multi-hop, comparison, technical depth, synthesis")
    print("Metrics: Recall@K, MRR, token-level F1, keyword accuracy")


if __name__ == "__main__":
    run_benchmark()
