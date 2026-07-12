import os
import time
import logging
import numpy as np
from dotenv import load_dotenv
from groq import Groq
import config
from circuit_breaker import llm_breaker

logger = logging.getLogger(__name__)

load_dotenv()

_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Add it to your .env file (local) or Streamlit Cloud secrets."
    )

client = Groq(api_key=_api_key)




def truncate_context(context, max_tokens=config.MAX_CONTEXT_TOKENS):
    """Truncate context to approximately max_tokens (estimate 4 chars per token)."""
    max_chars = max_tokens * 4
    if len(context) <= max_chars:
        return context
    logger.warning(
        f"Context too long ({len(context)} chars), truncating to ~{max_tokens} tokens"
    )
    return context[:max_chars]


# --------------------------------------------------
# Conversation-aware Query Rewriting
# --------------------------------------------------

def rewrite_query(question, history):

    conversation = ""

    for msg in history[-4:]:
        conversation += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""
Rewrite the user's latest question into a standalone search query.

Conversation:
{conversation}

Latest question:
{question}

Rules:
- Replace ALL pronouns (it, they, this, that, these, those) with the specific entity from the conversation
- If the question references something from earlier, include the full entity name
- Output MUST be a complete, specific search query — never vague
- Do NOT output phrases like "more details" or "tell me about it" — always resolve to specific terms
- Return ONLY the rewritten query, nothing else — no explanation, no prefix, no commentary
"""

    response = llm_breaker.call(lambda: client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.LLM_TEMPERATURE,
    ))

    rewritten = response.choices[0].message.content.strip()

    # Validation 1: reject vague or too-short rewrites
    if len(rewritten.split()) < 3:
        return question

    # Validation 2 (Fix 3): reject semantically drifted rewrites
    # Uses the existing bi-encoder singleton — zero extra LLM calls
    try:
        from retriever import get_bi_encoder
        model = get_bi_encoder()
        embs = model.encode([question, rewritten])
        cos_sim = float(
            np.dot(embs[0], embs[1])
            / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8)
        )
        if cos_sim < config.DRIFT_THRESHOLD:
            logger.debug(f"Rewrite rejected (similarity={cos_sim:.3f}): '{rewritten}'")
            return question
        logger.debug(f"Rewrite accepted (similarity={cos_sim:.3f}): '{rewritten}'")
    except Exception:
        return question  # Fail-safe: always prefer original on error

    return rewritten


# --------------------------------------------------
# Context Sufficiency Pre-Check
# --------------------------------------------------

def is_context_sufficient(context, question):
    """Quick LLM check to see if the context can answer the question."""

    # Fast path: if context is empty or trivially short, skip LLM call
    if not context or len(context.strip()) < 50:
        return False

    prompt = f"""Determine if the context below contains information relevant to answering the question. The context does NOT need to contain a complete answer — partial or related information is sufficient.

Context:
{context}

Question:
{question}

Respond with ONLY 'YES' or 'NO'. If the context discusses the same topic as the question, respond 'YES'.
"""

    response = llm_breaker.call(lambda: client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5,
    ))

    answer = response.choices[0].message.content.strip().upper()
    return "YES" in answer


# --------------------------------------------------
# Streaming Answer Generation
# --------------------------------------------------

def query_llm(context, question):

    # Truncate context to prevent token overflow errors
    context = truncate_context(context)

    prompt = f"""
Answer the question using the context provided below.

Context:
{context}

Question: {question}

Instructions:
- Use inline citations like [1], [2] to reference the numbered sources in the context
- Prioritize information from uploaded documents over web results
- If the context contains relevant information, use it to answer thoroughly
- If the answer is not present in the context, say you don't know
- Be concise and accurate
"""

    system_message = config.SYSTEM_PROMPT

    # Retry logic: 1 retry with delay for transient API failures
    last_error = None
    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            response = llm_breaker.call(lambda: client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.LLM_TEMPERATURE,
                stream=True
            ))

            for chunk in response:

                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            return  # Success — exit after streaming completes
        except Exception as e:
            last_error = e
            if attempt < config.LLM_MAX_RETRIES - 1:
                logger.warning(f"LLM API call failed (attempt {attempt + 1}/{config.LLM_MAX_RETRIES}), retrying in {config.LLM_RETRY_DELAY}s: {e}")
                time.sleep(config.LLM_RETRY_DELAY)
            else:
                logger.error(f"LLM API call failed after {config.LLM_MAX_RETRIES} attempts: {e}")
    raise last_error