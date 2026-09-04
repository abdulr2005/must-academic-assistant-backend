import requests

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from .config import RAG_API_URL
from .llm import llm
from .prompts import SYSTEM_PROMPT, build_turn_prompt


def retrieve_context(question: str, top_k: int = 3) -> dict:
    payload = {
        "question": question,
        "top_k": top_k,
    }

    try:
        response = requests.post(
            RAG_API_URL,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"RAG API request failed: {exc}"
        ) from exc

    data = response.json()
    results = data.get("results", [])

    context = []
    sources = []

    for item in results:
        text = item.get("text", "").strip()
        metadata = item.get("metadata") or {}

        if not text:
            continue

        chunk_id = metadata.get("chunk_id")

        chunk = {
            "chunk_id": chunk_id,
            "doc_type": metadata.get("doc_type"),
            "major": metadata.get("major"),
            "semester": metadata.get("semester"),
            "confidence": metadata.get("confidence"),
            "text": text,
        }

        context.append(chunk)

        if chunk_id:
            sources.append(chunk_id)

    return {
        "context": context,
        "sources": sources,
    }


def convert_history(history: list):
    messages = []

    for item in history:
        role = item.get("role")
        content = item.get("content", "")

        if role == "user":
            messages.append(
                HumanMessage(content=content)
            )

        elif role == "assistant":
            messages.append(
                AIMessage(content=content)
            )

    return messages


def needs_conversation_context(question: str) -> bool:
    """
    Return True only when the question depends on previous
    conversation context and does not identify the subject itself.
    """
    import re

    q = question.lower().strip()

    # If a course code is explicitly mentioned, the question
    # is self-contained even if it contains words like
    # "its", "ساعاتها", "متطلباتها", etc.
    course_code_pattern = r"\b[a-z]{2,5}\.?\d{3}\b"

    if re.search(course_code_pattern, q, flags=re.IGNORECASE):
        return False

    reference_phrases = [
        # English
        " its ",
        " it ",
        "that course",
        "this course",
        "that subject",
        "this subject",
        "and what",
        "what about",

        # Arabic
        "متطلباتها",
        "متطلباته",
        "ساعاتها",
        "ساعاته",
        "طب و",
        "طيب و",
        "وماذا عنها",
        "وماذا عنه",
    ]

    padded_q = f" {q} "

    return any(
        phrase in padded_q
        for phrase in reference_phrases
    )

def rewrite_question(question: str, history: list) -> str:
    if not history:
        return question

    conversation = []

    for item in history:
        role = item.get("role", "")
        content = item.get("content", "")

        conversation.append(
            f"{role}: {content}"
        )

    history_text = "\n".join(conversation)

    rewrite_prompt = f"""
Rewrite the student's latest question as a standalone question.

Use the conversation history only to resolve references such as:
it, its, that course, this subject, that requirement,
or similar follow-up references.

Do not answer the question.
Do not add academic facts.
Do not change the student's intent.

Return only the rewritten standalone question.

Conversation history:
{history_text}

Latest question:
{question}
"""

    rewritten = llm.invoke(rewrite_prompt)

    content = rewritten.content

    # Some Gemini models may return structured content blocks
    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):
                text = block.get("text")

                if text:
                    text_parts.append(text)

        content = "".join(text_parts)

    if not isinstance(content, str):
        raise RuntimeError(
            f"Unexpected LLM response type: {type(content)}"
        )

    return content.strip()


def generate_answer(question: str, context: list, history: list):
    turn_prompt = build_turn_prompt(
        history=history,
        context=context,
        question=question,
    )

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", turn_prompt),
    ]

    response = llm.invoke(messages)

    content = response.content

    # Handle Gemini structured content blocks
    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):
                text = block.get("text")

                if text:
                    text_parts.append(text)

        content = "".join(text_parts)

    if not isinstance(content, str):
        raise RuntimeError(
            f"Unexpected LLM response type: {type(content)}"
        )

    return content.strip()