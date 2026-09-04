import re
import time
import requests

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from .config import RAG_API_URL
from .llm import llm
from .prompts import (
    SYSTEM_PROMPT,
    build_turn_prompt,
)


# =========================================================
# Registration Load Detection
# =========================================================

def is_registration_load_question(
    question: str,
) -> bool:

    q = question.lower().strip()

    patterns = [
        # English
        r"how many\s+(?:credit\s+)?hours",
        r"how many\s+credits",
        r"maximum\s+(?:credit\s+)?hours",
        r"minimum\s+(?:credit\s+)?hours",
        r"registration\s+load",
        r"register\s+\d+\s*(?:credit\s*)?hours?",
        r"take\s+\d+\s*(?:credit\s*)?hours?",

        # Arabic
        r"كم\s+ساعة",
        r"كام\s+ساعة",
        r"عدد\s+الساعات",
        r"الحد\s+الأقصى.*ساعة",
        r"الحد\s+الادنى.*ساعة",
        r"الحد\s+الأدنى.*ساعة",
        r"اسجل\s+\d+\s*ساعة",
        r"أسجل\s+\d+\s*ساعة",
        r"اخد\s+\d+\s*ساعة",
        r"آخذ\s+\d+\s*ساعة",
    ]

    return any(
        re.search(
            pattern,
            q,
            flags=re.IGNORECASE,
        )
        for pattern in patterns
    )


# =========================================================
# GPA Helpers
# =========================================================

def extract_gpa_from_question(
    question: str,
):

    q = question.lower()

    patterns = [
        r"\b(?:gpa|cgpa)\s*"
        r"(?:of|is|=|:)?\s*"
        r"(\d(?:\.\d{1,2})?)\b",

        r"\bgpa\s+(\d(?:\.\d{1,2})?)\b",

        r"(?:معدلي|المعدل|المعدل التراكمي)"
        r"\s*(?:هو|=|:)?\s*"
        r"(\d(?:\.\d{1,2})?)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            q,
            flags=re.IGNORECASE,
        )

        if match:

            try:
                value = float(
                    match.group(1)
                )

                if 0.0 <= value <= 4.0:
                    return value

            except ValueError:
                pass

    return None


def get_profile_gpa(
    profile: dict = None,
):

    if not profile:
        return None

    value = profile.get("gpa")

    if value is None:
        return None

    try:
        value = float(value)

    except (TypeError, ValueError):
        return None

    if 0.0 <= value <= 4.0:
        return value

    return None


# =========================================================
# Retrieval Window
# =========================================================

def get_retrieval_top_k(
    question: str,
) -> int:

    if is_registration_load_question(
        question
    ):
        return 2

    return 3


# =========================================================
# Registration Filtering
# =========================================================

def filter_registration_results(
    results: list,
    question: str,
    final_top_k: int,
    profile: dict = None,
) -> list:

    gpa = get_profile_gpa(
        profile
    )

    if gpa is None:
        gpa = extract_gpa_from_question(
            question
        )

    gpa_results = []

    for item in results:

        metadata = (
            item.get("metadata")
            or {}
        )

        if (
            metadata.get("doc_type")
            == "gpa_article"
        ):
            gpa_results.append(
                item
            )

    if gpa is None:

        return gpa_results[
            :final_top_k
        ]

    if gpa < 2.0:

        preferred_ids = [
            "gpa_article_2",
            "gpa_article_1",
        ]

    elif gpa < 3.0:

        preferred_ids = [
            "gpa_article_3",
            "gpa_article_1",
        ]

    else:

        preferred_ids = [
            "gpa_article_1",
        ]

    by_chunk_id = {}

    for item in gpa_results:

        metadata = (
            item.get("metadata")
            or {}
        )

        chunk_id = metadata.get(
            "chunk_id"
        )

        if chunk_id:
            by_chunk_id[
                chunk_id
            ] = item

    selected = []

    for chunk_id in preferred_ids:

        item = by_chunk_id.get(
            chunk_id
        )

        if item is not None:

            selected.append(
                item
            )

    if len(selected) < len(
        preferred_ids
    ):

        for item in gpa_results:

            if item not in selected:

                selected.append(
                    item
                )

            if len(selected) >= final_top_k:
                break

    return selected[
        :final_top_k
    ]


from pathlib import Path
import json

_CHUNKS_CACHE = None

def _load_local_chunks():
    global _CHUNKS_CACHE
    if _CHUNKS_CACHE is not None:
        return _CHUNKS_CACHE
    chunks_path = Path(__file__).resolve().parent / "chunks_final.json"
    if not chunks_path.exists():
        chunks_path = Path(__file__).resolve().parents[2] / "must-rag-api-main" / "chunks_final.json"
    if chunks_path.exists():
        loaded = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    loaded.append(json.loads(line))
        _CHUNKS_CACHE = loaded
        return loaded
    return []

def _fallback_local_search(question: str, top_k: int) -> list:
    chunks = _load_local_chunks()
    if not chunks:
        return []
    q_words = set(re.findall(r'[\w\.]+', question.lower()))
    scored = []
    for c in chunks:
        text = c.get("chunk_text", "")
        metadata = c.get("metadata", {})
        score = 0.0
        course_code = metadata.get("course_code")
        if course_code and course_code.lower() in question.lower():
            score += 5.0
        chunk_id = c.get("chunk_id", "")
        if chunk_id and any(w in chunk_id.lower() for w in q_words if len(w) > 2):
            score += 2.0
        for w in q_words:
            if len(w) >= 2 and w in text.lower():
                score += 1.0
        if is_registration_load_question(question) and c.get("doc_type") == "gpa_article":
            score += 3.0
        if score > 0:
            scored.append((score, {
                "text": text,
                "score": score,
                "metadata": {
                    "chunk_id": chunk_id,
                    "doc_type": c.get("doc_type"),
                    "major": c.get("major"),
                    "semester": c.get("semester"),
                    "confidence": c.get("confidence", "verified"),
                }
            }))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


# =========================================================
# RAG Request With Retry
# =========================================================

def _request_rag(
    payload: dict,
) -> dict:
    """
    Call RAG API with retry for temporary upstream failures.

    Retries:
    - HTTP 502
    - HTTP 503
    - HTTP 504
    - connection / timeout errors
    """

    max_attempts = 3

    retry_statuses = {
        502,
        503,
        504,
    }

    last_error = None

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        try:

            response = requests.post(
                RAG_API_URL,
                json=payload,
                timeout=60,
            )

            # ---------------------------------------------
            # Temporary upstream failure
            # ---------------------------------------------

            if (
                response.status_code
                in retry_statuses
            ):

                last_error = RuntimeError(
                    "Temporary RAG upstream error: "
                    f"HTTP {response.status_code}"
                )

                if attempt < max_attempts:

                    wait_seconds = (
                        2 * attempt
                    )

                    print(
                        "[RAG] Temporary failure "
                        f"HTTP {response.status_code}. "
                        f"Retry {attempt}/{max_attempts} "
                        f"in {wait_seconds}s."
                    )

                    time.sleep(
                        wait_seconds
                    )

                    continue

                raise RuntimeError(
                    "RAG API is temporarily unavailable "
                    f"after {max_attempts} attempts. "
                    f"Last status: "
                    f"{response.status_code}"
                )

            # ---------------------------------------------
            # Permanent HTTP error
            # ---------------------------------------------

            response.raise_for_status()

            # ---------------------------------------------
            # JSON validation
            # ---------------------------------------------

            try:

                data = response.json()

            except ValueError as exc:

                raise RuntimeError(
                    "RAG API returned invalid JSON."
                ) from exc

            if not isinstance(
                data,
                dict,
            ):

                raise RuntimeError(
                    "RAG API returned an invalid "
                    "response structure."
                )

            return data


        except RuntimeError:

            # Our own controlled RuntimeError
            raise


        except requests.RequestException as exc:

            last_error = exc

            if attempt < max_attempts:

                wait_seconds = (
                    2 * attempt
                )

                print(
                    "[RAG] Request error "
                    f"{type(exc).__name__}. "
                    f"Retry {attempt}/{max_attempts} "
                    f"in {wait_seconds}s."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            raise RuntimeError(
                "RAG API request failed "
                f"after {max_attempts} attempts: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


    raise RuntimeError(
        "RAG API request failed: "
        f"{last_error}"
    )


# =========================================================
# RAG Retrieval
# =========================================================

def retrieve_context(
    question: str,
    top_k: int = 3,
    profile: dict = None,
) -> dict:

    registration_load = (
        is_registration_load_question(
            question
        )
    )

    retrieval_question = question

    # =====================================================
    # Inject profile GPA for registration routing
    # =====================================================

    if registration_load:
        profile_gpa = get_profile_gpa(profile)
        if profile_gpa is not None:
            retrieval_question = (
                f"{question}\n"
                f"Student current GPA: "
                f"{profile_gpa:.2f}"
            )

    request_top_k = (
        max(
            top_k,
            6,
        )
        if registration_load
        else top_k
    )

    payload = {
        "question":
            retrieval_question,
        "top_k":
            request_top_k,
    }

    # =====================================================
    # Request RAG API with Retry and Local Fallback
    # =====================================================
    results = []
    try:
        data = _request_rag(payload)
        results = data.get("results", [])
    except Exception as exc:
        print(f"[RAG] Remote RAG request failed ({exc}). Using local chunks fallback.")
        results = _fallback_local_search(retrieval_question, request_top_k)
        if not results:
            raise RuntimeError(
                f"RAG API request failed and no local chunks available: {exc}"
            ) from exc

    if not isinstance(
        results,
        list,
    ):

        raise RuntimeError(
            "RAG API 'results' field "
            "is not a list."
        )

    # =====================================================
    # Registration filtering
    # =====================================================

    if registration_load:

        results = (
            filter_registration_results(
                results=results,
                question=retrieval_question,
                final_top_k=top_k,
                profile=profile,
            )
        )

    else:

        results = results[
            :top_k
        ]

    # =====================================================
    # Normalize Context
    # =====================================================

    context = []
    sources = []

    for item in results:

        if not isinstance(
            item,
            dict,
        ):
            continue

        text = (
            item.get(
                "text",
                "",
            )
            .strip()
        )

        metadata = (
            item.get("metadata")
            or {}
        )

        if not text:
            continue

        chunk_id = metadata.get(
            "chunk_id"
        )

        chunk = {
            "chunk_id":
                chunk_id,

            "doc_type":
                metadata.get(
                    "doc_type"
                ),

            "major":
                metadata.get(
                    "major"
                ),

            "semester":
                metadata.get(
                    "semester"
                ),

            "confidence":
                metadata.get(
                    "confidence"
                ),

            "text":
                text,
        }

        context.append(
            chunk
        )

        if chunk_id:

            sources.append(
                chunk_id
            )

    return {
        "context":
            context,

        "sources":
            sources,
    }


# =========================================================
# Conversation History Conversion
# =========================================================

def convert_history(
    history: list,
):

    messages = []

    for item in history:

        role = item.get(
            "role"
        )

        content = item.get(
            "content",
            "",
        )

        if role == "user":

            messages.append(
                HumanMessage(
                    content=content
                )
            )

        elif role == "assistant":

            messages.append(
                AIMessage(
                    content=content
                )
            )

    return messages


# =========================================================
# Follow-up Detection
# =========================================================

def needs_conversation_context(
    question: str,
) -> bool:

    q = (
        question
        .lower()
        .strip()
    )

    course_code_pattern = (
        r"\b[a-z]{2,5}\.?\d{3}\b"
    )

    if re.search(
        course_code_pattern,
        q,
        flags=re.IGNORECASE,
    ):
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

    padded_q = (
        f" {q} "
    )

    return any(
        phrase in padded_q
        for phrase
        in reference_phrases
    )


# =========================================================
# Follow-up Rewrite
# =========================================================

def rewrite_question(
    question: str,
    history: list,
) -> str:

    if not history:
        return question

    conversation = []

    for item in history:

        role = item.get(
            "role",
            "",
        )

        content = item.get(
            "content",
            "",
        )

        conversation.append(
            f"{role}: {content}"
        )

    history_text = "\n".join(
        conversation
    )

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

    rewritten = llm.invoke(
        rewrite_prompt
    )

    content = rewritten.content

    if isinstance(
        content,
        list,
    ):

        text_parts = []

        for block in content:

            if isinstance(
                block,
                str,
            ):

                text_parts.append(
                    block
                )

            elif isinstance(
                block,
                dict,
            ):

                text = block.get(
                    "text"
                )

                if text:

                    text_parts.append(
                        text
                    )

        content = "".join(
            text_parts
        )

    if not isinstance(
        content,
        str,
    ):

        raise RuntimeError(
            "Unexpected LLM response type: "
            f"{type(content)}"
        )

    return content.strip()


# =========================================================
# Answer Generation
# =========================================================

def generate_answer(
    question: str,
    context: list,
    history: list,
    profile: dict = None,
):

    turn_prompt = build_turn_prompt(
        student_profile=profile,
        history=history,
        context=context,
        question=question,
    )

    messages = [
        (
            "system",
            SYSTEM_PROMPT,
        ),
        (
            "human",
            turn_prompt,
        ),
    ]

    response = llm.invoke(
        messages
    )

    content = response.content

    if isinstance(
        content,
        list,
    ):

        text_parts = []

        for block in content:

            if isinstance(
                block,
                str,
            ):

                text_parts.append(
                    block
                )

            elif isinstance(
                block,
                dict,
            ):

                text = block.get(
                    "text"
                )

                if text:

                    text_parts.append(
                        text
                    )

        content = "".join(
            text_parts
        )

    if not isinstance(
        content,
        str,
    ):

        raise RuntimeError(
            "Unexpected LLM response type: "
            f"{type(content)}"
        )

    return content.strip()