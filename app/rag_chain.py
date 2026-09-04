import re
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
    """
    Detect questions about HOW MANY credit hours
    the student is allowed to register.

    This must NOT trigger for generic course-registration
    questions such as:
        Can I register AI.499?
    """

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
    """
    Fallback GPA extraction from the question text.

    Profile GPA is preferred whenever available.
    """

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
    """
    Safely get GPA from session profile.
    """

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
    """
    Keep final context small.

    Registration-load questions use regulation-only
    filtering after a wider candidate retrieval.
    """

    if is_registration_load_question(
        question
    ):
        return 2

    return 3


# =========================================================
# Registration Rule Filtering
# =========================================================

def filter_registration_results(
    results: list,
    question: str,
    final_top_k: int,
    profile: dict = None,
) -> list:
    """
    Filter registration-load retrieval to the correct
    GPA regulation articles.

    Priority:
        1. GPA stored in session profile
        2. GPA mentioned in question text

    This prevents semester-plan/course chunks from
    contaminating registration-load answers.
    """

    # =====================================================
    # Prefer profile GPA
    # =====================================================

    gpa = get_profile_gpa(
        profile
    )

    # Fallback to question only if profile has no GPA
    if gpa is None:
        gpa = extract_gpa_from_question(
            question
        )


    # =====================================================
    # Keep only GPA regulation chunks
    # =====================================================

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


    # =====================================================
    # If GPA is unavailable
    # =====================================================

    if gpa is None:
        return gpa_results[
            :final_top_k
        ]


    # =====================================================
    # Select required regulation articles
    # =====================================================

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


    # =====================================================
    # Build lookup by chunk_id
    # =====================================================

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


    # =====================================================
    # Select preferred chunks
    # =====================================================

    selected = []

    for chunk_id in preferred_ids:

        item = by_chunk_id.get(
            chunk_id
        )

        if item is not None:
            selected.append(
                item
            )


    # =====================================================
    # Defensive fallback
    # =====================================================

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


# =========================================================
# RAG Retrieval
# =========================================================

def retrieve_context(
    question: str,
    top_k: int = 3,
    profile: dict = None,
) -> dict:
    """
    Retrieve RAG context.

    For registration-load questions, session profile GPA
    is injected into retrieval routing so image-extracted
    or previously stored GPA is always available.
    """

    registration_load = (
        is_registration_load_question(
            question
        )
    )


    # =====================================================
    # Build retrieval query
    # =====================================================

    retrieval_question = question

    if registration_load:

        profile_gpa = get_profile_gpa(
            profile
        )

        if profile_gpa is not None:

            retrieval_question = (
                f"{question}\n"
                f"Student current GPA: "
                f"{profile_gpa:.2f}"
            )


    # =====================================================
    # Wider candidate pool for registration rules
    # =====================================================

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
    # RAG API request
    # =====================================================

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

    results = data.get(
        "results",
        [],
    )


    # =====================================================
    # Registration filtering
    # =====================================================

    if registration_load:

        results = filter_registration_results(
            results=results,
            question=retrieval_question,
            final_top_k=top_k,
            profile=profile,
        )

    else:

        results = results[
            :top_k
        ]


    # =====================================================
    # Normalize context
    # =====================================================

    context = []
    sources = []

    for item in results:

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
    """
    Return True when the question depends on previous
    conversation context and does not identify the
    subject itself.
    """

    q = (
        question
        .lower()
        .strip()
    )


    # =====================================================
    # Explicit course code = self-contained
    # =====================================================

    course_code_pattern = (
        r"\b[a-z]{2,5}\.?\d{3}\b"
    )

    if re.search(
        course_code_pattern,
        q,
        flags=re.IGNORECASE,
    ):
        return False


    # =====================================================
    # Follow-up references
    # =====================================================

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
    """
    Rewrite contextual follow-ups as standalone questions.
    """

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


    # =====================================================
    # Gemini-style structured content
    # =====================================================

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
    """
    Generate the final grounded academic answer.
    """

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


    # =====================================================
    # Gemini-style structured content
    # =====================================================

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