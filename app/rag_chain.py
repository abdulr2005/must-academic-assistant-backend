from .nlp import parse_query, normalize_course_codes
from .answer_evidence import (
    as_result, compatible_snapshot, complete_core, registration_evidence,
    exact_courses, specialization_hours_guard, core_catalogue_answer,
)
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

    parsed = parse_query(question)
    if parsed.intent in {"SPECIFIC_COURSE", "PREREQUISITE", "SPECIALIZATION", "MAJOR_ELECTIVES", "CORE_REQUIRED_COURSES", "SEMESTER_PLAN", "BROAD_MAJOR_CURRICULUM"}:
        return False
    if parsed.intent == "REGISTRATION_HOURS":
        return True
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


def is_remaining_graduation_hours_question(
    question: str,
) -> bool:
    """Detect requests for a student's remaining degree credit hours."""

    q = question.lower().strip()

    patterns = [
        r"how many (?:credit )?hours (?:do i have )?left (?:to|until) graduat",
        r"remaining (?:credit )?hours.*graduat",
        r"hours.*remaining.*graduat",
        r"(?:فاضلي|باقي|متبقي(?:ة)?)\s+كام\s+ساعة.*(?:للتخرج|التخرج)",
        r"كم\s+ساعة\s+متبقية.*(?:للتخرج|التخرج)",
        r"ساعات\s+التخرج\s+المتبقية",
    ]

    return any(
        re.search(pattern, q, flags=re.IGNORECASE)
        for pattern in patterns
    )


def is_major_curriculum_question(
    question: str,
) -> bool:
    """Detect plan/elective questions whose sources are major-specific."""

    q = question.lower().strip()

    patterns = [
        r"semester\s*\d+",
        r"courses?.*(?:in|for)\s+semester",
        r"semester.*courses?",
        r"study plan",
        r"curriculum",
        r"major.*electives?",
        r"electives?.*(?:major|speciali[sz]ation)",
        r"مواد\s+(?:ال)?ترم",
        r"مواد\s+الفصل",
        r"الخطة\s+الدراسية",
        r"مواد\s+تخصص",
        r"اختياري.*تخصص",
    ]

    return any(
        re.search(pattern, q, flags=re.IGNORECASE)
        for pattern in patterns
    )


def extract_requested_semester(question: str):
    return parse_query(question).semester


def is_semester_plan_question(question: str) -> bool:
    return parse_query(question).intent == "SEMESTER_PLAN"


def is_specialization_requirements_question(question: str) -> bool:
    """Detect eligibility/rule questions about choosing a specialization."""

    parsed = parse_query(question)
    if parsed.intent == "SPECIALIZATION":
        return True
    if parsed.intent in {"SPECIFIC_COURSE", "PREREQUISITE", "MAJOR_ELECTIVES", "CORE_REQUIRED_COURSES", "SEMESTER_PLAN", "BROAD_MAJOR_CURRICULUM"}:
        return False
    q = question.lower().strip()
    patterns = [
        r"speciali[sz](?:e|ation|ing)",
        r"choose\s+(?:a\s+)?major",
        r"select\s+(?:a\s+)?major",
        r"major\s+(?:selection|requirements?|eligibility)",
        r"(?:اتخصص|أتخصص|التخصص|اختار\s+التخصص|أختار\s+التخصص)",
        r"(?:شروط|متطلبات).*تخصص",
    ]
    return any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in patterns)


def is_broad_major_curriculum_question(question: str) -> bool:
    return parse_query(question).intent == "BROAD_MAJOR_CURRICULUM"


def is_core_required_courses_question(question: str) -> bool:
    return parse_query(question).intent == "CORE_REQUIRED_COURSES"


def is_major_electives_question(question: str) -> bool:
    return parse_query(question).intent == "MAJOR_ELECTIVES"


def classify_curriculum_intent(question: str):
    intent = parse_query(question).intent
    return intent if intent in {"PREREQUISITE", "SPECIFIC_COURSE", "MAJOR_ELECTIVES", "CORE_REQUIRED_COURSES", "SEMESTER_PLAN", "BROAD_MAJOR_CURRICULUM"} else None


def is_narrow_course_identity_question(
    question: str,
) -> bool:
    """Match generic course-identity questions, excluding attribute queries."""

    q = question.lower().strip()
    course_code = r"[a-z]{2,5}[.\- ]?\d{3}"

    if not re.search(course_code, q, flags=re.IGNORECASE):
        return False

    excluded_attributes = (
        "prerequisite",
        "prereq",
        "credit hour",
        "hours",
        "semester",
        "level",
        "elective",
        "compulsory",
        "متطلب",
        "ساعات",
        "ساعة",
        "ترم",
        "فصل",
        "اختياري",
        "إجباري",
    )

    if any(attribute in q for attribute in excluded_attributes):
        return False

    return bool(
        re.fullmatch(
            rf"\s*(?:what is|what's|tell me about|ما هي|ما هو|ايه|إيه)\s+{course_code}\s*[?.؟]*\s*",
            q,
            flags=re.IGNORECASE,
        )
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

    curriculum_intent = classify_curriculum_intent(question)
    if curriculum_intent == "BROAD_MAJOR_CURRICULUM":
        return 4
    if curriculum_intent in {"CORE_REQUIRED_COURSES", "MAJOR_ELECTIVES"}:
        return 6

    if is_remaining_graduation_hours_question(
        question
    ):
        return 3

    if is_registration_load_question(
        question
    ):
        return 2

    return 3


def _profile_major(profile: dict = None):
    if not profile:
        return None

    major = profile.get("major")
    if not isinstance(major, str):
        return None

    normalized = major.strip().upper()
    return normalized if normalized in {"AI", "CS", "IS"} else None


def _curriculum_major(question: str, profile: dict = None):
    major = parse_query(question, profile).major
    return major.upper() if major else None


def filter_broad_curriculum_results(
    results: list,
    question: str,
    profile: dict,
    final_top_k: int,
) -> list:
    """Prefer compact same-major plans; fall back to same-major core records."""

    target_major = _curriculum_major(question, profile)
    if target_major is None:
        return []

    plans = []
    core_courses = []
    for item in results:
        metadata = item.get("metadata") or {}
        doc_type = str(metadata.get("doc_type") or "").lower()
        result_major = str(metadata.get("major") or "").upper()
        text = str(item.get("text") or "")

        matches_major = bool(
            re.search(rf"\b{re.escape(target_major)}\b", result_major)
        )
        if target_major == "GENERAL":
            matches_major = result_major.startswith("GENERAL")
        if not matches_major:
            continue

        if doc_type in {"semester_plan", "general_regulation_semester"}:
            plans.append(item)
        elif doc_type in {"major_regulation_course", "course"} and re.search(
            r"major core", text, flags=re.IGNORECASE
        ):
            core_courses.append(item)

    return (plans if plans else core_courses)[:final_top_k]


def filter_core_required_results(
    results: list,
    question: str,
    profile: dict,
    final_top_k: int,
) -> list:
    """Keep officially labelled same-major core courses, excluding electives."""

    target_major = _curriculum_major(question, profile)
    if target_major is None:
        return []

    official_core = []
    plan_fallback = []
    for item in results:
        metadata = item.get("metadata") or {}
        doc_type = str(metadata.get("doc_type") or "").lower()
        result_major = str(metadata.get("major") or "").upper()
        text = str(item.get("text") or "")
        matches_major = bool(re.search(rf"\b{re.escape(target_major)}\b", result_major))
        if target_major == "GENERAL":
            matches_major = result_major.startswith("GENERAL")
        if not matches_major:
            continue

        if doc_type in {"major_regulation_course", "course"}:
            if re.search(r"major core", text, flags=re.IGNORECASE):
                official_core.append(item)
            continue

        if doc_type in {"semester_plan", "general_regulation_semester"}:
            cleaned = re.sub(
                r"(?:EC\(\d+\)|Elective Course slot).*?(?=(?:[A-Z]{2,5}[.]?\d+|EC\(\d+\))\s*\(|$)",
                "",
                text,
                flags=re.IGNORECASE,
            )
            if cleaned.strip():
                derived = dict(item)
                derived["text"] = (
                    "Required/non-elective courses in the official study plan: "
                    + cleaned.strip()
                )
                plan_fallback.append(derived)

    return (official_core if official_core else plan_fallback)[:final_top_k]


def filter_major_elective_results(
    results: list,
    question: str,
    profile: dict,
    final_top_k: int,
) -> list:
    """Keep only actual elective-pool choices belonging to the resolved major."""

    target_major = _curriculum_major(question, profile)
    if target_major not in {"AI", "CS", "IS"}:
        return []

    prefix = f"elective_{target_major}_".lower()
    selected = []
    for item in results:
        metadata = item.get("metadata") or {}
        chunk_id = str(metadata.get("chunk_id") or "").lower()
        doc_type = str(metadata.get("doc_type") or "").lower()
        if doc_type == "elective_pool_course" and chunk_id.startswith(prefix):
            selected.append(item)
    return selected[:final_top_k]


def filter_major_curriculum_results(
    results: list,
    profile: dict,
    final_top_k: int,
) -> list:
    """Remove other-major curriculum chunks without filtering common policy."""

    profile_major = _profile_major(profile)
    if profile_major is None:
        return results[:final_top_k]

    matching_major = []
    neutral_or_common = []
    major_specific_types = {
        "semester_plan",
        "elective_pool_course",
    }

    for item in results:
        metadata = item.get("metadata") or {}
        doc_type = metadata.get("doc_type")
        result_major = str(metadata.get("major") or "").upper()

        if doc_type in major_specific_types:
            if result_major.startswith(profile_major):
                matching_major.append(item)
            elif "SHARED" in result_major or "COMMON" in result_major:
                neutral_or_common.append(item)
            continue

        neutral_or_common.append(item)

    return (
        matching_major
        + neutral_or_common
    )[:final_top_k]


def filter_semester_plan_results(
    results: list,
    question: str,
    profile: dict,
    final_top_k: int,
) -> list:
    """Keep the requested semester plan without mixing curriculum tracks."""

    semester = extract_requested_semester(question)
    profile_major = _curriculum_major(question, profile)
    preferred_major = "GENERAL" if semester is not None and semester <= 4 else profile_major
    plans = []
    common = []

    for item in results:
        metadata = item.get("metadata") or {}
        doc_type = str(metadata.get("doc_type") or "").lower()
        result_major = str(metadata.get("major") or "").upper()
        result_semester = metadata.get("semester")

        if doc_type in {"semester_plan", "general_regulation_semester"}:
            try:
                if semester is not None and int(result_semester) != semester:
                    continue
            except (TypeError, ValueError):
                continue

            if preferred_major == "GENERAL":
                if result_major.startswith("GENERAL"):
                    plans.append(item)
            elif preferred_major:
                if result_major.startswith(preferred_major):
                    plans.append(item)
                elif "COMMON" in result_major or "SHARED" in result_major:
                    common.append(item)
            elif result_major.startswith("GENERAL"):
                plans.append(item)
            continue

        # Preserve genuinely common faculty evidence, but do not let unrelated
        # GPA/registration articles leak into a semester-plan answer.
        if (
            doc_type in {"faculty_regulation", "common_regulation"}
            or "COMMON" in result_major
            or "SHARED" in result_major
        ):
            common.append(item)

    return (plans + common)[:final_top_k]


def filter_specialization_results(results: list, final_top_k: int) -> list:
    """Keep only authoritative chunks that state specialization rules."""

    selected = []
    allowed_types = {
        "specialization_transition",
        "specialization_requirements",
        "major_selection_rule",
    }
    for item in results:
        metadata = item.get("metadata") or {}
        doc_type = str(metadata.get("doc_type") or "").lower()
        chunk_id = str(metadata.get("chunk_id") or "").lower()
        if doc_type in allowed_types or "specialization" in chunk_id:
            selected.append(item)
    return selected[:final_top_k]


def filter_graduation_hours_results(
    results: list,
    profile: dict,
    final_top_k: int,
) -> list:
    """Keep only authoritative chunks that explicitly state a degree total."""

    profile_major = _profile_major(profile)
    selected = []

    for item in results:
        metadata = item.get("metadata") or {}
        doc_type = str(metadata.get("doc_type") or "").lower()
        chunk_id = str(metadata.get("chunk_id") or "").lower()
        result_major = str(metadata.get("major") or "").upper()
        text = str(item.get("text") or "").lower()

        authoritative_type = (
            doc_type in {
                "graduation_requirements",
                "program_requirements",
                "degree_requirements",
            }
            or "graduation_requirements" in chunk_id
            or "program_total_hours" in chunk_id
        )
        states_total = (
            re.search(
                r"(?:total|required|program|degree).*\d{2,3}.*credit hours",
                text,
            )
            or re.search(
                r"(?:إجمالي|المطلوبة).*\d{2,3}.*(?:ساعة|ساعات)",
                text,
            )
        )

        if not authoritative_type or not states_total:
            continue

        if (
            profile_major is not None
            and result_major
            and not result_major.startswith(profile_major)
            and "SHARED" not in result_major
            and "COMMON" not in result_major
        ):
            continue

        selected.append(item)

    return selected[:final_top_k]


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
        requested_semester = extract_requested_semester(question)
        if (
            is_semester_plan_question(question)
            and requested_semester is not None
            and c.get("doc_type") in {"semester_plan", "general_regulation_semester"}
            and c.get("semester") == requested_semester
        ):
            score += 8.0
        if (
            is_specialization_requirements_question(question)
            and c.get("doc_type") in {
                "specialization_transition",
                "specialization_requirements",
                "major_selection_rule",
            }
        ):
            score += 8.0
        curriculum_intent = classify_curriculum_intent(question)
        if curriculum_intent in {
            "BROAD_MAJOR_CURRICULUM",
            "CORE_REQUIRED_COURSES",
            "MAJOR_ELECTIVES",
        }:
            target_major = _curriculum_major(question)
            result_major = str(c.get("major") or "").upper()
            if target_major and re.search(rf"\b{re.escape(target_major)}\b", result_major):
                if curriculum_intent == "MAJOR_ELECTIVES" and (
                    c.get("doc_type") == "elective_pool_course"
                    and chunk_id.lower().startswith(f"elective_{target_major.lower()}_")
                ):
                    score += 10.0
                elif curriculum_intent == "CORE_REQUIRED_COURSES" and (
                    c.get("doc_type") in {"major_regulation_course", "course"}
                    and re.search(r"major core", text, flags=re.IGNORECASE)
                ):
                    score += 10.0
                elif c.get("doc_type") == "semester_plan":
                    score += 8.0
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

    question = normalize_course_codes(question)

    graduation_hours = (
        is_remaining_graduation_hours_question(
            question
        )
    )

    registration_load = (
        not graduation_hours
        and not is_specialization_requirements_question(question)
        and
        is_registration_load_question(
            question
        )
    )

    retrieval_question = question
    curriculum_intent = classify_curriculum_intent(question)
    routed_curriculum_intents = {
        "BROAD_MAJOR_CURRICULUM",
        "CORE_REQUIRED_COURSES",
        "MAJOR_ELECTIVES",
    }
    curriculum_major = (
        _curriculum_major(question, profile)
        if curriculum_intent in routed_curriculum_intents
        else None
    )
    if curriculum_intent in routed_curriculum_intents and curriculum_major:
        preference = {
            "BROAD_MAJOR_CURRICULUM": "authoritative semester plans",
            "CORE_REQUIRED_COURSES": "officially labelled Major Core course records",
            "MAJOR_ELECTIVES": "actual elective pool course records",
        }[curriculum_intent]
        retrieval_question = (
            f"{question}\n"
            f"Target major curriculum: {curriculum_major}. "
            f"Prefer {preference}."
        )

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

    # =====================================================
    # Wider candidate pool
    # =====================================================

    if graduation_hours:
        request_top_k = max(top_k, 10)
    elif curriculum_intent == "BROAD_MAJOR_CURRICULUM":
        request_top_k = max(top_k, 6)
    elif curriculum_intent in {"CORE_REQUIRED_COURSES", "MAJOR_ELECTIVES"}:
        request_top_k = max(top_k, 8)
    elif is_semester_plan_question(question):
        # The deployed SearchRequest schema accepts top_k values from 1 to 10.
        request_top_k = max(top_k, 10)
    elif is_specialization_requirements_question(question):
        request_top_k = max(top_k, 10)
    elif is_major_curriculum_question(question):
        request_top_k = max(top_k, 9)
    elif registration_load:
        request_top_k = max(top_k, 6)
    else:
        request_top_k = top_k

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

    if graduation_hours:

        results = filter_graduation_hours_results(
            results=results,
            profile=profile,
            final_top_k=top_k,
        )

    elif curriculum_intent == "MAJOR_ELECTIVES":

        results = filter_major_elective_results(
            results=results,
            question=question,
            profile=profile,
            final_top_k=top_k,
        )

    elif curriculum_intent == "CORE_REQUIRED_COURSES":

        results = filter_core_required_results(
            results=results,
            question=question,
            profile=profile,
            final_top_k=top_k,
        )

    elif curriculum_intent == "BROAD_MAJOR_CURRICULUM":

        results = filter_broad_curriculum_results(
            results=results,
            question=question,
            profile=profile,
            final_top_k=top_k,
        )

    elif is_semester_plan_question(question):

        results = filter_semester_plan_results(
            results=results,
            question=question,
            profile=profile,
            final_top_k=top_k,
        )

    elif is_specialization_requirements_question(question):

        results = filter_specialization_results(
            results=results,
            final_top_k=top_k,
        )

    elif (
        is_major_curriculum_question(question)
        and _profile_major(profile) is not None
    ):

        results = filter_major_curriculum_results(
            results=results,
            profile=profile,
            final_top_k=top_k,
        )

    elif registration_load:

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
    # Exact catalogue sets supplement bounded similarity windows only when the
    # remote and bundled source revisions agree. The remote top_k is unchanged.
    # =====================================================

    local_chunks = _load_local_chunks()
    if compatible_snapshot(results, local_chunks):
        if curriculum_intent == "CORE_REQUIRED_COURSES" and curriculum_major:
            results = complete_core(local_chunks, curriculum_major) or results
        elif registration_load:
            results = registration_evidence(local_chunks) or results

    if curriculum_intent in {"SPECIFIC_COURSE", "PREREQUISITE"} and parse_query(question).course_code:
        exact = exact_courses(results, question)
        if compatible_snapshot(results, local_chunks):
            exact += exact_courses([as_result(c) for c in local_chunks], question)
        seen = set()
        results = []
        for item in exact:
            chunk_id = (item.get("metadata") or {}).get("chunk_id")
            if chunk_id not in seen:
                seen.add(chunk_id)
                results.append(item)
        results = results[:top_k]

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

        for field in ("course_code", "course_name_en", "credit_hours"):
            if field in metadata:
                chunk[field] = metadata[field]

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

    curriculum_intent = classify_curriculum_intent(question)
    answer_profile = profile
    if curriculum_intent == "CORE_REQUIRED_COURSES":
        catalogue_answer = core_catalogue_answer(question, context)
        if catalogue_answer:
            return catalogue_answer
    specialization = is_specialization_requirements_question(question)
    if specialization:
        guarded = specialization_hours_guard(question, context)
        if guarded:
            return guarded
        if profile:
            answer_profile = dict(profile)
            answer_profile["completed_hours"] = None
    if curriculum_intent == "BROAD_MAJOR_CURRICULUM" and profile:
        # Completed hours do not establish the student's current level,
        # semester, completed courses, or remaining curriculum.
        answer_profile = dict(profile)
        answer_profile["completed_hours"] = None

    turn_prompt = build_turn_prompt(
        student_profile=answer_profile,
        history=history,
        context=context,
        question=question,
    )

    if is_narrow_course_identity_question(question):
        turn_prompt += (
            "\n\n<response_scope>\n"
            "This is a narrow course-identity request. "
            "Answer with only the course code, course name, and credit hours. "
            "Do not mention prerequisites, semester, contact hours, or other "
            "attributes unless the student explicitly asked for them.\n"
            "</response_scope>"
        )

    if curriculum_intent == "BROAD_MAJOR_CURRICULUM":
        turn_prompt += (
            "\n\n<response_scope>\n"
            "Present the full major curriculum represented by <context>. "
            "Do not infer the student's current level or semester from completed_hours. "
            "Do not infer that earlier semesters or courses are completed. "
            "An empty completed_courses list is untracked, not evidence of completion. "
            "Do not describe courses as remaining unless the student explicitly asks what "
            "is left and sufficient completed-course evidence is available.\n"
            "</response_scope>"
        )

    if specialization:
        turn_prompt += (
            "\n<response_scope>Completed hours never prove completion of a semester "
            "or its courses. A semester-based specialization rule must not be "
            "converted into a numerical credit-hour threshold. State explicitly "
            "when the retrieved evidence does not establish such a threshold.</response_scope>"
        )
    if is_registration_load_question(question):
        turn_prompt += (
            "\n<response_scope>State registration limits, exceptions, and conditions "
            "only when supported by the returned registration articles. Read all "
            "returned articles together; do not import numerical examples from "
            "instructions as policy or omit qualifications in the evidence.</response_scope>"
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
