import re
from typing import Optional


# =========================
# Validation
# =========================

def validate_gpa(value: float) -> Optional[float]:
    """
    Accept GPA values on the 0.0 - 4.0 scale.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if 0.0 <= value <= 4.0:
        return round(value, 2)

    return None


def validate_completed_hours(value: int) -> Optional[int]:
    """
    Validate completed credit hours.

    Upper bound is intentionally generous because this is
    input validation, not an academic-policy rule.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None

    if 0 <= value <= 300:
        return value

    return None


# =========================
# Extraction
# =========================

def extract_gpa(text: str) -> Optional[float]:
    """
    Extract GPA from Arabic, English, Egyptian Arabic,
    and mixed-language text.

    Examples:
    - My GPA is 2.8
    - GPA 3.1
    - معدلي 2.7
    - المعدل التراكمي 2.99
    - انا GPA بتاعي 3.2
    """

    patterns = [
        r"\b(?:gpa|cgpa)\s*(?:(?:بتاعي|بتاعى|حقي|حقّي)\s*)?(?:is|=|:)?\s*(\d(?:\.\d{1,2})?)\b",

        r"(?:معدلي|معدلى|المعدل(?:\s+التراكمي)?|معدل(?:ي)?(?:\s+التراكمي)?)"
        r"\s*(?:هو|=|:)?\s*(\d(?:\.\d{1,2})?)",

        r"\b(\d(?:\.\d{1,2})?)\s*(?:gpa|cgpa)\b",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return validate_gpa(
                float(match.group(1))
            )

    return None


def extract_completed_hours(text: str) -> Optional[int]:
    """
    Extract already-completed credit hours.

    We intentionally look for completion-related phrases
    so that a question like:
        "Can I register 18 hours?"
    is NOT mistaken for:
        completed_hours = 18
    """

    patterns = [
        # English
        r"(?:completed|finished|passed|earned)"
        r"\s*(\d{1,3})"
        r"\s*(?:credit\s*hours?|hours?|credits?)",

        r"(\d{1,3})"
        r"\s*(?:credit\s*hours?|hours?|credits?)"
        r"\s*(?:completed|finished|passed|earned)",

        # Arabic / Egyptian Arabic
        r"(?:خلصت|مخلص|مخلّص|أنهيت|انهيت|اجتزت|معدي|مُنجز|منجز)"
        r"\s*(\d{1,3})"
        r"\s*(?:ساعة|ساعه|ساعات)?",

        r"(?:عدد\s+الساعات\s+(?:اللي|التي)?\s*(?:خلصتها|أنهيتها|انهيتها|اجتزتها))"
        r"\s*(?:هو|=|:)?\s*(\d{1,3})",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return validate_completed_hours(
                int(match.group(1))
            )

    return None


def extract_major(text: str) -> Optional[str]:
    """
    Extract the student's explicitly stated major.

    Returns:
    - AI
    - CS
    - IS
    - General
    - None

    Important:
    Course codes such as AI.499 must NOT be treated
    as proof that the student's major is AI.
    """

    text_lower = text.lower().strip()

    # General / not specialized yet
    general_patterns = [
        r"\bgeneral\b",
        r"\bundeclared\b",
        r"لسه\s+(?:عام|جنرال)",
        r"انا\s+(?:عام|جنرال)",
        r"أنا\s+(?:عام|جنرال)",
        r"تخصصي\s+(?:عام|جنرال)",
        r"لسه\s+ما\s+تخصصت",
        r"لم\s+اتخصص",
        r"لم\s+أتخصص",
    ]

    if any(
        re.search(pattern, text_lower, flags=re.IGNORECASE)
        for pattern in general_patterns
    ):
        return "General"

    # AI
    ai_patterns = [
        r"(?:تخصصي|تخصص|major(?:\s+is)?|i(?:'m| am))\s*(?:ai|artificial intelligence)",
        r"(?:تخصصي|تخصص)\s+(?:ذكاء\s+اصطناعي|الذكاء\s+الاصطناعي)",
        r"انا\s+(?:في\s+)?(?:ذكاء\s+اصطناعي|الذكاء\s+الاصطناعي)",
        r"أنا\s+(?:في\s+)?(?:ذكاء\s+اصطناعي|الذكاء\s+الاصطناعي)",
    ]

    if any(
        re.search(pattern, text_lower, flags=re.IGNORECASE)
        for pattern in ai_patterns
    ):
        return "AI"

    # CS
    cs_patterns = [
        r"(?:تخصصي|تخصص|major(?:\s+is)?|i(?:'m| am))\s*(?:cs|computer science)",
        r"(?:تخصصي|تخصص)\s+(?:علوم\s+حاسب|علوم\s+الحاسب)",
        r"انا\s+(?:في\s+)?(?:علوم\s+حاسب|علوم\s+الحاسب)",
        r"أنا\s+(?:في\s+)?(?:علوم\s+حاسب|علوم\s+الحاسب)",
    ]

    if any(
        re.search(pattern, text_lower, flags=re.IGNORECASE)
        for pattern in cs_patterns
    ):
        return "CS"

    # IS
    is_patterns = [
        r"(?:تخصصي|تخصص|major(?:\s+is)?|i(?:'m| am))\s*(?:is|information systems?)",
        r"(?:تخصصي|تخصص)\s+(?:نظم\s+معلومات|نظم\s+المعلومات)",
        r"انا\s+(?:في\s+)?(?:نظم\s+معلومات|نظم\s+المعلومات)",
        r"أنا\s+(?:في\s+)?(?:نظم\s+معلومات|نظم\s+المعلومات)",
    ]

    if any(
        re.search(pattern, text_lower, flags=re.IGNORECASE)
        for pattern in is_patterns
    ):
        return "IS"

    return None


def extract_profile_updates(text: str) -> dict:
    """
    Extract any student-profile fields supplied in a message.
    """

    return {
        "gpa": extract_gpa(text),
        "completed_hours": extract_completed_hours(text),
        "major": extract_major(text),
    } 
   


# =========================
# Onboarding State
# =========================

def get_missing_profile_fields(profile: dict) -> list[str]:
    missing = []

    if profile.get("gpa") is None:
        missing.append("gpa")

    if profile.get("completed_hours") is None:
        missing.append("completed_hours")

    if profile.get("major") is None:
        missing.append("major")

    return missing



def profile_is_ready(profile: dict) -> bool:
    return len(
        get_missing_profile_fields(profile)
    ) == 0


# =========================
# Language
# =========================

def contains_arabic(text: str) -> bool:
    return bool(
        re.search(r"[\u0600-\u06FF]", text)
    )


# =========================
# Onboarding Messages
# =========================

def onboarding_message(
    profile: dict,
    user_text: str = "",
) -> str:
    """
    Ask only for the required profile fields
    that are still missing.
    """

    missing = get_missing_profile_fields(profile)
    arabic = contains_arabic(user_text)

    # ==================================
    # Profile complete
    # ==================================
    if not missing:
        if arabic:
            return (
                "تمام، تم تسجيل بياناتك لهذه المحادثة. "
                "كيف أقدر أساعدك أكاديميًا؟"
            )

        return (
            "Thanks. Your academic information has been saved "
            "for this session. How can I help you?"
        )

    # ==================================
    # Arabic
    # ==================================
    if arabic:
        lines = []

        # First onboarding message
        if len(missing) == 3:
            lines.append(
                "أهلًا بك في المساعد الأكاديمي لـ MUST 👋"
            )
            lines.append(
                "قبل ما نبدأ، زودني من فضلك بـ:"
            )
        else:
            lines.append(
                "تمام، باقي بس أحتاج منك:"
            )

        if "gpa" in missing:
            lines.append(
                "• المعدل التراكمي (GPA)"
            )

        if "completed_hours" in missing:
            lines.append(
                "• عدد الساعات المعتمدة التي أنهيتها"
            )

        if "major" in missing:
            lines.append(
                "• تخصصك الحالي: AI أو CS أو IS أو General"
            )

        return "\n".join(lines)

    # ==================================
    # English
    # ==================================
    lines = []

    if len(missing) == 3:
        lines.append(
            "Welcome to the MUST Academic Assistant 👋"
        )
        lines.append(
            "Before we begin, please provide:"
        )
    else:
        lines.append(
            "Thanks. I still need:"
        )

    if "gpa" in missing:
        lines.append(
            "• Your current GPA"
        )

    if "completed_hours" in missing:
        lines.append(
            "• The number of credit hours you have completed"
        )

    if "major" in missing:
        lines.append(
            "• Your current major: AI, CS, IS, or General"
        )

    return "\n".join(lines)
def question_requires_major(text: str) -> bool:
    """
    Detect questions that cannot be answered personally
    without knowing the student's academic major.
    """

    q = text.lower()

    major_dependent_phrases = [
        # English
        "what courses can i register",
        "which courses can i register",
        "what subjects can i register",
        "my semester plan",
        "my study plan",
        "courses for my major",
        "subjects for my major",

        # Arabic
        "ايه المواد اللي اقدر اسجلها",
        "إيه المواد اللي أقدر أسجلها",
        "اي المواد اللي اقدر اسجلها",
        "ما المواد التي استطيع تسجيلها",
        "ما المواد التي أستطيع تسجيلها",
        "مواد الترم",
        "مواد تخصصي",
        "خطة تخصصي",
        "الخطة بتاعتي",
        "الخطة الدراسية بتاعتي",
    ]

    return any(
        phrase in q
        for phrase in major_dependent_phrases
    )