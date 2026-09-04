import json
import re
from typing import Optional

from .llm import llm


# =========================================================
# Validation
# =========================================================

def validate_gpa(value) -> Optional[float]:
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if 0.0 <= value <= 4.0:
        return round(value, 2)

    return None


def validate_completed_hours(value) -> Optional[int]:
    if value is None:
        return None

    try:
        value = int(float(value))
    except (TypeError, ValueError):
        return None

    if 0 <= value <= 300:
        return value

    return None


def validate_major(value) -> Optional[str]:
    if value is None:
        return None

    value = str(value).strip().lower()

    mapping = {
        "ai": "AI",
        "artificial intelligence": "AI",
        "ذكاء اصطناعي": "AI",
        "الذكاء الاصطناعي": "AI",

        "cs": "CS",
        "computer science": "CS",
        "علوم حاسب": "CS",
        "علوم الحاسب": "CS",

        "is": "IS",
        "information systems": "IS",
        "information system": "IS",
        "نظم معلومات": "IS",
        "نظم المعلومات": "IS",

        "general": "General",
        "undeclared": "General",
        "عام": "General",
        "جنرال": "General",
    }

    return mapping.get(value)


# =========================================================
# Response Content Helper
# =========================================================

def _extract_llm_text(response) -> str:
    """
    Normalize LLM response content.

    Supports normal string responses and Gemini-style
    structured content blocks.
    """

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)

            elif isinstance(block, dict):
                text = block.get("text")

                if text:
                    parts.append(text)

        return "".join(parts).strip()

    return ""


# =========================================================
# LLM Profile Parser
# =========================================================

def extract_profile_with_llm(text: str) -> dict:
    """
    Primary semantic parser for student profile information.

    The LLM understands the meaning of the student's message
    rather than depending on fixed wording or language.
    """

    prompt = f"""
You are the profile parser for the MUST Academic Assistant.

Your job is ONLY to extract academic profile information
that the student explicitly provides about THEMSELVES.

The user may write naturally in any language or style,
including:

- Arabic
- Egyptian Arabic
- Gulf/Saudi Arabic
- English
- Bengali
- mixed languages
- Arabizi / Franco-Arabic
- abbreviations
- spelling mistakes
- grammar mistakes
- informal conversational language
- very short answers

Understand the MEANING of the message.
Do NOT depend on exact keywords or sentence structure.

Return ONLY valid JSON.

Use EXACTLY this structure:

{{
    "gpa": null,
    "completed_hours": null,
    "major": null
}}

============================================================
GPA
============================================================

"gpa" means the student's CURRENT cumulative GPA.

Examples:

"My GPA is 2.8"
"gpa 3"
"cgpa 2.7"
"معدلي 2.8"
"المعدل بتاعي 3"
"انا معدلي 2.99"
"mo3adaly 2.8"

These should extract the corresponding GPA.

Valid GPA range:
0.0 - 4.0

Do NOT extract:
- a GPA requirement for a course
- another student's GPA
- a hypothetical GPA
- a GPA the student is asking about

Example:

"If my GPA becomes 3 can I register 23 hours?"

gpa = null

because 3 is hypothetical, not the student's stated
current GPA.


============================================================
COMPLETED HOURS
============================================================

"completed_hours" means the total credit hours the
student has ALREADY completed, passed, earned or finished.

Examples:

"I completed 108 credit hours"
"I've finished 70 hours"
"I have completed 90 credits"
"My completed hours are 108"
"My credit hours are 108"
"I currently have 108 completed hours"

"خلصت 108 ساعة"
"مخلص 70 ساعة"
"أنا مخلص 90"
"ساعاتي اللي خلصتها 108"
"عديت 108 ساعة"
"أنا عندي 70 ساعة مخلصة"

"ana 5alast 108 sa3a"
"5alast 70 credit hours"
"ana mkhlas 90 sa3a"

These should extract completed_hours.

IMPORTANT:

Registration hours are NOT completed hours.

Examples:

"Can I register 18 hours?"
"Can I take 21 credits?"
"How many hours can I register?"
"Is 15 hours allowed?"
"Can I register more than 18 hours?"

"اقدر اسجل 18 ساعة؟"
"ينفع اخد 21 ساعة؟"
"كم ساعة اقدر اسجل؟"
"هل مسموح لي 15 ساعة؟"

For all of these:

completed_hours = null


============================================================
MAJOR
============================================================

"major" means the student's CURRENT academic major.

Allowed output values ONLY:

"AI"
"CS"
"IS"
"General"
null


Normalize these meanings:

Artificial Intelligence
AI
ذكاء اصطناعي
الذكاء الاصطناعي
→ "AI"

Computer Science
CS
علوم الحاسب
علوم حاسب
→ "CS"

Information Systems
IS
نظم المعلومات
نظم معلومات
→ "IS"

General
Undeclared
not specialized yet
عام
جنرال
لسه ما تخصصت
→ "General"


Examples:

"My major is AI"
"I'm an AI student"
"I study Artificial Intelligence"
"تخصصي ذكاء اصطناعي"
"انا AI"
"t5asosy AI"

major = "AI"


CRITICAL RULE:

NEVER infer the student's major from a course code,
course name, or academic question.

Examples:

"What is AI.499?"
"Can I take CS.301?"
"What is the prerequisite of IS.402?"

All of these:

major = null


============================================================
MULTIPLE VALUES
============================================================

The student may provide several profile fields
in the same message.

Example:

"My GPA is 2.8, I completed 70 credit hours,
and my major is AI."

Return:

{{
    "gpa": 2.8,
    "completed_hours": 70,
    "major": "AI"
}}

Example:

"معدلي 3 ومخلص 108 ساعة وتخصصي ذكاء اصطناعي"

Return:

{{
    "gpa": 3.0,
    "completed_hours": 108,
    "major": "AI"
}}


============================================================
CONVERSATIONAL REPLIES
============================================================

The user may answer a question with a very short reply.

However, this parser receives ONLY the current message.

Therefore:

If a bare value has no clear meaning by itself,
do NOT guess.

Example:

"108"

Return null fields unless the meaning is explicitly clear
from the message itself.

Never invent profile information.


============================================================
STRICT SAFETY RULES
============================================================

1. Extract only information about the student.

2. Never invent missing information.

3. Never calculate or estimate GPA.

4. Never infer completed hours from semester number.

5. Never interpret desired registration hours as
   completed hours.

6. Never infer major from a course code.

7. Never infer major from a course question.

8. Never change an uncertain value into a guess.

9. If information is absent:
   return null.

10. Return JSON ONLY.

No markdown.
No explanation.
No comments.
No text before or after the JSON.


STUDENT MESSAGE:

{text}
"""

    response = llm.invoke(prompt)

    content = _extract_llm_text(response)

    # Remove markdown fences defensively
    content = re.sub(
        r"^```(?:json)?\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )

    content = re.sub(
        r"\s*```$",
        "",
        content,
    )

    content = content.strip()

    result = json.loads(content)

    return {
        "gpa": validate_gpa(
            result.get("gpa")
        ),
        "completed_hours": validate_completed_hours(
            result.get("completed_hours")
        ),
        "major": validate_major(
            result.get("major")
        ),
    }


# =========================================================
# Emergency Regex Fallback
# =========================================================

def extract_profile_fallback(text: str) -> dict:
    """
    Emergency fallback ONLY.

    This is used when the LLM provider fails or returns
    invalid output.

    It is intentionally conservative.
    """

    result = {
        "gpa": None,
        "completed_hours": None,
        "major": None,
    }

    # -----------------------------------------------------
    # GPA
    # -----------------------------------------------------

    gpa_patterns = [
        r"\b(?:gpa|cgpa)\s*(?:is|=|:)?\s*"
        r"(\d(?:\.\d{1,2})?)\b",

        r"(?:معدلي|معدلى|المعدل(?:\s+التراكمي)?)"
        r"\s*(?:هو|=|:)?\s*"
        r"(\d(?:\.\d{1,2})?)",
    ]

    for pattern in gpa_patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            result["gpa"] = validate_gpa(
                match.group(1)
            )
            break

    # -----------------------------------------------------
    # Completed Hours
    # -----------------------------------------------------

    hours_patterns = [
        r"(?:completed|finished|passed|earned)"
        r"\s*(\d{1,3})"
        r"\s*(?:credit\s*hours?|credits?|hours?)",

        r"(?:خلصت|مخلص|مخلّص|أنهيت|انهيت|اجتزت)"
        r"\s*(\d{1,3})"
        r"\s*(?:ساعة|ساعه|ساعات)?",
    ]

    for pattern in hours_patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            result["completed_hours"] = (
                validate_completed_hours(
                    match.group(1)
                )
            )
            break

    # -----------------------------------------------------
    # Major
    # -----------------------------------------------------

    major_patterns = [
        (
            r"(?:my\s+major\s+is|تخصصي)"
            r"\s*(?:ai|artificial intelligence|"
            r"ذكاء اصطناعي|الذكاء الاصطناعي)",
            "AI",
        ),
        (
            r"(?:my\s+major\s+is|تخصصي)"
            r"\s*(?:cs|computer science|"
            r"علوم حاسب|علوم الحاسب)",
            "CS",
        ),
        (
            r"(?:my\s+major\s+is|تخصصي)"
            r"\s*(?:is|information systems?|"
            r"نظم معلومات|نظم المعلومات)",
            "IS",
        ),
        (
            r"(?:my\s+major\s+is|تخصصي)"
            r"\s*(?:general|undeclared|عام|جنرال)",
            "General",
        ),
    ]

    for pattern, major in major_patterns:
        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            result["major"] = major
            break

    return result


# =========================================================
# Public Profile Extraction Function
# =========================================================

def extract_profile_updates(text: str) -> dict:
    """
    Main profile extraction entry point.

    Primary:
        LLM semantic understanding.

    Emergency fallback:
        Conservative Regex.

    A failure in profile extraction must NEVER crash /chat.
    """

    try:
        return extract_profile_with_llm(
            text
        )

    except Exception as exc:
        print(
            "[PROFILE] LLM parser failed. "
            "Using conservative fallback. "
            f"Reason: {type(exc).__name__}"
        )

        try:
            return extract_profile_fallback(
                text
            )

        except Exception as fallback_exc:
            print(
                "[PROFILE] Fallback parser failed. "
                f"Reason: {type(fallback_exc).__name__}"
            )

            return {
                "gpa": None,
                "completed_hours": None,
                "major": None,
            }


# =========================================================
# Onboarding State
# =========================================================

def get_missing_profile_fields(
    profile: dict,
) -> list[str]:

    missing = []

    if profile.get("gpa") is None:
        missing.append("gpa")

    if profile.get("completed_hours") is None:
        missing.append("completed_hours")

    if profile.get("major") is None:
        missing.append("major")

    return missing


def profile_is_ready(
    profile: dict,
) -> bool:

    return len(
        get_missing_profile_fields(
            profile
        )
    ) == 0


# =========================================================
# Language
# =========================================================

def contains_arabic(text: str) -> bool:
    return bool(
        re.search(
            r"[\u0600-\u06FF]",
            text,
        )
    )


# =========================================================
# Onboarding Messages
# =========================================================

def onboarding_message(
    profile: dict,
    user_text: str = "",
) -> str:

    missing = get_missing_profile_fields(
        profile
    )

    arabic = contains_arabic(
        user_text
    )

    # -----------------------------------------------------
    # Complete
    # -----------------------------------------------------

    if not missing:

        if arabic:
            return (
                "تمام، تم تسجيل بياناتك لهذه المحادثة. "
                "كيف أقدر أساعدك أكاديميًا؟"
            )

        return (
            "Thanks. Your academic information has been "
            "saved for this session. How can I help you?"
        )

    # -----------------------------------------------------
    # Arabic
    # -----------------------------------------------------

    if arabic:
        lines = []

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

    # -----------------------------------------------------
    # English / Other
    # -----------------------------------------------------

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


# =========================================================
# Major-Dependent Questions
# =========================================================

def question_requires_major(
    text: str,
) -> bool:

    q = text.lower()

    phrases = [
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
        for phrase in phrases
    )