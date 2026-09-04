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

"completed_hours" means the student's completed credit hours (ساعات معتمدة).

CRITICAL UNDERSTANDING OF HOURS / SYNONYMS:
Students frequently use:
"hours", "hour", "credit hours", "credits", "cr hrs", "ساعة", "ساعات", "ساعه"
ALL OF THESE MEAN COMPLETED CREDIT HOURS in this academic advising context!

Examples that MUST extract completed_hours:

"I completed 108 credit hours" -> completed_hours = 108
"108 hours" -> completed_hours = 108
"108 credit hours" -> completed_hours = 108
"108 credits" -> completed_hours = 108
"108" -> completed_hours = 108
"30 and ai" -> completed_hours = 30, major = "AI"
"30" -> completed_hours = 30
"30ساعه" or "30 ساعة" -> completed_hours = 30
"ساعاتي 30" or "ساعاتي 70" -> completed_hours = 30 or 70
"ساعاتي اللي خلصتها 108" -> completed_hours = 108
"خلصت 108 ساعة" -> completed_hours = 108
"مخلص 70 ساعة" -> completed_hours = 70
"أنا مخلص 90" -> completed_hours = 90
"I currently have 108 completed hours" -> completed_hours = 108
"my hours are 108" -> completed_hours = 108
"hours: 108" -> completed_hours = 108

IMPORTANT:
ONLY return completed_hours = null if the student is asking to REGISTER future hours:
Examples:
"Can I register 18 hours?" -> completed_hours = null (asking about future registration load)
"Can I take 21 credits?" -> completed_hours = null (asking about future registration load)
"How many hours can I register?" -> completed_hours = null (asking about registration limit)
"اقدر اسجل 18 ساعة؟" -> completed_hours = null
"كم ساعة اقدر اسجل؟" -> completed_hours = null


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
CONVERSATIONAL & SHORT REPLIES
============================================================

Students often answer onboarding questions with short, direct values:
- "30 and ai" -> {{"gpa": null, "completed_hours": 30, "major": "AI"}}
- "108" -> {{"gpa": null, "completed_hours": 108, "major": null}}
- "108 hours" -> {{"gpa": null, "completed_hours": 108, "major": null}}
- "30" -> {{"gpa": null, "completed_hours": 30, "major": null}}
- "30ساعه" or "30 ساعة" -> {{"gpa": null, "completed_hours": 30, "major": null}}
- "3.2" -> {{"gpa": 3.2, "completed_hours": null, "major": null}}
- "AI" -> {{"gpa": null, "completed_hours": null, "major": "AI"}}

Do NOT return null for these clear short values. An integer between 10 and 160 provided by the student in this onboarding context is their completed_hours.


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
        r"\b(?:gpa|cgpa)\s*(?:is|=|:)?\s*(\d(?:\.\d{1,2})?)\b",
        r"(?:معدلي|معدلى|المعدل(?:\s+التراكمي)?)\s*(?:هو|=|:)?\s*(\d(?:\.\d{1,2})?)",
    ]

    for pattern in gpa_patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        if match:
            result["gpa"] = validate_gpa(match.group(1))
            break

    if result["gpa"] is None:
        bare_gpa = re.fullmatch(r"\s*([0-4](?:\.\d{1,2})?)\s*", text)
        if bare_gpa:
            result["gpa"] = validate_gpa(bare_gpa.group(1))

    # -----------------------------------------------------
    # Completed Hours
    # -----------------------------------------------------
    # Exclude questions asking about future registration load (e.g. Can I register 18 hours?)
    is_reg_query = bool(
        re.search(
            r"(?:register|take|enroll\s*in|اسجل|أسجل|تسجيل|اخد|آخذ)\s*(?:more\s+than\s+)?(\d{1,3})",
            text,
            flags=re.IGNORECASE,
        )
    )

    if not is_reg_query:
        # Combined hours + major (e.g. "30 and ai", "30 و ai", "108 cs")
        comb = re.search(
            r"\b(\d{1,3})\s*(?:and|&|و)?\s*(ai|cs|is|general|ذكاء\s*اصطناعي|علوم\s*حاسب|نظم\s*معلومات)\b",
            text,
            flags=re.IGNORECASE,
        )
        if comb:
            result["completed_hours"] = validate_completed_hours(comb.group(1))
            result["major"] = validate_major(comb.group(2))

        if result["completed_hours"] is None:
            hours_patterns = [
                r"(?:completed|finished|passed|earned)\s*(\d{1,3})\s*(?:credit\s*hours?|credits?|hours?)?",
                r"(?:خلصت|مخلص|مخلّص|أنهيت|انهيت|اجتزت)\s*(\d{1,3})\s*(?:ساعة|ساعه|ساعات)?",
                r"(?:ساعات|ساعاتي|الساعات|ساعة|ساعه)\s*[:=]?\s*(\d{1,3})",
                r"\b(\d{1,3})\s*(?:credit\s*hours?|credits?|hours?|hrs?)\b",
                r"(\d{1,3})\s*(?:ساعة|ساعه|ساعات)",
            ]
            for pattern in hours_patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    result["completed_hours"] = validate_completed_hours(match.group(1))
                    break

        # Bare integer hours (e.g. "30", "108")
        if result["completed_hours"] is None:
            bare_num = re.fullmatch(r"\s*(\d{1,3})\s*", text)
            if bare_num:
                result["completed_hours"] = validate_completed_hours(bare_num.group(1))

    # -----------------------------------------------------
    # Major
    # -----------------------------------------------------
    if result["major"] is None:
        major_patterns = [
            (
                r"(?:my\s+major\s+is|تخصصي)\s*(?:ai|artificial intelligence|ذكاء اصطناعي|الذكاء الاصطناعي)",
                "AI",
            ),
            (
                r"(?:my\s+major\s+is|تخصصي)\s*(?:cs|computer science|علوم حاسب|علوم الحاسب)",
                "CS",
            ),
            (
                r"(?:my\s+major\s+is|تخصصي)\s*(?:is|information systems?|نظم معلومات|نظم المعلومات)",
                "IS",
            ),
            (
                r"(?:my\s+major\s+is|تخصصي)\s*(?:general|undeclared|عام|جنرال)",
                "General",
            ),
            (r"\b(ai|artificial intelligence|ذكاء\s*اصطناعي|الذكاء\s*الاصطناعي)\b", "AI"),
            (r"\b(cs|computer science|علوم\s*حاسب|علوم\s*الحاسب)\b", "CS"),
            (r"\b(is|information systems?|نظم\s*معلومات|نظم\s*المعلومات)\b", "IS"),
            (r"\b(general|undeclared|عام|جنرال)\b", "General"),
        ]

        for pattern, major in major_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
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


def is_neutral_input(text: str) -> bool:
    """True if input is purely digits, punctuation, or bare major tokens like '3.5', '30', 'ai'."""
    cleaned = re.sub(r"[\d\.\s\:\-\,\/\%]", "", text.lower())
    return cleaned in ("", "ai", "cs", "is", "general")


# =========================================================
# Onboarding Messages
# =========================================================

def onboarding_message(
    profile: dict,
    user_text: str = "",
    history_is_arabic: bool = False,
) -> str:

    missing = get_missing_profile_fields(
        profile
    )

    arabic = contains_arabic(
        user_text
    )

    # If user replies with neutral digits/codes (e.g. 3.5, 30, AI), preserve conversation language
    if not arabic and history_is_arabic and is_neutral_input(user_text):
        arabic = True

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
    # Arabic - Sequential One-by-One Onboarding
    # -----------------------------------------------------

    if arabic:
        if "gpa" in missing:
            if len(missing) == 3:
                return (
                    "أهلًا بك في المساعد الأكاديمي لـ MUST 👋\n"
                    "ممكن تبعت الـ GPA بتاعك؟"
                )
            return "تمام، ممكن تبعت الـ GPA بتاعك؟"

        if "completed_hours" in missing:
            return "تمام! كم عدد الساعات المعتمدة اللي خلصتها حتى الآن؟"

        if "major" in missing:
            return "تمام! إيه تخصصك الحالي: AI أو CS أو IS أو General؟"

    # -----------------------------------------------------
    # English / Other - Sequential One-by-One Onboarding
    # -----------------------------------------------------

    if "gpa" in missing:
        if len(missing) == 3:
            return (
                "Welcome to MUST Academic Assistant 👋\n"
                "Could you please provide your current cumulative GPA (e.g., 3.2)?"
            )
        return "Got it. Could you please provide your current GPA?"

    if "completed_hours" in missing:
        return "Thanks! How many credit hours have you completed so far?"

    if "major" in missing:
        return "Almost there! What is your current major (AI, CS, IS, or General)?"

    return "Your student profile is active! How can I assist you today?"


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