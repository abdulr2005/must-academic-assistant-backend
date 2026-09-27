import re
from .normalize import normalize_text
from .entities import extract_major, extract_semester, extract_course_codes, has_phrase
from .vocabulary import SIGNALS, SPELLINGS
from .models import ParsedQuery
def parse_query(text: str, profile: dict | None = None) -> ParsedQuery:
    q = normalize_text(text)
    for old, new in SPELLINGS.items():
        q = re.sub(r"(?<!\w)" + re.escape(old) + r"(?!\w)", new, q)
    semester = extract_semester(q)
    codes = extract_course_codes(text)
    major = extract_major(text) or (profile or {}).get("major")
    signals = {intent: tuple(p for p in phrases if has_phrase(q, p)) for intent, phrases in SIGNALS.items()}
    if codes:
        intent = "PREREQUISITE" if signals["PREREQUISITE"] else "SPECIFIC_COURSE"
    elif signals["MAJOR_ELECTIVES"]:
        intent = "MAJOR_ELECTIVES"
    elif signals["PREREQUISITE"]:
        intent = "PREREQUISITE"
    elif signals["CORE_REQUIRED_COURSES"] or re.search(r"\b(?:required|mandatory)\s+(?:(?:ai|cs|is)\s+)?(?:major\s+)?(?:courses?|subjects?)\b", q):
        intent = "CORE_REQUIRED_COURSES"
    elif semester is not None and re.search(r"semester|sem|term|سمستر|ترم|فصل|مواد|courses|subjects|مقررات", q):
        intent = "SEMESTER_PLAN"
    elif signals["BROAD_MAJOR_CURRICULUM"] or (major and re.search(r"مواد|courses|subjects", q)):
        intent = "BROAD_MAJOR_CURRICULUM"
    elif signals["SPECIALIZATION"]:
        intent = "SPECIALIZATION"
    elif re.search(r"registration|register|maximum.*hours|max hours|how many (?:credit )?hours|اسجل|كم ساعة|كام ساعة|الحد الاقصي.*ساعات", q):
        intent = "REGISTRATION_HOURS"
    elif re.search(r"\b(?:its|it|that course|what about)\b|متطلباتها|ساعاتها", q):
        intent = "CONTEXTUAL_FOLLOWUP"
    else:
        intent = "UNKNOWN"
    return ParsedQuery(intent, major, semester, codes[0] if len(codes) == 1 else None, signals.get(intent, ()))

