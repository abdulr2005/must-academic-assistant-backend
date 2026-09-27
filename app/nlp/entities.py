import re
from .normalize import normalize_text, normalize_course_codes, COURSE
from .vocabulary import MAJORS, NUMBERS, SEMESTER
def has_phrase(text, phrase):
    return re.search(r"(?<!\w)" + re.escape(normalize_text(phrase)) + r"(?!\w)", text) is not None
def major_alias(text):
    q = normalize_text(text)
    return next((major for major, aliases in MAJORS.items() if q in tuple(map(normalize_text, aliases))), None)
def extract_major(text):
    # Course prefixes are not majors. Lowercase English 'is' is usually a verb.
    # Also exclude malformed code-like tokens; AI.49 must not imply AI major.
    raw = re.sub(r"(?<!\w)[a-z]{2,5}[. -]?\d+\w*", " ", text, flags=re.I)
    q = normalize_text(raw)
    found = set()
    for major, aliases in MAJORS.items():
        for alias in aliases:
            if alias == "is" and not (raw.strip().lower() == "is" or re.search(r"\bIS\b", raw) or re.search(r"(?:major|تخصص)\s+(?:is\s+)?is\b", q)):
                continue
            if has_phrase(q, alias):
                found.add(major)
    return next(iter(found)) if len(found) == 1 else None
def extract_semester(text):
    q = normalize_text(text)
    match = re.search(r"\b" + SEMESTER + r"\s+0?([1-8])(?!\d)", q)
    if match:
        return int(match[1])
    if re.fullmatch(r"0?[1-8]", q):
        return int(q)
    for number, words in enumerate(NUMBERS, 1):
        for word in words:
            if has_phrase(q, word) and (re.search(SEMESTER, q) or q == word or re.search(r"مواد|courses|subjects|مقررات", q)):
                return number
    return None
def extract_course_codes(text):
    return tuple(dict.fromkeys(m[0] for m in COURSE.finditer(normalize_course_codes(text))))

