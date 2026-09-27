"""Expected-field onboarding and explicit updates; never invokes a model."""
import re
from .normalize import normalize_text
from .entities import major_alias
FIELDS = ("gpa", "completed_hours", "major")
NUMBER = r"(-?\d+(?:\.\d+)?)"
def expected_field(profile):
    return next((field for field in FIELDS if profile.get(field) is None), None)
def parse_field(text, field):
    q = normalize_text(text)
    if re.search(r"\b(?:if|what|can|may|would|friend|his|her|لو|هل|اقدر|ينفع|اسجل)\b", q) or "?" in text or "؟" in text:
        return None
    if field == "major":
        q = re.sub(r"^(?:my major is|i study|تخصصي|انا)\s+", "", q)
        return major_alias(q)
    if field == "gpa":
        pattern = rf"(?:(?:my\s+)?(?:gpa|cgpa|معدلي|المعدل|mo3adaly)\s*(?:(?:is|now|هو|اتغير|ل|بتاعي|=)\s*)*)?{NUMBER}"
    elif field == "completed_hours":
        pattern = rf"(?:(?:i |ana |انا )?(?:have )?(?:completed|finished|earned|خلصت|مخلص|انهيت|5alast)\s+)?{NUMBER}\s*(?:credit hours?|credits?|hours?|hrs?|ساعة|ساعه|ساعات|sa3a|sa3at)?"
    else:
        return None
    match = re.fullmatch(pattern, q)
    if not match:
        return None
    value = float(match[1])
    if field == "gpa":
        return round(value, 2) if 0 <= value <= 4 else None
    return int(value) if value.is_integer() and 0 <= value <= 300 else None
def parse_onboarding(text, profile):
    result = dict.fromkeys(FIELDS)
    field = expected_field(profile)
    if field:
        result[field] = parse_field(text, field)
    return result
def explicit_profile_updates(text):
    q = normalize_text(text)
    result = dict.fromkeys(FIELDS)
    markers = {
        "gpa": r"^(?:my (?:gpa|cgpa)|معدلي|mo3adaly)\b",
        "completed_hours": r"^(?:i (?:have )?(?:completed|finished|earned)|انا (?:خلصت|مخلص)|خلصت|ana 5alast)\b",
        "major": r"^(?:my major is|تخصصي|i study)\b",
    }
    for field, marker in markers.items():
        if re.search(marker, q):
            result[field] = parse_field(text, field)
    return result

