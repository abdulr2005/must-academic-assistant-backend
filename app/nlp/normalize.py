"""Conservative normalization preserving course identifiers."""
import re
import unicodedata
COURSE = re.compile(r"(?<![\w.])([a-z]{2,5})[. -]?(\d{3})(?!\w)", re.I)
def normalize_course_codes(text: str) -> str:
    return COURSE.sub(lambda m: f"{m[1].upper()}.{m[2]}", text)
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹أإآى", "01234567890123456789اااي"))
    text = re.sub(r"[\u064b-\u065f\u0670ـ]", "", text).replace("٫", ".")
    return re.sub(r"\s+", " ", re.sub(r"[،؛؟?!,:;]", " ", text.lower())).strip()

