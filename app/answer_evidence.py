"""Exact evidence selection for queries that similarity windows cannot complete."""
import re

from .nlp import normalize_course_codes, normalize_text
from .nlp.entities import extract_course_codes
from .prompts import FALLBACK_AR, FALLBACK_EN


def response_language(question):
    q = normalize_text(question)
    if re.search(r"(?:answer|reply|respond|explain)\s+(?:in\s+)?english|بالانجليزي|بالانجليزية", q):
        return "en"
    if re.search(r"(?:answer|reply|respond|explain)\s+(?:in\s+)?arabic|بالعربي|بالعربية", q):
        return "ar"
    # Codes are identifiers, not a vote for English in mixed questions.
    q = re.sub(r"[a-z]{2,5}[. -]?\d{3}", "", q, flags=re.I)
    arabic = len(re.findall(r"[\u0621-\u064a]+", q))
    english = len(re.findall(r"[a-z]+", q))
    return "ar" if arabic and arabic >= english else "en"


def no_data_answer(question):
    return FALLBACK_AR if response_language(question) == "ar" else FALLBACK_EN


def core_catalogue_answer(question, context):
    """Render exact verified catalogue rows without inventing extra requirements."""
    if not context or any(c.get("doc_type") != "course" or c.get("confidence") != "verified"
                          or any(c.get(k) is None for k in ("course_code", "course_name_en", "credit_hours"))
                          for c in context):
        return None
    arabic = response_language(question) == "ar"
    heading = "المواد الأساسية للتخصص في السجلات الرسمية المتاحة:" if arabic else "Major core courses in the available official records:"
    unit = "ساعة معتمدة" if arabic else "credit hours"
    rows = [f"- `{c['course_code']}` — {c['course_name_en']} ({c['credit_hours']:g} {unit})" for c in context]
    return heading + "\n\n" + "\n".join(rows)


def as_result(chunk):
    metadata = dict(chunk.get("metadata") or {})
    metadata.update({key: chunk.get(key) for key in
                     ("chunk_id", "doc_type", "major", "semester", "confidence", "source_type")})
    return {"text": chunk.get("chunk_text", ""), "metadata": metadata}


def compatible_snapshot(results, chunks):
    """Do not combine local and remote revisions with conflicting source text."""
    local = {c.get("chunk_id"): c.get("chunk_text", "").strip() for c in chunks}
    return all(local.get((r.get("metadata") or {}).get("chunk_id"), r.get("text", "").strip())
               == r.get("text", "").strip() for r in results)


def complete_core(chunks, major):
    selected = []
    for c in chunks:
        if c.get("doc_type") != "course" or c.get("confidence") != "verified" or c.get("source_type") != "official":
            continue
        meta = c.get("metadata") or {}
        roles = meta.get("role_per_major") or {}
        if roles:
            role = next((v for k, v in roles.items() if k.upper().split()[0] == major), None)
        elif re.search(rf"\b{re.escape(major)}\b", str(c.get("major") or "").upper()):
            role = meta.get("category")
        else:
            role = None
        if role == "Major Core":
            selected.append(as_result(c))
    return selected


def registration_evidence(chunks):
    # Keep all authoritative registration articles together. Applicability and
    # numeric limits come from these sources, not policy constants in code.
    return [as_result(c) for c in chunks if c.get("doc_type") == "gpa_article"
            and c.get("source_type") == "official" and c.get("confidence") == "verified"]


def exact_courses(results, question):
    codes = set(extract_course_codes(question))
    selected = []
    for item in results:
        meta = item.get("metadata") or {}
        if meta.get("doc_type") not in {"course", "major_regulation_course", "elective_pool_course"}:
            continue
        code = normalize_course_codes(str(meta.get("course_code") or ""))
        if not code:
            match = re.search(r"([a-z]{2,5}\.\d{3})$", str(meta.get("chunk_id") or ""), re.I)
            code = match[1].upper() if match else ""
        if code in codes:
            selected.append(item)
    return selected


def specialization_hours_guard(question, context):
    """Explain missing numerical evidence without turning a user's number into policy."""
    q = normalize_text(question)
    if not re.search(r"\d+\s*(?:credit\s*)?(?:hours?|credits?|ساعة|ساعه|ساعات)", q):
        return None
    texts = " ".join(c.get("text", "") for c in context)
    # A future explicit numerical rule must continue through grounded generation.
    if re.search(r"\d+\s*(?:completed\s+)?(?:credit\s*)?(?:hours?|credits?|ساعة|ساعه|ساعات)", normalize_text(texts)):
        return None
    if response_language(question) == "ar":
        return ("المصادر المسترجعة لا تثبت حدًا عدديًا للساعات المعتمدة للتخصص. "
                "عدد الساعات المكتملة وحده لا يثبت إتمام فصل دراسي أو مقرراته، "
                "لذلك لا أستطيع تأكيد أهليتك للتخصص بهذه الأرقام. تأكد من المرشد الأكاديمي.")
    return ("The retrieved sources do not establish a numerical credit-hour threshold for specialization. "
            "Completed hours alone do not prove completion of a semester or its courses, "
            "so I cannot confirm your eligibility from those numbers. Please confirm with your academic advisor.")
