from dataclasses import dataclass

@dataclass(frozen=True)
class ParsedQuery:
    intent: str = "UNKNOWN"
    major: str | None = None
    semester: int | None = None
    course_code: str | None = None
    matched_signals: tuple[str, ...] = ()
