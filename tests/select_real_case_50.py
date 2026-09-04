"""Select, but never regenerate, the focused 50-case evaluation set."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "evaluation_cases.json"
OUTPUT = HERE / "real_case_50_cases.json"

IDS = [
    "COURSE_001", "COURSE_002", "COURSE_003", "COURSE_004",
    "COURSE_005", "COURSE_006", "COURSE_011", "COURSE_012",
    "COURSE_093", "COURSE_096",
    "PLAN_EN_01", "PLAN_AR_02", "PLAN_EN_04", "PLAN_AR_04", "PLAN_EN_05",
    "ELECTIVE_01", "ELECTIVE_03", "PREREQ_AR_03", "PREREQ_AR_06", "PREREQ_AR_08",
    "GPA_EN_01", "GPA_MIX_02", "GPA_EN_03", "GPA_MIX_05", "GPA_EN_06", "GPA_MIX_08",
    "GPA_MEMORY_01", "GPA_MEMORY_03",
    "ISOLATION_NEW_01", "ISOLATION_DELETE_01", "ISOLATION_CROSS_01",
    "TYPO_EN_01", "TYPO_EN_02", "TYPO_EN_03",
    "AR_EG_01", "AR_EG_02", "MIXED_01", "MIXED_02",
    "AMBIG_01", "AMBIG_03", "FALLBACK_01", "FALLBACK_02", "FALLBACK_03",
    "UNSUPPORTED_02", "UNSUPPORTED_07",
]

EXTRA = [
    {
        "id": "TRAINING_EN_01", "category": "practical_training", "language": "en",
        "conversation_type": "single_turn", "problem_pattern": "graduation requirements / practical training",
        "turns": [{"question": "How many practical training hours and weeks are required?"}],
        "expected": {"contains": ["90", "6"], "top_source_contains": "practical_training"},
        "support_status": "SUPPORTED",
    },
    {
        "id": "TRAINING_AR_01", "category": "practical_training", "language": "ar",
        "conversation_type": "single_turn", "problem_pattern": "هل التدريب العملي مطلوب للتخرج وما شروطه؟",
        "turns": [{"question": "ما شروط التدريب العملي وهل يُحسب في المعدل؟"}],
        "expected": {"contains": ["63", "0"], "contains_any": ["المعدل", "GPA"], "top_source_contains": "practical_training"},
        "support_status": "SUPPORTED",
    },
    {
        "id": "GRAD_PROJECT_EN_01", "category": "graduation_project", "language": "en",
        "conversation_type": "single_turn", "problem_pattern": "graduation project prerequisites",
        "turns": [{"question": "What is the prerequisite for AI.498 Graduation Project I?"}],
        "expected": {"contains": ["98"], "top_source_contains": "AI.498"},
        "support_status": "SUPPORTED",
    },
    {
        "id": "GRAD_PROJECT_AR_01", "category": "graduation_project", "language": "ar-eg",
        "conversation_type": "single_turn", "problem_pattern": "graduation project sequence",
        "turns": [{"question": "مشروع التخرج AI.499 لازم أكون واخد إيه قبله؟"}],
        "expected": {"contains": ["AI.498"], "top_source_contains": "AI.499"},
        "support_status": "SUPPORTED",
    },
    {
        "id": "FOLLOW_FOCUSED_01", "category": "follow_up", "language": "en",
        "conversation_type": "multi_turn", "problem_pattern": "course identity then prerequisite follow-up",
        "turns": [{"question": "What is AI.499?"}, {"question": "And what is its prerequisite?"}],
        "expected": {"turn_2_standalone_contains": "AI.499", "turn_2_answer_contains": "AI.498"},
        "support_status": "SUPPORTED",
    },
]

def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in source}
    missing = [case_id for case_id in IDS if case_id not in by_id]
    if missing:
        raise RuntimeError(f"Existing suite is missing selected IDs: {missing}")
    selected = [by_id[case_id] for case_id in IDS] + EXTRA
    if len(selected) != 50 or len({c['id'] for c in selected}) != 50:
        raise RuntimeError(f"Expected 50 unique cases, got {len(selected)}")
    OUTPUT.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected {len(selected)} cases into {OUTPUT}")

if __name__ == "__main__":
    main()
