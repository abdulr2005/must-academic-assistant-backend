"""Regression coverage for semester-plan and specialization-rule routing.

Index audit:
- General semester 3 is authoritatively represented by ``plan_General_sem3``
  and ``general_sem3``.
- ``general_specialization_transition`` is the only qualifying specialization
  rule. It states selection occurs after semester 4; it does not establish a
  63-credit-hour specialization threshold.
"""

from app import rag_chain
from app.student_profile import extract_profile_updates


def result(chunk_id, doc_type, major, semester, text):
    return {
        "text": text,
        "metadata": {
            "chunk_id": chunk_id,
            "doc_type": doc_type,
            "major": major,
            "semester": semester,
            "confidence": "verified",
        },
    }


GENERAL_SEM3 = result(
    "plan_General_sem3",
    "semester_plan",
    "General",
    3,
    "General semester 3: CS.211, CS.212, MATH251, ECE202, and optional courses.",
)

SPECIALIZATION_RULE = result(
    "general_specialization_transition",
    "specialization_transition",
    "General",
    4,
    "After Semester 4, students choose CS, AI, or IS.",
)


def test_arabic_general_semester_3_retains_authoritative_plan(monkeypatch):
    candidates = [
        result("plan_AI_sem3", "semester_plan", "AI", 3, "Wrong AI plan"),
        GENERAL_SEM3,
        result("gpa_article_2", "gpa_article", "", None, "GPA registration rule"),
    ]
    monkeypatch.setattr(rag_chain, "_request_rag", lambda payload: {"results": candidates})

    retrieved = rag_chain.retrieve_context(
        question="طيب ايه هي مواد السمستر 3",
        top_k=3,
        profile={"major": "AI"},
    )

    assert retrieved["sources"] == ["plan_General_sem3"]
    assert retrieved["context"][0]["text"].startswith("General semester 3")


def test_english_general_semester_3_retains_authoritative_plan(monkeypatch):
    monkeypatch.setattr(
        rag_chain,
        "_request_rag",
        lambda payload: {"results": [GENERAL_SEM3]},
    )

    retrieved = rag_chain.retrieve_context(
        question="What are the semester 3 courses?",
        top_k=3,
        profile={"major": "General"},
    )

    assert retrieved["sources"] == ["plan_General_sem3"]


def test_ai_semester_7_still_excludes_other_major_plans(monkeypatch):
    candidates = [
        result("plan_CS_sem7", "semester_plan", "CS", 7, "CS plan"),
        result("plan_AI_sem7", "semester_plan", "AI", 7, "AI plan"),
        result("plan_IS_sem7", "semester_plan", "IS", 7, "IS plan"),
    ]
    monkeypatch.setattr(rag_chain, "_request_rag", lambda payload: {"results": candidates})

    retrieved = rag_chain.retrieve_context(
        question="What courses do I take in semester 7?",
        top_k=3,
        profile={"major": "AI"},
    )

    assert retrieved["sources"] == ["plan_AI_sem7"]


def test_arabic_specialization_routes_only_to_rule_chunk(monkeypatch):
    candidates = [
        result("gpa_article_2", "gpa_article", "", None, "Low-GPA load rule"),
        SPECIALIZATION_RULE,
    ]
    monkeypatch.setattr(rag_chain, "_request_rag", lambda payload: {"results": candidates})

    retrieved = rag_chain.retrieve_context(
        question="هل اقدر اتخصص وانا مخلص 62 ساعة من اصل 63 ساعة؟",
        top_k=3,
        profile={"major": "General", "completed_hours": 62},
    )

    assert retrieved["sources"] == ["general_specialization_transition"]
    assert "gpa_article_2" not in retrieved["sources"]


def test_english_specialization_routes_only_to_rule_chunk(monkeypatch):
    monkeypatch.setattr(
        rag_chain,
        "_request_rag",
        lambda payload: {"results": [SPECIALIZATION_RULE]},
    )

    retrieved = rag_chain.retrieve_context(
        question="I completed 62 credit hours. Can I specialize?",
        top_k=3,
        profile={"major": "General", "completed_hours": 62},
    )

    assert retrieved["sources"] == ["general_specialization_transition"]


def test_missing_authoritative_specialization_rule_preserves_safe_fallback(monkeypatch):
    monkeypatch.setattr(
        rag_chain,
        "_request_rag",
        lambda payload: {"results": [
            result("gpa_article_2", "gpa_article", "", None, "Registration rule"),
            result("practical_training", "practical_training", "Shared", None, "Training needs 63 hours"),
        ]},
    )

    retrieved = rag_chain.retrieve_context(
        question="What are the specialization requirements?",
        top_k=3,
        profile={"major": "General"},
    )

    assert retrieved == {"context": [], "sources": []}


def test_specialization_numeric_phrase_does_not_update_completed_hours():
    updates = extract_profile_updates(
        "هل اقدر اتخصص وانا مخلص 62 ساعة من اصل 63 ساعة؟"
    )
    assert updates["completed_hours"] is None


def test_registration_routing_remains_unchanged(monkeypatch):
    candidates = [
        result("gpa_article_3", "gpa_article", "", None, "GPA 2 to below 3"),
        result("gpa_article_1", "gpa_article", "", None, "General registration cap"),
        SPECIALIZATION_RULE,
    ]
    monkeypatch.setattr(rag_chain, "_request_rag", lambda payload: {"results": candidates})

    retrieved = rag_chain.retrieve_context(
        question="How many credit hours can I register?",
        top_k=2,
        profile={"major": "AI", "gpa": 2.65},
    )

    assert retrieved["sources"] == ["gpa_article_3", "gpa_article_1"]


def test_intent_detectors_do_not_change_course_identity_or_prerequisite_followup():
    assert rag_chain.is_narrow_course_identity_question("What is AI.499?")
    assert not rag_chain.is_specialization_requirements_question("What is AI.499?")
    assert rag_chain.needs_conversation_context("What is its prerequisite?")
