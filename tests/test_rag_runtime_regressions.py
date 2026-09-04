from types import SimpleNamespace

from app import rag_chain


def result(chunk_id, doc_type, major, text):
    return {
        "text": text,
        "metadata": {
            "chunk_id": chunk_id,
            "doc_type": doc_type,
            "major": major,
            "semester": 7,
            "confidence": "verified",
        },
    }


def test_ai_semester_plan_excludes_other_major_plans(monkeypatch):
    candidates = [
        result("common_policy", "faculty_regulation", "", "Common policy"),
        result("plan_AI_sem7", "semester_plan", "AI", "AI semester 7 plan"),
        result("plan_CS_sem7", "semester_plan", "CS", "CS semester 7 plan"),
        result("plan_IS_sem7", "semester_plan", "IS", "IS semester 7 plan"),
    ]
    monkeypatch.setattr(
        rag_chain,
        "_request_rag",
        lambda payload: {"results": candidates},
    )

    retrieved = rag_chain.retrieve_context(
        question="ايه مواد الترم السابع؟",
        top_k=3,
        profile={"major": "AI"},
    )

    assert retrieved["sources"] == ["plan_AI_sem7", "common_policy"]
    assert "plan_CS_sem7" not in retrieved["sources"]
    assert "plan_IS_sem7" not in retrieved["sources"]


def test_remaining_hours_rejects_gpa_articles_without_degree_total(monkeypatch):
    candidates = [
        result("gpa_article_3", "gpa_article", "", "GPA registration rule"),
        result("gpa_article_1", "gpa_article", "", "18 hour semester cap"),
        result("plan_AI_sem7", "semester_plan", "AI", "Semester total: 17"),
    ]
    monkeypatch.setattr(
        rag_chain,
        "_request_rag",
        lambda payload: {"results": candidates},
    )

    retrieved = rag_chain.retrieve_context(
        question="فاضلي كام ساعة للتخرج؟",
        top_k=3,
        profile={"major": "AI", "completed_hours": 106},
    )

    assert retrieved == {"context": [], "sources": []}


def test_authoritative_degree_total_is_preferred_if_added_later(monkeypatch):
    candidates = [
        result("gpa_article_1", "gpa_article", "", "18 hour semester cap"),
        result(
            "program_total_hours_AI",
            "graduation_requirements",
            "AI",
            "The AI degree requires a total of 140 credit hours for graduation.",
        ),
    ]
    monkeypatch.setattr(
        rag_chain,
        "_request_rag",
        lambda payload: {"results": candidates},
    )

    retrieved = rag_chain.retrieve_context(
        question="How many credit hours do I have left to graduate?",
        top_k=3,
        profile={"major": "AI", "completed_hours": 106},
    )

    assert retrieved["sources"] == ["program_total_hours_AI"]


def test_narrow_course_identity_adds_turn_specific_scope(monkeypatch):
    captured = {}

    class FakeLLM:
        def invoke(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(
                content="`AI.499` is Graduation Project II, worth 3 credit hours."
            )

    monkeypatch.setattr(rag_chain, "llm", FakeLLM())
    answer = rag_chain.generate_answer(
        question="What is AI.499?",
        context=[{
            "chunk_id": "course_AI.499",
            "doc_type": "course",
            "major": "AI",
            "semester": 8,
            "confidence": "verified",
            "text": "AI.499 is Graduation Project II. Prerequisite: AI.498.",
        }],
        history=[],
        profile={"major": "AI"},
    )

    human_prompt = captured["messages"][1][1]
    assert "<response_scope>" in human_prompt
    assert "only the course code, course name, and credit hours" in human_prompt
    assert "Do not mention prerequisites" in human_prompt
    assert "AI.498" not in answer


def test_prerequisite_followup_still_rewrites_and_answers(monkeypatch):
    responses = iter([
        "What is the prerequisite for AI.499?",
        "The prerequisite for `AI.499` is `AI.498`.",
    ])

    class FakeLLM:
        def invoke(self, messages):
            return SimpleNamespace(content=next(responses))

    monkeypatch.setattr(rag_chain, "llm", FakeLLM())
    history = [
        {"role": "user", "content": "What is AI.499?"},
        {"role": "assistant", "content": "It is Graduation Project II."},
    ]

    assert rag_chain.needs_conversation_context(
        "What is its prerequisite?"
    )
    standalone = rag_chain.rewrite_question(
        question="What is its prerequisite?",
        history=history,
    )
    assert standalone == "What is the prerequisite for AI.499?"

    answer = rag_chain.generate_answer(
        question="What is its prerequisite?",
        context=[{
            "chunk_id": "reg_AI_sem8_AI.499",
            "doc_type": "major_regulation_course",
            "major": "AI",
            "semester": 8,
            "confidence": "verified",
            "text": "The prerequisite for AI.499 is AI.498.",
        }],
        history=history,
        profile={"major": "AI"},
    )
    assert "AI.498" in answer


def test_registration_question_retains_common_gpa_chunks(monkeypatch):
    candidates = [
        result("gpa_article_3", "gpa_article", "", "Rule for GPA 2 to below 3"),
        result("gpa_article_1", "gpa_article", "", "General registration cap"),
        result("plan_CS_sem7", "semester_plan", "CS", "CS semester plan"),
    ]
    monkeypatch.setattr(
        rag_chain,
        "_request_rag",
        lambda payload: {"results": candidates},
    )

    retrieved = rag_chain.retrieve_context(
        question="How many credit hours can I register?",
        top_k=2,
        profile={"major": "AI", "gpa": 2.65},
    )

    assert retrieved["sources"] == ["gpa_article_3", "gpa_article_1"]
