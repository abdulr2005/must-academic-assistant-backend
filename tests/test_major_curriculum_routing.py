"""Focused regression tests for broad major/core curriculum routing.

The index has no consolidated AI-core chunk. The authoritative compact sources
are plan_AI_sem5, plan_AI_sem6, plan_AI_sem7, and plan_AI_sem8; individual
reg_AI_sem* and course_* core records are a fallback when plans are unavailable.
"""

from types import SimpleNamespace

from app import rag_chain


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


AI_PLANS = [
    result(f"plan_AI_sem{semester}", "semester_plan", "AI", semester, f"AI semester {semester} plan")
    for semester in range(5, 9)
]


def retrieve(monkeypatch, question, profile, candidates, top_k=None):
    captured = {}

    def fake_request(payload):
        captured.update(payload)
        return {"results": candidates}

    monkeypatch.setattr(rag_chain, "_request_rag", fake_request)
    effective_top_k = top_k or rag_chain.get_retrieval_top_k(question)
    response = rag_chain.retrieve_context(question, effective_top_k, profile)
    return response, captured


def test_arabic_ai_core_courses_prefers_ai_plans(monkeypatch):
    candidates = [
        result("plan_CS_sem5", "semester_plan", "CS", 5, "CS plan"),
        *AI_PLANS,
        result("plan_IS_sem5", "semester_plan", "IS", 5, "IS plan"),
    ]
    response, captured = retrieve(
        monkeypatch, "ممكن تقولي مواد الكور بتاعت AI؟", {"major": "AI"}, candidates
    )

    assert response["sources"] == [f"plan_AI_sem{n}" for n in range(5, 9)]
    assert captured["top_k"] == 8


def test_english_ai_core_courses_prefers_ai_plans(monkeypatch):
    response, _ = retrieve(
        monkeypatch, "What are the AI core courses?", {"major": "AI"}, AI_PLANS
    )
    assert response["sources"] == [f"plan_AI_sem{n}" for n in range(5, 9)]


def test_generic_major_curriculum_uses_profile_ai(monkeypatch):
    candidates = [
        result("plan_CS_sem5", "semester_plan", "CS", 5, "CS plan"),
        *AI_PLANS,
    ]
    response, captured = retrieve(
        monkeypatch,
        "What courses do I need for my major?",
        {"major": "AI"},
        candidates,
    )
    assert response["sources"] == [f"plan_AI_sem{n}" for n in range(5, 9)]
    assert "Target major curriculum: AI" in captured["question"]


def test_ai_curriculum_excludes_cs_and_is(monkeypatch):
    candidates = [
        result("plan_CS_sem6", "semester_plan", "CS", 6, "CS plan"),
        result("plan_IS_sem6", "semester_plan", "IS", 6, "IS plan"),
        AI_PLANS[1],
    ]
    response, _ = retrieve(
        monkeypatch, "What is the AI curriculum?", {"major": "AI"}, candidates
    )
    assert response["sources"] == ["plan_AI_sem6"]


def test_cs_curriculum_remains_isolated(monkeypatch):
    candidates = [
        AI_PLANS[0],
        result("plan_CS_sem5", "semester_plan", "CS", 5, "CS semester 5 plan"),
        result("plan_IS_sem5", "semester_plan", "IS", 5, "IS plan"),
    ]
    response, _ = retrieve(
        monkeypatch, "What are the required CS courses?", {"major": "CS"}, candidates
    )
    assert response["sources"] == ["plan_CS_sem5"]


def test_existing_ai_semester_7_route_is_not_broad_curriculum(monkeypatch):
    candidates = [
        result("plan_CS_sem7", "semester_plan", "CS", 7, "CS plan"),
        AI_PLANS[2],
    ]
    response, _ = retrieve(
        monkeypatch, "What courses do I take in semester 7?", {"major": "AI"}, candidates, 3
    )
    assert response["sources"] == ["plan_AI_sem7"]


def test_registration_route_keeps_small_gpa_window(monkeypatch):
    candidates = [
        result("gpa_article_3", "gpa_article", "", None, "GPA 2 to below 3"),
        result("gpa_article_1", "gpa_article", "", None, "General cap"),
        AI_PLANS[0],
    ]
    response, _ = retrieve(
        monkeypatch,
        "How many credit hours can I register?",
        {"major": "AI", "gpa": 2.65},
        candidates,
    )
    assert rag_chain.get_retrieval_top_k("How many credit hours can I register?") == 2
    assert response["sources"] == ["gpa_article_3", "gpa_article_1"]


def test_narrow_ai_499_keeps_default_window_and_scope():
    question = "What is AI.499?"
    assert not rag_chain.is_broad_major_curriculum_question(question)
    assert rag_chain.get_retrieval_top_k(question) == 3
    assert rag_chain.is_narrow_course_identity_question(question)


def test_arabic_ai_core_plan_fallback_excludes_elective_slots(monkeypatch):
    plan = result(
        "plan_AI_sem6",
        "semester_plan",
        "AI",
        6,
        "AI.343 (Neural Networks), EC(1) (Elective Course slot 1), AI.332 (Robotics)",
    )
    response, _ = retrieve(
        monkeypatch, "ممكن تقولي مواد الكور بتاعت AI؟", {"major": "AI"}, [plan]
    )
    assert response["sources"] == ["plan_AI_sem6"]
    assert "Required/non-elective courses" in response["context"][0]["text"]
    assert "EC(1)" not in response["context"][0]["text"]


def test_english_core_prefers_official_major_core_records(monkeypatch):
    candidates = [
        AI_PLANS[0],
        result("reg_AI_sem5_AI.301", "major_regulation_course", "AI", 5, "AI.301. Major Core."),
        result("elective_AI_AI.303", "elective_pool_course", "AI", None, "AI elective"),
        result("reg_CS_sem5_CS.301", "major_regulation_course", "CS", 5, "CS.301. Major Core."),
    ]
    response, _ = retrieve(
        monkeypatch, "What are the required AI courses?", {"major": "AI"}, candidates
    )
    assert response["sources"] == ["reg_AI_sem5_AI.301"]


def test_arabic_ai_electives_use_pool_not_ec_placeholders(monkeypatch):
    candidates = [
        result("elective_CS_CS.311", "elective_pool_course", "CS", None, "CS choice"),
        result("elective_AI_AI.303", "elective_pool_course", "AI", None, "AI.303 choice"),
        result("plan_AI_sem6", "semester_plan", "AI", 6, "EC(1) placeholder"),
        result("elective_IS_IS.343", "elective_pool_course", "IS", None, "IS choice"),
    ]
    response, _ = retrieve(
        monkeypatch,
        "ايه الـ Elective Courses بتاعت تخصصي؟",
        {"major": "AI"},
        candidates,
    )
    assert response["sources"] == ["elective_AI_AI.303"]


def test_english_ai_electives_are_isolated(monkeypatch):
    candidates = [
        result("elective_AI_AI.352", "elective_pool_course", "AI", None, "AI choice"),
        result("elective_CS_AI.464", "elective_pool_course", "CS", None, "CS choice"),
        result("elective_IS_IS.411", "elective_pool_course", "IS", None, "IS choice"),
    ]
    response, captured = retrieve(
        monkeypatch, "What are my AI elective courses?", {"major": "AI"}, candidates
    )
    assert response["sources"] == ["elective_AI_AI.352"]
    assert captured["top_k"] == 8


def test_cs_and_is_elective_pools_remain_separate(monkeypatch):
    candidates = [
        result("elective_CS_CS.311", "elective_pool_course", "CS", None, "CS choice"),
        result("elective_IS_IS.343", "elective_pool_course", "IS", None, "IS choice"),
    ]
    cs_response, _ = retrieve(
        monkeypatch, "What are my elective courses?", {"major": "CS"}, candidates
    )
    is_response, _ = retrieve(
        monkeypatch, "What are my elective courses?", {"major": "IS"}, candidates
    )
    assert cs_response["sources"] == ["elective_CS_CS.311"]
    assert is_response["sources"] == ["elective_IS_IS.343"]


def test_curriculum_intent_priority_keeps_specific_queries_narrow():
    assert rag_chain.classify_curriculum_intent("What is AI.499?") == "SPECIFIC_COURSE"
    assert rag_chain.classify_curriculum_intent(
        "What is the prerequisite for AI.499?"
    ) == "PREREQUISITE"
    assert rag_chain.classify_curriculum_intent(
        "What are my AI elective courses?"
    ) == "MAJOR_ELECTIVES"


def test_semester_remote_payload_respects_deployed_search_schema(monkeypatch):
    captured = {}

    def fake_request(payload):
        captured.update(payload)
        return {"results": [AI_PLANS[2]]}

    monkeypatch.setattr(rag_chain, "_request_rag", fake_request)
    rag_chain.retrieve_context(
        question="ايه مواد السمستر 7؟",
        top_k=3,
        profile={"major": "AI"},
    )

    assert set(captured) == {"question", "top_k"}
    assert isinstance(captured["question"], str)
    assert captured["question"] == "ايه مواد السمستر 7؟"
    assert isinstance(captured["top_k"], int)
    assert 1 <= captured["top_k"] <= 10
    assert captured["top_k"] == 10


def test_broad_curriculum_does_not_infer_progression_from_hours(monkeypatch):
    captured = {}

    class FakeLLM:
        def invoke(self, messages):
            captured["prompt"] = messages[1][1]
            return SimpleNamespace(content="AI curriculum from semesters 5 through 8.")

    monkeypatch.setattr(rag_chain, "llm", FakeLLM())
    context = [
        {
            "chunk_id": f"plan_AI_sem{semester}",
            "doc_type": "semester_plan",
            "major": "AI",
            "semester": semester,
            "confidence": "verified",
            "text": f"AI semester {semester} plan",
        }
        for semester in range(5, 9)
    ]

    rag_chain.generate_answer(
        question="What courses do I need for my major?",
        context=context,
        history=[],
        profile={
            "gpa": 2.65,
            "completed_hours": 106,
            "major": "AI",
            "completed_courses": [],
        },
    )

    prompt = captured["prompt"]
    assert "completed_hours: null" in prompt
    assert "completed_hours: 106" not in prompt
    assert "Do not infer the student's current level or semester" in prompt
    assert "Do not infer that earlier semesters or courses are completed" in prompt
    for semester in range(5, 9):
        assert f"AI semester {semester} plan" in prompt
