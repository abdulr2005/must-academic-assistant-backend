from fastapi.testclient import TestClient

from app import api
from app.session_store import session_store
from app.llm import LLMServiceUnavailableError
from app.student_profile import (
    extract_profile_updates,
    extract_short_onboarding_reply,
)


def test_deterministic_profile_parser_examples():
    updates = extract_profile_updates(
        "My GPA is 2.65 and I have completed 106 credit hours"
    )
    assert updates["gpa"] == 2.65
    assert updates["completed_hours"] == 106

    updates = extract_profile_updates("My major is AI")
    assert updates["major"] == "AI"


def test_short_reply_and_registration_safety():
    profile = {
        "gpa": 2.65,
        "completed_hours": None,
        "major": None,
    }
    assert extract_short_onboarding_reply(
        "106 hours", profile
    )["completed_hours"] == 106
    assert extract_short_onboarding_reply(
        "106 sa3a", profile
    )["completed_hours"] == 106

    only_hours_missing = {
        "gpa": 2.65,
        "completed_hours": None,
        "major": "AI",
    }
    assert extract_short_onboarding_reply(
        "106", only_hours_missing
    )["completed_hours"] == 106

    for question in (
        "Can I register 18 hours?",
        "How many hours can I register?",
        "اقدر اسجل 18 ساعة؟",
        "كم ساعة اقدر اسجل؟",
    ):
        assert extract_profile_updates(question)["completed_hours"] is None
        assert extract_short_onboarding_reply(
            question, profile
        )["completed_hours"] is None


def test_chat_onboarding_and_llm_call_gating(monkeypatch):
    session_id = "onboarding-optimization-test"
    session_store.clear_session(session_id)
    client = TestClient(api.app)

    original_extract = api.extract_profile_updates
    calls = {"extract": 0, "rewrite": 0}

    def tracked_extract(text):
        calls["extract"] += 1
        return original_extract(text)

    def tracked_rewrite(question, history):
        calls["rewrite"] += 1
        return "What is the prerequisite for AI.499?"

    monkeypatch.setattr(api, "extract_profile_updates", tracked_extract)
    monkeypatch.setattr(api, "rewrite_question", tracked_rewrite)
    monkeypatch.setattr(
        api,
        "retrieve_context",
        lambda question, top_k, profile: {
            "context": [{
                "chunk_id": "course_AI.499",
                "doc_type": "course",
                "major": "AI",
                "semester": 8,
                "confidence": "verified",
                "text": "AI.499 has prerequisite AI.498.",
            }],
            "sources": ["course_AI.499"],
        },
    )
    monkeypatch.setattr(
        api,
        "generate_answer",
        lambda question, context, history, profile: "stub answer",
    )

    expected_profiles = [
        {"gpa": 2.65, "completed_hours": None, "major": None},
        {"gpa": 2.65, "completed_hours": 106, "major": None},
        {"gpa": 2.65, "completed_hours": 106, "major": "AI"},
    ]
    for question, expected in zip(
        ("My GPA is 2.65", "106 hours", "AI"),
        expected_profiles,
    ):
        response = client.post(
            "/chat",
            json={"session_id": session_id, "question": question},
        )
        assert response.status_code == 200
        body = response.json()
        for key, value in expected.items():
            assert body["profile"][key] == value

    assert body["onboarding_complete"] is True
    assert calls == {"extract": 3, "rewrite": 0}

    response = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "question": "How many credit hours can I register?",
        },
    )
    assert response.status_code == 200
    assert calls == {"extract": 3, "rewrite": 0}

    client.post(
        "/chat",
        json={"session_id": session_id, "question": "What is AI.499?"},
    )
    assert calls["rewrite"] == 0

    response = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "question": "What is its prerequisite?",
        },
    )
    assert response.status_code == 200
    assert response.json()["standalone_question"] == (
        "What is the prerequisite for AI.499?"
    )
    assert calls == {"extract": 3, "rewrite": 1}

    session_store.clear_session(session_id)


def test_chat_registration_question_does_not_set_completed_hours():
    session_id = "registration-safety-test"
    session_store.clear_session(session_id)


def _ready_profile(session_id):
    session_store.clear_session(session_id)
    session_store.update_profile(
        session_id,
        gpa=2.65,
        completed_hours=106,
        major="AI",
    )


def _temporary_llm_error():
    return LLMServiceUnavailableError([
        ("Groq", RuntimeError("429 rate limit")),
        ("Gemini", RuntimeError("429 RESOURCE_EXHAUSTED")),
    ])


def test_answer_generation_llm_unavailable_is_http_200(monkeypatch):
    session_id = "answer-llm-unavailable-test"
    _ready_profile(session_id)
    profile_before = session_store.get_profile(session_id)

    monkeypatch.setattr(
        api,
        "retrieve_context",
        lambda question, top_k, profile: {
            "context": [{
                "chunk_id": "course_AI.499",
                "doc_type": "course",
                "major": "AI",
                "semester": 8,
                "confidence": "verified",
                "text": "AI.499 is Graduation Project II.",
            }],
            "sources": ["course_AI.499"],
        },
    )
    monkeypatch.setattr(
        api,
        "generate_answer",
        lambda **kwargs: (_ for _ in ()).throw(
            _temporary_llm_error()
        ),
    )

    response = TestClient(api.app).post(
        "/chat",
        json={"session_id": session_id, "question": "What is AI.499?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["service_status"] == "llm_temporarily_unavailable"
    assert body["sources"] == []
    assert body["profile"] == profile_before
    assert body["standalone_question"] == "What is AI.499?"
    assert session_store.get_history(session_id) == []
    session_store.clear_session(session_id)


def test_answer_generation_llm_unavailable_uses_arabic_message(monkeypatch):
    session_id = "answer-llm-unavailable-ar-test"
    _ready_profile(session_id)
    monkeypatch.setattr(
        api,
        "retrieve_context",
        lambda **kwargs: {
            "context": [{
                "chunk_id": "course_AI.499",
                "doc_type": "course",
                "major": "AI",
                "semester": 8,
                "confidence": "verified",
                "text": "AI.499 is Graduation Project II.",
            }],
            "sources": ["course_AI.499"],
        },
    )
    monkeypatch.setattr(
        api,
        "generate_answer",
        lambda **kwargs: (_ for _ in ()).throw(
            _temporary_llm_error()
        ),
    )

    response = TestClient(api.app).post(
        "/chat",
        json={"session_id": session_id, "question": "ما هي مادة AI.499؟"},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == (
        "خدمة الذكاء الاصطناعي مشغولة مؤقتًا. "
        "حاول مرة أخرى بعد قليل."
    )
    assert response.json()["service_status"] == (
        "llm_temporarily_unavailable"
    )
    session_store.clear_session(session_id)


def test_rewrite_llm_unavailable_is_http_200_without_guess(monkeypatch):
    session_id = "rewrite-llm-unavailable-test"
    _ready_profile(session_id)
    session_store.add_message(
        session_id, "user", "What is AI.499?"
    )
    session_store.add_message(
        session_id, "assistant", "It is Graduation Project II."
    )
    history_before = session_store.get_history(session_id)
    monkeypatch.setattr(
        api,
        "rewrite_question",
        lambda **kwargs: (_ for _ in ()).throw(
            _temporary_llm_error()
        ),
    )

    response = TestClient(api.app).post(
        "/chat",
        json={
            "session_id": session_id,
            "question": "What is its prerequisite?",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["service_status"] == "llm_temporarily_unavailable"
    assert body["sources"] == []
    assert "standalone_question" not in body
    assert session_store.get_history(session_id) == history_before
    session_store.clear_session(session_id)


def test_unexpected_programming_error_stays_http_500(monkeypatch):
    session_id = "unexpected-programming-error-test"
    _ready_profile(session_id)
    monkeypatch.setattr(
        api,
        "retrieve_context",
        lambda **kwargs: (_ for _ in ()).throw(
            AttributeError("application bug")
        ),
    )

    response = TestClient(api.app).post(
        "/chat",
        json={"session_id": session_id, "question": "What is AI.499?"},
    )
    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Internal agent error during rag_retrieval."
    )
    session_store.clear_session(session_id)


def test_successful_llm_path_still_reports_ok(monkeypatch):
    session_id = "successful-llm-path-test"
    _ready_profile(session_id)
    monkeypatch.setattr(
        api,
        "retrieve_context",
        lambda **kwargs: {
            "context": [{
                "chunk_id": "course_AI.499",
                "doc_type": "course",
                "major": "AI",
                "semester": 8,
                "confidence": "verified",
                "text": "AI.499 is Graduation Project II.",
            }],
            "sources": ["course_AI.499"],
        },
    )
    monkeypatch.setattr(
        api,
        "generate_answer",
        lambda **kwargs: "Graduation Project II.",
    )

    response = TestClient(api.app).post(
        "/chat",
        json={"session_id": session_id, "question": "What is AI.499?"},
    )
    assert response.status_code == 200
    assert response.json()["service_status"] == "ok"
    session_store.clear_session(session_id)
    response = TestClient(api.app).post(
        "/chat",
        json={
            "session_id": session_id,
            "question": "Can I register 18 hours?",
        },
    )
    assert response.status_code == 200
    assert response.json()["profile"]["completed_hours"] is None
    session_store.clear_session(session_id)
