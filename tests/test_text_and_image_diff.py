import base64
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.api import app, resolve_chat_input, ChatRequest
from app.session_store import session_store
from app.student_profile import (
    extract_profile_fallback,
    get_missing_profile_fields,
    onboarding_message,
    profile_is_ready,
)


@pytest.fixture
def client():
    return TestClient(app)


# =====================================================================
# 1. Unit Tests: Input Resolution & Differentiation
# =====================================================================

def test_resolve_pure_text_input():
    req = ChatRequest(
        session_id="test_sess_001",
        question="What is the prerequisite for AI.499?",
    )
    input_type, img_bytes, content_type, text_q = resolve_chat_input(req)
    assert input_type == "text"
    assert img_bytes is None
    assert text_q == "What is the prerequisite for AI.499?"


def test_resolve_pure_image_base64():
    fake_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    b64_str = base64.b64encode(fake_png).decode("utf-8")
    req = ChatRequest(
        session_id="test_sess_002",
        image_base64=b64_str,
        image_content_type="image/png",
    )
    input_type, img_bytes, content_type, text_q = resolve_chat_input(req)
    assert input_type == "image"
    assert img_bytes == fake_png
    assert content_type == "image/png"
    assert text_q is None


def test_resolve_pure_image_data_uri():
    fake_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    b64_str = base64.b64encode(fake_png).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64_str}"
    req = ChatRequest(
        session_id="test_sess_003",
        image_base64=data_uri,
    )
    input_type, img_bytes, content_type, text_q = resolve_chat_input(req)
    assert input_type == "image"
    assert img_bytes == fake_png
    assert content_type == "image/png"
    assert text_q is None


def test_resolve_multimodal_image_and_question():
    fake_jpg = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    b64_str = base64.b64encode(fake_jpg).decode("utf-8")
    req = ChatRequest(
        session_id="test_sess_004",
        question="What courses can I register based on this transcript?",
        image_base64=b64_str,
        image_content_type="image/jpeg",
    )
    input_type, img_bytes, content_type, text_q = resolve_chat_input(req)
    assert input_type == "multimodal"
    assert img_bytes == fake_jpg
    assert content_type == "image/jpeg"
    assert text_q == "What courses can I register based on this transcript?"


def test_resolve_empty_input_raises():
    req = ChatRequest(
        session_id="test_sess_005",
    )
    with pytest.raises(Exception):
        resolve_chat_input(req)


# =====================================================================
# 2. Student Profile Extraction: Hours / Synonyms / Limits vs Completed
# =====================================================================

def test_profile_fallback_hours_synonyms():
    # Credit hours equivalence: hour, hours, ساعة, ساعات, credits
    assert extract_profile_fallback("108 hours")["completed_hours"] == 108
    assert extract_profile_fallback("108 credit hours")["completed_hours"] == 108
    assert extract_profile_fallback("108 credits")["completed_hours"] == 108
    assert extract_profile_fallback("خلصت 108 ساعة")["completed_hours"] == 108
    assert extract_profile_fallback("30ساعه")["completed_hours"] == 30
    assert extract_profile_fallback("ساعاتي 70")["completed_hours"] == 70
    assert extract_profile_fallback("30 and ai") == {"gpa": None, "completed_hours": 30, "major": "AI"}


def test_profile_fallback_reg_load_not_completed():
    # Inquiries about registering future hours must NOT be parsed as completed hours!
    assert extract_profile_fallback("Can I register 18 hours?")["completed_hours"] is None
    assert extract_profile_fallback("Can I take 21 credits?")["completed_hours"] is None
    assert extract_profile_fallback("اقدر اسجل 18 ساعة؟")["completed_hours"] is None
    assert extract_profile_fallback("كم ساعة اقدر اسجل؟")["completed_hours"] is None


def test_profile_fallback_gpa_and_combined():
    # GPA and combined queries
    assert extract_profile_fallback("3.5")["gpa"] == 3.5
    assert extract_profile_fallback("معدلي 3.2")["gpa"] == 3.2
    combined = extract_profile_fallback("معدلي 3.5 ومخلص 30 ساعة وتخصصي ذكاء اصطناعي")
    assert combined["gpa"] == 3.5
    assert combined["completed_hours"] == 30
    assert combined["major"] == "AI"


# =====================================================================
# 3. Sequential Onboarding Steps (WhatsApp flow: GPA -> Hours -> Major)
# =====================================================================

def test_sequential_onboarding_arabic():
    profile = {"gpa": None, "completed_hours": None, "major": None}
    
    # Step 1: Nothing known -> asks for GPA
    msg1 = onboarding_message(profile, user_text="مرحبا")
    assert "GPA" in msg1
    assert "ممكن تبعت الـ GPA" in msg1

    # Student enters GPA: "3.5" in an Arabic session
    profile["gpa"] = 3.50
    # Step 2: GPA known -> asks for completed hours
    msg2 = onboarding_message(profile, user_text="3.5", history_is_arabic=True)
    assert "الساعات المعتمدة" in msg2

    # Student enters hours: "30" in an Arabic session
    profile["completed_hours"] = 30
    # Step 3: GPA and hours known -> asks for major
    msg3 = onboarding_message(profile, user_text="30", history_is_arabic=True)
    assert "تخصصك" in msg3
    assert "AI" in msg3

    # Student enters major: "AI" in an Arabic session
    profile["major"] = "AI"
    # Step 4: Ready
    assert profile_is_ready(profile)
    msg4 = onboarding_message(profile, user_text="AI", history_is_arabic=True)
    assert "تم تسجيل بياناتك" in msg4


# =====================================================================
# 4. End-to-End API Tests: Text vs Image vs Multimodal
# =====================================================================

def test_api_text_chat_sequential_onboarding(client):
    sess_id = "test_text_seq_001"
    session_store.clear_session(sess_id)

    # Turn 1: Student says "Hi" -> system asks for GPA
    res1 = client.post("/chat", json={"session_id": sess_id, "question": "Hi"})
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["input_type"] == "text"
    assert data1["onboarding_complete"] is False
    assert "GPA" in data1["answer"]

    # Turn 2: Student enters "3.5" -> system updates GPA and asks for hours
    res2 = client.post("/chat", json={"session_id": sess_id, "question": "3.5"})
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["input_type"] == "text"
    assert data2["profile"]["gpa"] == 3.5
    assert data2["onboarding_complete"] is False
    assert "hours" in data2["answer"].lower() or "credit hours" in data2["answer"].lower()

    # Turn 3: Student enters "30 hours" -> system updates hours and asks for major
    res3 = client.post("/chat", json={"session_id": sess_id, "question": "30 hours"})
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["profile"]["completed_hours"] == 30
    assert data3["onboarding_complete"] is False
    assert "major" in data3["answer"].lower()

    # Turn 4: Student enters "AI" -> onboarding completes
    res4 = client.post("/chat", json={"session_id": sess_id, "question": "AI"})
    assert res4.status_code == 200
    data4 = res4.json()
    assert data4["profile"]["major"] == "AI"
    assert data4["onboarding_complete"] is True


def test_api_image_upload_differentiates_and_completes_profile(client):
    sess_id = "test_image_upload_001"
    session_store.clear_session(sess_id)

    mock_parsed_record = {
        "gpa": 3.75,
        "completed_hours": 108,
        "major": "AI",
        "completed_courses": ["CS.101", "AI.201", "AI.301"],
    }

    fake_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    b64_str = base64.b64encode(fake_png).decode("utf-8")

    with patch("app.api.parse_academic_record", return_value=mock_parsed_record):
        # 1. Pure Image Chat via POST /chat
        res = client.post("/chat", json={
            "session_id": sess_id,
            "image_base64": b64_str,
            "image_content_type": "image/png",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["input_type"] == "image"
        assert data["question"] is None
        assert data["onboarding_complete"] is True
        assert data["profile"]["gpa"] == 3.75
        assert data["profile"]["completed_hours"] == 108
        assert data["profile"]["major"] == "AI"
        assert data["profile"]["completed_courses"] == ["CS.101", "AI.201", "AI.301"]
        assert "Academic Record Uploaded & Analyzed" in data["answer"]


def test_api_multimodal_image_with_question(client):
    sess_id = "test_multimodal_001"
    session_store.clear_session(sess_id)

    mock_parsed_record = {
        "gpa": 3.20,
        "completed_hours": 90,
        "major": "CS",
        "completed_courses": ["CS.101", "CS.201"],
    }

    fake_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    b64_str = base64.b64encode(fake_png).decode("utf-8")

    with patch("app.api.parse_academic_record", return_value=mock_parsed_record), \
         patch("app.api.generate_answer", return_value="`AI.499` is Graduation Project II for Artificial Intelligence (**3 credit hours**)."):
        res = client.post("/chat", json={
            "session_id": sess_id,
            "question": "What is AI.499?",
            "image_base64": b64_str,
            "image_content_type": "image/png",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["input_type"] == "multimodal"
        assert data["question"] == "What is AI.499?"
        assert data["onboarding_complete"] is True
        assert data["profile"]["gpa"] == 3.20
        assert data["profile"]["completed_hours"] == 90
        assert "Academic record processed" in data["answer"]


def test_api_multipart_transcript_upload(client):
    sess_id = "test_multipart_001"
    session_store.clear_session(sess_id)

    mock_parsed_record = {
        "gpa": 2.95,
        "completed_hours": 65,
        "major": "IS",
        "completed_courses": ["IS.101", "CS.101"],
    }

    fake_pdf = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

    with patch("app.api.parse_academic_record", return_value=mock_parsed_record):
        res = client.post(
            f"/session/{sess_id}/transcript",
            files={"file": ("transcript.pdf", fake_pdf, "application/pdf")},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["input_type"] == "image"
        assert data["profile"]["gpa"] == 2.95
        assert data["profile"]["completed_hours"] == 65
        assert data["profile"]["major"] == "IS"
        assert data["onboarding_complete"] is True


def test_api_text_chat_sequential_onboarding_arabic(client):
    sess_id = "test_text_seq_ar_001"
    session_store.clear_session(sess_id)

    # Turn 1: Student says "مرحبا" -> system asks for GPA in Arabic
    res1 = client.post("/chat", json={"session_id": sess_id, "question": "مرحبا"})
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["input_type"] == "text"
    assert data1["onboarding_complete"] is False
    assert "GPA" in data1["answer"]
    assert "ممكن تبعت الـ GPA" in data1["answer"]

    # Turn 2: Student enters "3.5" -> system updates GPA and asks for hours in Arabic
    res2 = client.post("/chat", json={"session_id": sess_id, "question": "3.5"})
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["input_type"] == "text"
    assert data2["profile"]["gpa"] == 3.5
    assert data2["onboarding_complete"] is False
    assert "الساعات المعتمدة" in data2["answer"]

    # Turn 3: Student enters "30" -> system updates hours and asks for major in Arabic
    res3 = client.post("/chat", json={"session_id": sess_id, "question": "30"})
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["profile"]["completed_hours"] == 30
    assert data3["onboarding_complete"] is False
    assert "تخصصك" in data3["answer"]

    # Turn 4: Student enters "AI" -> onboarding completes in Arabic
    res4 = client.post("/chat", json={"session_id": sess_id, "question": "AI"})
    assert res4.status_code == 200
    data4 = res4.json()
    assert data4["profile"]["major"] == "AI"
    assert data4["onboarding_complete"] is True
    assert "تم تسجيل بياناتك" in data4["answer"]


def test_prompt_language_switching_and_hours_instructions():
    from app.prompts import SYSTEM_PROMPT
    # Verify strict language switching instructions
    assert "LANGUAGE SWITCHING RULE (CRITICAL)" in SYSTEM_PROMPT
    assert "If <question> contains Arabic characters/words" in SYSTEM_PROMPT
    assert "YOU MUST RESPOND IN ARABIC" in SYSTEM_PROMPT
    assert "If <question> is in English" in SYSTEM_PROMPT
    assert "YOU MUST RESPOND IN ENGLISH" in SYSTEM_PROMPT
    assert "Do NOT retain the language of previous turns from <history>" in SYSTEM_PROMPT

    # Verify credit hours equivalence and remaining hours formula
    assert "CREDIT HOURS & REMAINING HOURS ARITHMETIC" in SYSTEM_PROMPT
    assert "فاضلي كام ساعة للتخرج؟" in SYSTEM_PROMPT
    assert "(Total Required Hours) - (completed_hours) = Remaining Hours" in SYSTEM_PROMPT

