import base64
import logging
import re
from typing import Optional, List, Dict, Any

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .academic_record_parser import parse_academic_record
from .student_profile import (
    extract_profile_updates,
    onboarding_message,
    profile_is_ready,
    contains_arabic,
)
from .rag_chain import (
    retrieve_context,
    generate_answer,
    rewrite_question,
    needs_conversation_context,
    get_retrieval_top_k,
)
from .session_store import session_store


logger = logging.getLogger(__name__)


# ==================================
# FastAPI App & CORS
# ==================================

app = FastAPI(
    title="MUST Academic Assistant API",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================
# Request Models
# ==================================

class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=8,
    )

    question: Optional[str] = Field(
        default=None,
        description="Text question or prompt from the student.",
    )

    image_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded image or PDF string, or data URI (data:image/...;base64,...)",
    )

    image_content_type: Optional[str] = Field(
        default="image/png",
        description="MIME type of the image: image/jpeg, image/png, application/pdf",
    )


def resolve_chat_input(req: ChatRequest) -> tuple:
    """
    Differentiates between input types:
      - "text": Pure text question, no image.
      - "image": Pure image provided, no text question.
      - "multimodal": Both an image and a text question provided in the same turn.
    """
    image_bytes = None
    content_type = req.image_content_type or "image/png"
    text_question = req.question.strip() if req.question else None
    raw_img = req.image_base64

    # Support if question itself contains a data URL (e.g. data:image/png;base64,...)
    if text_question and text_question.startswith("data:image/"):
        raw_img = text_question
        text_question = None

    if raw_img:
        if raw_img.startswith("data:"):
            header, _, encoded = raw_img.partition(",")
            if ";" in header:
                mime_part = header.split(";")[0].replace("data:", "").strip()
                if mime_part:
                    content_type = mime_part
            raw_img = encoded

        try:
            image_bytes = base64.b64decode(raw_img)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 image data: {e}",
            )

    if image_bytes and text_question:
        return ("multimodal", image_bytes, content_type, text_question)
    elif image_bytes:
        return ("image", image_bytes, content_type, None)
    elif text_question:
        return ("text", None, content_type, text_question)
    else:
        raise HTTPException(
            status_code=400,
            detail="Please provide a text question or an academic record image.",
        )


# ==================================
# Health
# ==================================

@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "MUST Academic Assistant API",
        "version": "3.0.0",
    }


# ==================================
# Chat
# ==================================

@app.post("/chat")
def chat(req: ChatRequest):
    stage = "start"

    try:
        # 1. Load session state
        stage = "load_session"
        history = session_store.get_history(req.session_id)
        profile = session_store.get_profile(req.session_id)

        # 2. Differentiate Input Type (Text vs Image vs Multimodal)
        stage = "resolve_input"
        input_type, image_bytes, content_type, text_question = resolve_chat_input(req)

        extracted = None

        # 3. Vision Extraction (if image provided)
        if input_type in ("image", "multimodal"):
            stage = "parse_academic_image"
            extracted = parse_academic_record(
                file_bytes=image_bytes,
                content_type=content_type,
            )

            # Update student profile in session store with extracted metrics
            session_store.update_profile(
                req.session_id,
                gpa=extracted.get("gpa"),
                completed_hours=extracted.get("completed_hours"),
                major=extracted.get("major"),
                completed_courses=extracted.get("completed_courses"),
            )
            profile = session_store.get_profile(req.session_id)

            if input_type == "image":
                stage = "image_response"
                lines = [
                    "✅ **Academic Record Uploaded & Analyzed!**",
                    "",
                    f"• **Cumulative GPA**: {profile[gpa] if profile[gpa] is not None else "N/A"}",
                    f"• **Completed Hours**: {str(profile[completed_hours]) + " hrs" if profile[completed_hours] is not None else "N/A"}",
                    f"• **Major**: {profile[major] or "Not specified"}",
                ]
                if profile.get("completed_courses"):
                    courses_str = ", ".join(f"`{c}`" for c in profile["completed_courses"])
                    lines.append(f"• **Completed Courses**: {courses_str}")

                if profile_is_ready(profile):
                    lines.append("\nYour student session profile is now complete and active! How can I assist you with your courses, registration, or academic rules?")
                else:
                    missing = []
                    if profile.get("gpa") is None: missing.append("GPA")
                    if profile.get("completed_hours") is None: missing.append("Completed Hours")
                    if profile.get("major") is None: missing.append("Major")
                    lines.append(f"\nAlmost there! Please also provide: {", ".join(missing)}.")

                answer = "\n".join(lines)
                session_store.add_message(req.session_id, role="user", content="[Uploaded Academic Record Image]")
                session_store.add_message(req.session_id, role="assistant", content=answer)

                return {
                    "session_id": req.session_id,
                    "input_type": "image",
                    "question": None,
                    "answer": answer,
                    "sources": [],
                    "profile": profile,
                    "extracted": extracted,
                    "onboarding_complete": profile_is_ready(profile),
                    "history_size": len(history) + 2,
                }

        current_question = text_question

        # 4. Extract personal student data from text
        if input_type == "text":
            stage = "extract_profile"
            updates = extract_profile_updates(current_question)

            if updates.get("gpa") is not None:
                session_store.update_profile(req.session_id, gpa=updates["gpa"])
            if updates.get("completed_hours") is not None:
                session_store.update_profile(req.session_id, completed_hours=updates["completed_hours"])
            if updates.get("major") is not None:
                session_store.update_profile(req.session_id, major=updates["major"])

            profile = session_store.get_profile(req.session_id)

        # 5. Onboarding Check (Sequential 1-by-1 with language continuity)
        stage = "onboarding"
        history_is_arabic = any(
            contains_arabic(h.get("content", ""))
            for h in history
        )
        if not profile_is_ready(profile):
            answer = onboarding_message(
                profile=profile,
                user_text=current_question,
                history_is_arabic=history_is_arabic,
            )
            session_store.add_message(req.session_id, role="user", content=current_question)
            session_store.add_message(req.session_id, role="assistant", content=answer)

            return {
                "session_id": req.session_id,
                "input_type": input_type,
                "question": current_question,
                "answer": answer,
                "sources": [],
                "profile": profile,
                "extracted": extracted,
                "onboarding_complete": False,
                "history_size": len(history) + 2,
            }

        # 6. Detect profile-only message
        if input_type == "text":
            stage = "profile_only_check"
            profile_data_was_supplied = (
                updates.get("gpa") is not None
                or updates.get("completed_hours") is not None
                or updates.get("major") is not None
            )

            question_indicators = [
                "?", "؟",
                "what", "how", "can i", "may i", "which",
                "هل", "كام", "كم", "ايه", "إيه", "أقدر", "اقدر", "ينفع",
            ]
            question_lower = current_question.lower()
            looks_like_question = any(indicator in question_lower for indicator in question_indicators)

            if profile_data_was_supplied and not looks_like_question:
                answer = onboarding_message(
                    profile=profile,
                    user_text=current_question,
                    history_is_arabic=history_is_arabic,
                )
                session_store.add_message(req.session_id, role="user", content=current_question)
                session_store.add_message(req.session_id, role="assistant", content=answer)

                return {
                    "session_id": req.session_id,
                    "input_type": "text",
                    "question": current_question,
                    "answer": answer,
                    "sources": [],
                    "profile": profile,
                    "onboarding_complete": True,
                    "history_size": len(history) + 2,
                }

        # 7. Ambiguous follow-up protection
        stage = "followup_validation"
        if not history and needs_conversation_context(current_question):
            answer = "Which course or subject do you mean?"
            session_store.add_message(req.session_id, role="user", content=current_question)
            session_store.add_message(req.session_id, role="assistant", content=answer)

            return {
                "session_id": req.session_id,
                "input_type": input_type,
                "question": current_question,
                "standalone_question": current_question,
                "answer": answer,
                "sources": [],
                "profile": profile,
                "extracted": extracted,
                "onboarding_complete": True,
                "history_size": len(history) + 2,
            }

        # 8. Rewrite follow-up question
        stage = "rewrite_question"
        standalone_question = rewrite_question(
            question=current_question,
            history=history,
        )

        # 9. Retrieve RAG context
        stage = "rag_retrieval"
        top_k = get_retrieval_top_k(standalone_question)
        retrieved = retrieve_context(
            question=standalone_question,
            top_k=top_k,
            profile=profile,
        )
        context = retrieved["context"]

        # 10. Generate personalized answer
        stage = "answer_generation"
        if not context:
            answer = (
                "The available academic information is not sufficient "
                "to answer this question accurately."
            )
        else:
            answer = generate_answer(
                question=current_question,
                context=context,
                history=history,
                profile=profile,
            )

        if input_type == "multimodal" and extracted:
            header_note = (
                f"📄 *Academic record processed (GPA: {profile.get("gpa")}, "
                f"Hours: {profile.get("completed_hours")}, Major: {profile.get("major")})*\n\n"
            )
            answer = header_note + answer

        # 11. Save conversation
        stage = "save_conversation"
        user_msg = f"[With Academic Record Image] {current_question}" if input_type == "multimodal" else current_question
        session_store.add_message(req.session_id, role="user", content=user_msg)
        session_store.add_message(req.session_id, role="assistant", content=answer)

        # 12. Response
        stage = "response"
        return {
            "session_id": req.session_id,
            "input_type": input_type,
            "question": current_question,
            "standalone_question": standalone_question,
            "answer": answer,
            "sources": retrieved.get("sources", []),
            "profile": profile,
            "extracted": extracted,
            "onboarding_complete": True,
            "history_size": len(history) + 2,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled /chat error | stage=%s | session_id=%s", stage, req.session_id)
        raise HTTPException(
            status_code=500,
            detail=f"Internal agent error during {stage}.",
        ) from exc


# ==================================
# Academic Record Upload
# ==================================

async def _process_uploaded_transcript(session_id: str, file: UploadFile) -> dict:
    stage = "start"
    try:
        stage = "validate_type"
        allowed_content_types = {
            "image/jpeg",
            "image/jpg",
            "image/png",
            "application/pdf",
        }

        if file.content_type not in allowed_content_types:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload JPG, PNG, or PDF.",
            )

        stage = "read_file"
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        max_size = 10 * 1024 * 1024
        if len(file_bytes) > max_size:
            raise HTTPException(
                status_code=400,
                detail="File is too large. Maximum allowed size is 10 MB.",
            )

        stage = "parse_record"
        extracted = parse_academic_record(
            file_bytes=file_bytes,
            content_type=file.content_type,
        )

        stage = "update_profile"
        session_store.update_profile(
            session_id,
            gpa=extracted.get("gpa"),
            completed_hours=extracted.get("completed_hours"),
            major=extracted.get("major"),
            completed_courses=extracted.get("completed_courses"),
        )

        profile = session_store.get_profile(session_id)

        session_store.add_message(
            session_id,
            role="user",
            content=f"[Uploaded transcript: {file.filename}]",
        )
        session_store.add_message(
            session_id,
            role="assistant",
            content=(
                f"Extracted academic record: GPA={profile.get("gpa")}, "
                f"Hours={profile.get("completed_hours")}, Major={profile.get("major")}"
            ),
        )

        return {
            "status": "success",
            "session_id": session_id,
            "input_type": "image",
            "file": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(file_bytes),
            },
            "extracted": extracted,
            "profile": profile,
            "onboarding_complete": profile_is_ready(profile),
            "message": "Academic record parsed and profile updated successfully.",
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled transcript upload error | stage=%s | session_id=%s", stage, session_id)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error while processing uploaded file during {stage}.",
        ) from exc


@app.post("/session/{session_id}/transcript")
async def upload_transcript_path(
    session_id: str,
    file: UploadFile = File(...),
):
    return await _process_uploaded_transcript(session_id=session_id, file=file)


@app.post("/session/{session_id}/academic-record")
async def upload_academic_record_alias(
    session_id: str,
    file: UploadFile = File(...),
):
    """Alias endpoint for /transcript preferred by frontend."""
    return await _process_uploaded_transcript(session_id=session_id, file=file)


@app.post("/upload-record")
async def upload_transcript_form(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    return await _process_uploaded_transcript(session_id=session_id, file=file)


# ==================================
# End Session
# ==================================

@app.delete("/session/{session_id}")
def end_session(session_id: str):
    session_store.clear_session(session_id)
    return {
        "status": "ended",
        "session_id": session_id,
    }
