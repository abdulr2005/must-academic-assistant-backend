from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
)
from pydantic import BaseModel, Field

from .student_profile import (
    extract_profile_updates,
    onboarding_message,
    profile_is_ready,
)

from .rag_chain import (
    retrieve_context,
    generate_answer,
    rewrite_question,
    needs_conversation_context,
    get_retrieval_top_k,
)
from .session_store import session_store


app = FastAPI(
    title="MUST Academic Assistant API",
    version="3.0.0",
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=8)
    question: str = Field(..., min_length=2)


@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "MUST Academic Assistant API",
    }


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        # ==================================
        # 1. Load session state
        # ==================================
        history = session_store.get_history(
            req.session_id
        )

        profile = session_store.get_profile(
            req.session_id
        )

        # ==================================
        # 2. Extract personal student data
        # ==================================
        updates = extract_profile_updates(
            req.question
        )

        if updates.get("gpa") is not None:
            session_store.update_profile(
                req.session_id,
                gpa=updates["gpa"],
            )

        if updates.get("completed_hours") is not None:
            session_store.update_profile(
                req.session_id,
                completed_hours=updates["completed_hours"],
            )

        if updates.get("major") is not None:
            session_store.update_profile(
                req.session_id,
                major=updates["major"],
            )

        # Reload profile after updates
        profile = session_store.get_profile(
            req.session_id
        )

        # ==================================
        # 3. Onboarding
        # ==================================
        if not profile_is_ready(profile):

            answer = onboarding_message(
                profile=profile,
                user_text=req.question,
            )

            session_store.add_message(
                req.session_id,
                role="user",
                content=req.question,
            )

            session_store.add_message(
                req.session_id,
                role="assistant",
                content=answer,
            )

            return {
                "session_id": req.session_id,
                "question": req.question,
                "answer": answer,
                "sources": [],
                "profile": profile,
                "onboarding_complete": False,
                "history_size": len(history) + 2,
            }

        # ==================================
        # 4. If message only supplied profile data
        # ==================================
        profile_data_was_supplied = (
            updates.get("gpa") is not None
            or updates.get("completed_hours") is not None
            or updates.get("major") is not None
        )

        question_indicators = [
            "?",
            "؟",
            "what",
            "how",
            "can i",
            "may i",
            "which",
            "هل",
            "كام",
            "كم",
            "ايه",
            "إيه",
            "أقدر",
            "اقدر",
            "ينفع",
        ]

        looks_like_question = any(
            indicator in req.question.lower()
            for indicator in question_indicators
        )

        if (
            profile_data_was_supplied
            and not looks_like_question
        ):
            answer = onboarding_message(
                profile=profile,
                user_text=req.question,
            )

            session_store.add_message(
                req.session_id,
                role="user",
                content=req.question,
            )

            session_store.add_message(
                req.session_id,
                role="assistant",
                content=answer,
            )

            return {
                "session_id": req.session_id,
                "question": req.question,
                "answer": answer,
                "sources": [],
                "profile": profile,
                "onboarding_complete": True,
                "history_size": len(history) + 2,
            }

        # ==================================
        # 5. Ambiguous follow-up protection
        # ==================================
        if (
            not history
            and needs_conversation_context(
                req.question
            )
        ):
            answer = (
                "Which course or subject do you mean?"
            )

            session_store.add_message(
                req.session_id,
                role="user",
                content=req.question,
            )

            session_store.add_message(
                req.session_id,
                role="assistant",
                content=answer,
            )

            return {
                "session_id": req.session_id,
                "question": req.question,
                "standalone_question": req.question,
                "answer": answer,
                "sources": [],
                "profile": profile,
                "onboarding_complete": True,
                "history_size": len(history) + 2,
            }

        # ==================================
        # 6. Rewrite follow-up
        # ==================================
        standalone_question = rewrite_question(
            question=req.question,
            history=history,
        )

        # ==================================
        # 7. Retrieve RAG context
        # ==================================
        top_k = get_retrieval_top_k(
            standalone_question
        )

        retrieved = retrieve_context(
            question=standalone_question,
            top_k=top_k,
        )

        context = retrieved["context"]

        # ==================================
        # 8. Generate personalized answer
        # ==================================
        if not context:
            answer = (
                "The available academic information is not "
                "sufficient to answer this question accurately."
            )

        else:
            answer = generate_answer(
                question=req.question,
                context=context,
                history=history,
                profile=profile,
            )

        # ==================================
        # 9. Save conversation
        # ==================================
        session_store.add_message(
            req.session_id,
            role="user",
            content=req.question,
        )

        session_store.add_message(
            req.session_id,
            role="assistant",
            content=answer,
        )

        return {
            "session_id": req.session_id,
            "question": req.question,
            "standalone_question": standalone_question,
            "answer": answer,
            "sources": retrieved["sources"],
            "profile": profile,
            "onboarding_complete": True,
            "history_size": len(history) + 2,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc



@app.post("/session/{session_id}/transcript")
async def upload_transcript(
    session_id: str,
    file: UploadFile = File(...),
):
    try:
        # ==================================
        # 1. Validate file type
        # ==================================
        allowed_content_types = {
            "image/jpeg",
            "image/png",
            "application/pdf",
        }

        if file.content_type not in allowed_content_types:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported file type. "
                    "Please upload JPG, PNG, or PDF."
                ),
            )

        # ==================================
        # 2. Read uploaded file
        # ==================================
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        # ==================================
        # 3. Validate file size
        # ==================================
        max_size = 10 * 1024 * 1024  # 10 MB

        if len(file_bytes) > max_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    "File is too large. "
                    "Maximum allowed size is 10 MB."
                ),
            )

        # ==================================
        # 4. Ensure session exists
        # ==================================
        session_store.update_profile(
            session_id,
        )

        profile = session_store.get_profile(
            session_id
        )

        # ==================================
        # 5. Extraction comes next
        # ==================================
        return {
            "status": "uploaded",
            "session_id": session_id,
            "file": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(file_bytes),
            },
            "profile": profile,
            "extraction_status": "pending",
            "message": (
                "Transcript uploaded successfully. "
                "Profile extraction is not enabled yet."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc



@app.delete("/session/{session_id}")
def end_session(session_id: str):
    session_store.clear_session(session_id)

    return {
        "status": "ended",
        "session_id": session_id,
    }