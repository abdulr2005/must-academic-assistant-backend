import logging

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
)
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .academic_record_parser import (
    parse_academic_record,
)

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


# =========================================================
# Logging
# =========================================================

logger = logging.getLogger(__name__)


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="MUST Academic Assistant API",
    version="3.1.1",
)


# =========================================================
# Request Models
# =========================================================

class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=8,
    )

    question: str = Field(
        ...,
        min_length=2,
    )


# =========================================================
# Health
# =========================================================

@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "MUST Academic Assistant API",
        "version": "3.1.1",
        "vision": True,
    }


# =========================================================
# Chat
# =========================================================

@app.post("/chat")
def chat(req: ChatRequest):

    stage = "start"

    try:

        # =================================================
        # 1. Load session
        # =================================================

        stage = "load_session"

        history = session_store.get_history(
            req.session_id
        )

        profile = session_store.get_profile(
            req.session_id
        )


        # =================================================
        # 2. Extract profile information from text
        # =================================================

        stage = "extract_profile"

        updates = extract_profile_updates(
            req.question
        )


        if updates.get("gpa") is not None:

            session_store.update_profile(
                req.session_id,
                gpa=updates["gpa"],
            )


        if updates.get(
            "completed_hours"
        ) is not None:

            session_store.update_profile(
                req.session_id,
                completed_hours=updates[
                    "completed_hours"
                ],
            )


        if updates.get("major") is not None:

            session_store.update_profile(
                req.session_id,
                major=updates["major"],
            )


        profile = session_store.get_profile(
            req.session_id
        )


        # =================================================
        # 3. Onboarding
        # =================================================

        stage = "onboarding"

        if not profile_is_ready(
            profile
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
                "session_id":
                    req.session_id,

                "question":
                    req.question,

                "answer":
                    answer,

                "sources":
                    [],

                "profile":
                    profile,

                "onboarding_complete":
                    False,

                "history_size":
                    len(history) + 2,
            }


        # =================================================
        # 4. Profile-only message detection
        # =================================================

        stage = "profile_only_check"


        profile_data_was_supplied = (
            updates.get("gpa") is not None
            or updates.get(
                "completed_hours"
            ) is not None
            or updates.get("major") is not None
        )


        question_indicators = [
            "?",
            "؟",

            # English
            "what",
            "how",
            "can i",
            "may i",
            "which",
            "why",
            "when",
            "where",

            # Arabic
            "هل",
            "كام",
            "كم",
            "ايه",
            "إيه",
            "أقدر",
            "اقدر",
            "ينفع",
            "ليه",
            "لماذا",
            "متى",
            "فين",
            "أين",
        ]


        question_lower = (
            req.question
            .lower()
        )


        looks_like_question = any(
            indicator in question_lower
            for indicator
            in question_indicators
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
                "session_id":
                    req.session_id,

                "question":
                    req.question,

                "answer":
                    answer,

                "sources":
                    [],

                "profile":
                    profile,

                "onboarding_complete":
                    True,

                "history_size":
                    len(history) + 2,
            }


        # =================================================
        # 5. Ambiguous follow-up protection
        # =================================================

        stage = "followup_validation"


        if (
            not history
            and needs_conversation_context(
                req.question
            )
        ):

            if contains_arabic(
                req.question
            ):

                answer = (
                    "تقصد أي مادة أو موضوع؟"
                )

            else:

                answer = (
                    "Which course or subject "
                    "do you mean?"
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
                "session_id":
                    req.session_id,

                "question":
                    req.question,

                "standalone_question":
                    req.question,

                "answer":
                    answer,

                "sources":
                    [],

                "profile":
                    profile,

                "onboarding_complete":
                    True,

                "history_size":
                    len(history) + 2,
            }


        # =================================================
        # 6. Rewrite follow-up
        # =================================================

        stage = "rewrite_question"


        standalone_question = (
            rewrite_question(
                question=req.question,
                history=history,
            )
        )


        # =================================================
        # 7. RAG Retrieval
        # =================================================

        stage = "rag_retrieval"


        top_k = get_retrieval_top_k(
            standalone_question
        )


        try:

            retrieved = retrieve_context(
                question=standalone_question,
                top_k=top_k,
                profile=profile,
            )


        except RuntimeError as exc:

            logger.warning(
                "RAG temporarily unavailable | "
                "session_id=%s | error=%s",
                req.session_id,
                exc,
            )


            # =============================================
            # Graceful degradation
            # =============================================

            if contains_arabic(
                req.question
            ):

                answer = (
                    "خدمة المعلومات الأكاديمية "
                    "غير متاحة مؤقتًا. "
                    "حاول مرة ثانية بعد لحظات."
                )

            else:

                answer = (
                    "The academic knowledge service "
                    "is temporarily unavailable. "
                    "Please try again in a moment."
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
                "session_id":
                    req.session_id,

                "question":
                    req.question,

                "standalone_question":
                    standalone_question,

                "answer":
                    answer,

                "sources":
                    [],

                "profile":
                    profile,

                "onboarding_complete":
                    True,

                "history_size":
                    len(history) + 2,

                "service_status":
                    "rag_temporarily_unavailable",
            }


        context = retrieved[
            "context"
        ]


        # =================================================
        # 8. Generate answer
        # =================================================

        stage = "answer_generation"


        if not context:

            if contains_arabic(
                req.question
            ):

                answer = (
                    "المعلومات الأكاديمية المتاحة "
                    "غير كافية للإجابة على هذا "
                    "السؤال بدقة."
                )

            else:

                answer = (
                    "The available academic "
                    "information is not sufficient "
                    "to answer this question "
                    "accurately."
                )


        else:

            answer = generate_answer(
                question=req.question,
                context=context,
                history=history,
                profile=profile,
            )


        # =================================================
        # 9. Save conversation
        # =================================================

        stage = "save_conversation"


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


        # =================================================
        # 10. Response
        # =================================================

        stage = "response"


        return {
            "session_id":
                req.session_id,

            "question":
                req.question,

            "standalone_question":
                standalone_question,

            "answer":
                answer,

            "sources":
                retrieved["sources"],

            "profile":
                profile,

            "onboarding_complete":
                True,

            "history_size":
                len(history) + 2,

            "service_status":
                "ok",
        }


    except Exception as exc:

        logger.exception(
            "Unhandled /chat error | "
            "stage=%s | session_id=%s",
            stage,
            req.session_id,
        )


        raise HTTPException(
            status_code=500,
            detail=(
                "Internal agent error during "
                f"{stage}."
            ),
        ) from exc


# =========================================================
# Academic Record Upload
# =========================================================

@app.post(
    "/session/{session_id}/transcript"
)
async def upload_transcript(
    session_id: str,
    file: UploadFile = File(...),
):

    stage = "start"

    try:

        # =================================================
        # 1. Validate file type
        # =================================================

        stage = "validate_file_type"


        allowed_content_types = {
            "image/jpeg",
            "image/png",
            "application/pdf",
        }


        if (
            file.content_type
            not in allowed_content_types
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported file type. "
                    "Please upload JPG, "
                    "PNG, or PDF."
                ),
            )


        # =================================================
        # 2. Read file
        # =================================================

        stage = "read_file"


        file_bytes = await file.read()


        if not file_bytes:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Uploaded file is empty."
                ),
            )


        # =================================================
        # 3. Validate size
        # =================================================

        stage = "validate_file_size"


        max_size = (
            10
            * 1024
            * 1024
        )


        if len(file_bytes) > max_size:

            raise HTTPException(
                status_code=400,
                detail=(
                    "File is too large. "
                    "Maximum allowed size "
                    "is 10 MB."
                ),
            )


        # =================================================
        # 4. Ensure session exists
        # =================================================

        stage = "ensure_session"


        session_store.update_profile(
            session_id
        )


        # =================================================
        # 5. Vision extraction
        # =================================================

        stage = "vision_extraction"


        extracted = await run_in_threadpool(
            parse_academic_record,
            file_bytes,
            file.content_type,
        )


        # =================================================
        # 6. Validate parser contract
        # =================================================

        stage = "validate_extraction"


        if not isinstance(
            extracted,
            dict,
        ):

            raise RuntimeError(
                "Vision parser returned "
                "an invalid response type."
            )


        expected_fields = {
            "gpa",
            "completed_hours",
            "major",
            "completed_courses",
        }


        missing_keys = (
            expected_fields
            - set(extracted.keys())
        )


        if missing_keys:

            raise RuntimeError(
                "Vision parser response is "
                "missing required keys: "
                + ", ".join(
                    sorted(missing_keys)
                )
            )


        # =================================================
        # 7. Update student session profile
        # =================================================

        stage = "update_profile"


        update_kwargs = {}


        # GPA
        if extracted.get(
            "gpa"
        ) is not None:

            update_kwargs[
                "gpa"
            ] = extracted[
                "gpa"
            ]


        # Completed Hours
        if extracted.get(
            "completed_hours"
        ) is not None:

            update_kwargs[
                "completed_hours"
            ] = extracted[
                "completed_hours"
            ]


        # Major
        if extracted.get(
            "major"
        ) is not None:

            update_kwargs[
                "major"
            ] = extracted[
                "major"
            ]


        # Completed Courses
        completed_courses = (
            extracted.get(
                "completed_courses"
            )
        )


        if isinstance(
            completed_courses,
            list,
        ) and completed_courses:

            update_kwargs[
                "completed_courses"
            ] = completed_courses


        if update_kwargs:

            session_store.update_profile(
                session_id,
                **update_kwargs,
            )


        # =================================================
        # 8. Reload updated profile
        # =================================================

        stage = "reload_profile"


        profile = (
            session_store.get_profile(
                session_id
            )
        )


        # =================================================
        # 9. Determine onboarding state
        # =================================================

        stage = "profile_status"


        onboarding_complete = (
            profile_is_ready(
                profile
            )
        )


        # =================================================
        # 10. Response
        # =================================================

        stage = "response"


        return {
            "status":
                "processed",

            "session_id":
                session_id,

            "file": {
                "filename":
                    file.filename,

                "content_type":
                    file.content_type,

                "size_bytes":
                    len(file_bytes),
            },

            "extracted":
                extracted,

            "profile":
                profile,

            "onboarding_complete":
                onboarding_complete,

            "extraction_status":
                "completed",

            "service_status":
                "ok",
        }


    except HTTPException:
        raise


    except Exception as exc:

        logger.exception(
            "Unhandled academic record "
            "upload error | "
            "stage=%s | session_id=%s",
            stage,
            session_id,
        )


        raise HTTPException(
            status_code=500,
            detail=(
                "Internal academic record "
                "processing error during "
                f"{stage}."
            ),
        ) from exc


# =========================================================
# Academic Record Alias
# =========================================================

@app.post(
    "/session/{session_id}/academic-record"
)
async def upload_academic_record(
    session_id: str,
    file: UploadFile = File(...),
):
    """
    Preferred frontend endpoint for Academic Record upload.

    The older /transcript endpoint remains available
    for backward compatibility.
    """

    return await upload_transcript(
        session_id=session_id,
        file=file,
    )


# =========================================================
# End Session
# =========================================================

@app.delete(
    "/session/{session_id}"
)
def end_session(
    session_id: str,
):

    session_store.clear_session(
        session_id
    )


    return {
        "status":
            "ended",

        "session_id":
            session_id,
    }