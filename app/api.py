from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .rag_chain import (
    retrieve_context,
    generate_answer,
    rewrite_question,
    needs_conversation_context,
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
        # Get conversation history for this session only
        history = session_store.get_history(req.session_id)

        # If the question depends on previous conversation context
        # but this session has no history, ask for clarification.
        if not history and needs_conversation_context(req.question):
            answer = "Which course or subject do you mean?"

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
                "history_size": 2,
            }

        # Rewrite follow-up questions using conversation history
        standalone_question = rewrite_question(
            question=req.question,
            history=history,
        )

        # Retrieve academic context using the standalone question
        retrieved = retrieve_context(
            question=standalone_question,
            top_k=3,
        )

        context = retrieved["context"]

        # Generate final answer
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
            )

        # Save user message
        session_store.add_message(
            req.session_id,
            role="user",
            content=req.question,
        )

        # Save assistant response
        session_store.add_message(
            req.session_id,
            role="assistant",
            content=answer,
        )

        return {
            "session_id": req.session_id,
            "question": req.question,

            # Temporary field for debugging query rewriting
            "standalone_question": standalone_question,

            "answer": answer,
            "sources": retrieved["sources"],
            "history_size": len(history) + 2,
        }

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