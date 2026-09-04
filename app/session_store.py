from typing import Dict, List, Optional
from threading import Lock


class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, dict] = {}
        self._lock = Lock()

    def _create_session(self) -> dict:
        """
        Create a new temporary student session.
        Everything here exists only while the session is alive.
        """
        return {
            "history": [],
            "profile": {
                "gpa": None,
                "completed_hours": None,
                "major": None,
                "completed_courses": [],
            },
        }

    def _ensure_session(self, session_id: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = self._create_session()

    def get_history(self, session_id: str) -> List[dict]:
        with self._lock:
            session = self._sessions.get(session_id)

            if not session:
                return []

            return list(session["history"])

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        with self._lock:
            self._ensure_session(session_id)

            self._sessions[session_id]["history"].append({
                "role": role,
                "content": content,
            })

    def get_profile(self, session_id: str) -> dict:
        with self._lock:
            session = self._sessions.get(session_id)

            if not session:
                return {
                    "gpa": None,
                    "completed_hours": None,
                    "major": None,
                    "completed_courses": [],
                }

            profile = session["profile"]

            return {
                "gpa": profile["gpa"],
                "completed_hours": profile["completed_hours"],
                "major": profile["major"],
                "completed_courses": list(
                    profile["completed_courses"]
                ),
            }

    def update_profile(
        self,
        session_id: str,
        gpa: Optional[float] = None,
        completed_hours: Optional[int] = None,
        major: Optional[str] = None,
        completed_courses: Optional[List[str]] = None,
    ) -> None:
        with self._lock:
            self._ensure_session(session_id)

            profile = self._sessions[session_id]["profile"]

            if gpa is not None:
                profile["gpa"] = gpa

            if completed_hours is not None:
                profile["completed_hours"] = completed_hours

            if major is not None:
                profile["major"] = major

            if completed_courses is not None:
                profile["completed_courses"] = list(
                    completed_courses
                )

    def profile_is_complete(self, session_id: str) -> bool:
        """
        Minimum information required before personalized
        academic advising can begin.
        """
        with self._lock:
            session = self._sessions.get(session_id)

            if not session:
                return False

            profile = session["profile"]

            return (
                profile["gpa"] is not None
                and profile["completed_hours"] is not None
            )

    def clear_session(self, session_id: str) -> None:
        """
        Delete conversation history and student profile.
        """
        with self._lock:
            self._sessions.pop(session_id, None)

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions


session_store = SessionStore()