from typing import Dict, List
from threading import Lock


class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, List[dict]] = {}
        self._lock = Lock()

    def get_history(self, session_id: str) -> List[dict]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = []

            self._sessions[session_id].append({
                "role": role,
                "content": content,
            })

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions


session_store = SessionStore()