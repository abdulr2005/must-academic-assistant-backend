import requests
import uuid

BASE_URL = "http://127.0.0.1:8000"

session_id = f"demo_{uuid.uuid4().hex[:12]}"

print("MUST Academic Assistant")
print(f"Session: {session_id}")
print("Type 'exit' to end the conversation.\n")

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() in {"exit", "quit", "bye"}:
        try:
            requests.delete(
                f"{BASE_URL}/session/{session_id}",
                timeout=30,
            )
        except Exception:
            pass

        print("Assistant: Session ended.")
        break

    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={
                "session_id": session_id,
                "question": user_input,
            },
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()

        print("\nAssistant:", data.get("answer", "No answer returned."))

        profile = data.get("profile")
        if profile:
            print(
                f"[Profile: GPA={profile.get('gpa')}, "
                f"Hours={profile.get('completed_hours')}, "
                f"Major={profile.get('major')}]"
            )

        print()

    except requests.RequestException as exc:
        print(f"\n[ERROR] {exc}\n")