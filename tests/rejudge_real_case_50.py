"""Rejudge captured REAL_CASE_50 outputs after evaluator-only ground-truth corrections."""
import csv
import json
from pathlib import Path

from run_full_evaluation import judge, report_markdown

HERE = Path(__file__).resolve().parent
cases = json.loads((HERE / "real_case_50_cases.json").read_text(encoding="utf-8"))
result_path = HERE / "real_case_50_results.json"
raw = json.loads(result_path.read_text(encoding="utf-8"))
by_case = {c["id"]: c for c in cases}

for turn in raw["turns"]:
    case = by_case[turn["case_id"]]
    payload = {k: turn.get(k) for k in ("answer", "standalone_question", "sources", "history_size")}
    status, reason = judge(case, turn["turn_index"], payload, turn["http_status"])
    turn["status"], turn["failure_reason"] = status, reason
    # The unchanged server logs confirmed Gemini quota failures followed by Groq
    # fallback during this run. Clarification guards return before invoking an LLM.
    if not turn.get("sources") and turn.get("answer") == "Which course or subject do you mean?":
        turn["provider"] = "none (deterministic guard)"
    else:
        turn["provider"] = "Groq (inferred from server fallback log)"

for result in raw["cases"]:
    if result["support_status"] != "SUPPORTED":
        continue
    rows = [t for t in raw["turns"] if t["case_id"] == result["case_id"]]
    statuses = [t["status"] for t in rows]
    result["status"] = "FAIL" if "FAIL" in statuses else "MANUAL_REVIEW" if any(s in {"STYLE_CHECK", "NEEDS_MANUAL_REVIEW"} for s in statuses) else "PASS"
    result["failure_reason"] = "; ".join(t["failure_reason"] for t in rows if t["failure_reason"])
    result["expected"] = by_case[result["case_id"]]["expected"]

for result in raw["cases"]:
    result.setdefault("expected", by_case[result["case_id"]]["expected"])
raw["selected_cases"] = cases
result_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

fields = ["case_id","category","language","support_status","session_id","turn_index","question","http_status",
          "standalone_question","answer","sources","expected_result","actual_result","history_size","latency_ms",
          "provider","retries","status","failure_reason"]
with (HERE / "real_case_50_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
    for row in raw["turns"]:
        writer.writerow({**row, "sources": json.dumps(row["sources"], ensure_ascii=False),
                         "expected_result": json.dumps(by_case[row["case_id"]]["expected"], ensure_ascii=False),
                         "actual_result": row["answer"]})

report_markdown(cases, raw["turns"], raw["cases"], HERE / "REAL_CASE_50_REPORT.md")
print("Rejudged captured outputs and refreshed focused reports")
