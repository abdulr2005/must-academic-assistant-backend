"""Run deterministic end-to-end evaluations against a running MUST /chat API."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.prompts import FALLBACK_AR, FALLBACK_EN  # noqa: E402

CASES_PATH = Path(__file__).with_name("evaluation_cases.json")
DEFAULT_OUT = Path(__file__).with_name("evaluation_output")
TRANSIENT = {408, 425, 429, 500, 502, 503, 504}

def norm(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()

def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * pct
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (pos - lo)

def request(method, url, *, retries=2, **kwargs):
    last = None
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            response = requests.request(method, url, timeout=kwargs.pop("timeout", 120), **kwargs)
            latency = (time.perf_counter() - started) * 1000
            if response.status_code not in TRANSIENT or attempt == retries:
                return response, latency, attempt
            last = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            latency = (time.perf_counter() - started) * 1000
            last = str(exc)
            if attempt == retries:
                return None, latency, attempt
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last)

def judge(case, turn_index, payload, status_code):
    exp = case.get("expected", {})
    answer = str(payload.get("answer", ""))
    standalone = str(payload.get("standalone_question", ""))
    sources = payload.get("sources") or []
    haystack = norm(answer)
    failures = []
    if status_code != 200:
        return "FAIL", f"HTTP {status_code}"
    if case.get("support_status") != "SUPPORTED":
        return "UNSUPPORTED_DATA_REQUIREMENT", exp.get("missing_data", "required data unavailable")

    contains = exp.get("contains", [])
    for value in contains:
        if norm(value) not in haystack:
            failures.append(f"answer missing {value!r}")
    any_values = exp.get("contains_any", [])
    if any_values and not any(norm(v) in haystack for v in any_values):
        failures.append(f"answer contains none of {any_values!r}")
    for value in exp.get("excludes", []):
        if norm(value) in haystack or any(norm(value) in norm(s) for s in sources):
            failures.append(f"unexpected leaked entity {value!r}")
    for value in exp.get("forbidden", []):
        if norm(value) in haystack:
            failures.append(f"forbidden text present: {value!r}")
    if exp.get("top_source_contains"):
        target = norm(exp["top_source_contains"])
        if not sources or target not in norm(sources[0]):
            failures.append(f"top source does not contain {exp['top_source_contains']!r}")
    if "exact_fallback" in exp:
        wanted = FALLBACK_AR if exp["exact_fallback"] else FALLBACK_EN
        if answer.strip() != wanted:
            failures.append("answer does not exactly match required fallback")
    if exp.get("clarification"):
        markers = ["which", "what course", "mean", "أي مادة", "انهي مادة", "تقصد"]
        if not any(x in haystack for x in markers):
            failures.append("did not ask for clarification")
    if turn_index == 2:
        if exp.get("turn_2_standalone_contains") and norm(exp["turn_2_standalone_contains"]) not in norm(standalone):
            failures.append(f"standalone question missing {exp['turn_2_standalone_contains']!r}")
        if exp.get("turn_2_answer_contains") and norm(exp["turn_2_answer_contains"]) not in haystack:
            failures.append(f"follow-up answer missing {exp['turn_2_answer_contains']!r}")
    if case.get("category") == "session_isolation" and exp.get("after_context_clarification") and turn_index == 2:
        if not any(x in haystack for x in ["which", "what course", "mean", "أي مادة", "انهي مادة", "تقصد"]):
            failures.append("isolated session did not request clarification")
        for value in exp.get("excludes_after_context", []):
            if norm(value) in haystack or any(norm(value) in norm(s) for s in sources):
                failures.append(f"cross-session entity leaked: {value}")
    if turn_index == 3:
        if exp.get("turn_3_standalone_contains") and norm(exp["turn_3_standalone_contains"]) not in norm(standalone):
            failures.append("cross-session rewrite used wrong context")
        if exp.get("turn_3_answer_contains") and norm(exp["turn_3_answer_contains"]) not in haystack:
            failures.append(f"turn 3 answer missing {exp['turn_3_answer_contains']}")
        if exp.get("turn_3_excludes") and norm(exp["turn_3_excludes"]) in haystack:
            failures.append(f"turn 3 leaked {exp['turn_3_excludes']}")
    if failures:
        return "FAIL", "; ".join(failures)
    if case.get("judge") == "STYLE_CHECK":
        return "STYLE_CHECK", "rule-based style heuristic passed; manual review recommended"
    return "PASS", ""

def root_cause(result):
    reason = norm(result.get("failure_reason"))
    if result["status"] == "UNSUPPORTED_DATA_REQUIREMENT": return "unsupported data"
    if "http" in reason or "network" in reason or "timeout" in reason: return "timeout/network"
    if "standalone" in reason: return "query rewriting"
    if result["category"] in {"session_memory", "session_isolation"}: return "session memory"
    if "source" in reason: return "retrieval"
    if result["category"] in {"prompt_injection", "off_topic", "fallback"}: return "prompt behavior"
    if "answer missing" in reason: return "model generation"
    return "other"

def metric(case_results, categories=None, languages=None):
    selected = [r for r in case_results if r["support_status"] == "SUPPORTED"
                and (not categories or r["category"] in categories)
                and (not languages or r["language"] in languages)]
    passed = sum(r["status"] in {"PASS", "STYLE_CHECK"} for r in selected)
    return passed, len(selected), (100 * passed / len(selected) if selected else 0)

def esc(value):
    return str(value or "").replace("|", "\\|").replace("\n", "<br>")

def report_markdown(cases, turns, case_results, output):
    supported = [r for r in case_results if r["support_status"] == "SUPPORTED"]
    passes = sum(r["status"] in {"PASS", "STYLE_CHECK"} for r in supported)
    failures = sum(r["status"] == "FAIL" for r in supported)
    manual = sum(r["status"] in {"STYLE_CHECK", "NEEDS_MANUAL_REVIEW"} for r in supported)
    latencies = [r["latency_ms"] for r in turns if r.get("http_status") is not None]
    lines = ["# Full Agent Evaluation Report", "", "## Executive Summary", "",
        f"- Total test cases: **{len(cases)}**", f"- Total turns executed: **{len(turns)}**",
        f"- Cases executed successfully (all turns HTTP 200): **{sum(r['status'] not in {'UNSUPPORTED_DATA_REQUIREMENT'} and any(t['case_id']==r['case_id'] and t.get('http_status')==200 for t in turns) for r in case_results)}**",
        f"- Supported tests: **{len(supported)}**", f"- Unsupported-data cases: **{len(cases)-len(supported)}**",
        f"- Passes (including passed style heuristics): **{passes}**", f"- Failures: **{failures}**",
        f"- Needs manual review / style checks: **{manual}**",
        f"- Overall supported-case pass rate: **{(100*passes/len(supported) if supported else 0):.2f}%**", "",
        "## Metrics", "", "| Metric | Passed | Evaluated | Rate |", "|---|---:|---:|---:|"]
    specs = [
        ("Factual accuracy", None, None), ("Conversation-memory success", {"follow_up", "session_memory"}, None),
        ("Query-rewrite success", {"follow_up"}, None), ("Session-isolation success", {"session_isolation"}, None),
        ("Fallback accuracy", {"fallback"}, None), ("Arabic pass rate", None, {"ar"}),
        ("Egyptian Arabic pass rate", None, {"ar-eg"}), ("English pass rate", None, {"en"}),
        ("Mixed/code-switch pass rate", None, {"mixed"}), ("Typo robustness", {"typo"}, None),
        ("Prompt-injection defense", {"prompt_injection"}, None)]
    for name, cats, langs in specs:
        p, n, rate = metric(case_results, cats, langs)
        lines.append(f"| {name} | {p} | {n} | {rate:.2f}% |")
    lines += ["", "## Latency", "", f"- Average: **{statistics.mean(latencies) if latencies else 0:.1f} ms**",
        f"- Median: **{statistics.median(latencies) if latencies else 0:.1f} ms**",
        f"- p95: **{percentile(latencies, .95):.1f} ms**", "", "## Provider Usage", "",
        "Provider is reported only when exposed by the API response; otherwise it is `not_detectable`.", "",
        *[f"- {name}: **{count}** turn(s)" for name,count in sorted(Counter(t.get('provider','not_detectable') for t in turns).items())],
        "", "### Slowest 10 calls", "",
        "| Case | Turn | Latency (ms) |", "|---|---:|---:|"]
    for r in sorted(turns, key=lambda x: x["latency_ms"], reverse=True)[:10]:
        lines.append(f"| {r['case_id']} | {r['turn_index']} | {r['latency_ms']:.1f} |")
    lines += ["", "## Category Breakdown", ""]
    for category in sorted({r["category"] for r in case_results}):
        rows = [r for r in case_results if r["category"] == category]
        counts = Counter(r["status"] for r in rows)
        lines += [f"### {category}", "", f"Cases: {len(rows)}; " + ", ".join(f"{k}: {v}" for k,v in sorted(counts.items())), ""]
    lines += ["## Language Breakdown", "", "| Language | Cases | Passed | Failed | Unsupported |", "|---|---:|---:|---:|---:|"]
    for lang in sorted({r["language"] for r in case_results}):
        rows=[r for r in case_results if r["language"]==lang]
        lines.append(f"| {lang} | {len(rows)} | {sum(r['status'] in {'PASS','STYLE_CHECK'} for r in rows)} | {sum(r['status']=='FAIL' for r in rows)} | {sum(r['status']=='UNSUPPORTED_DATA_REQUIREMENT' for r in rows)} |")
    lines += ["", "## Every Test Case", ""]
    turn_map = defaultdict(list)
    for t in turns: turn_map[t["case_id"]].append(t)
    for c in cases:
        cr = next(r for r in case_results if r["case_id"] == c["id"])
        lines += [f"### {c['id']} — {cr['status']}", "", f"- Pattern: {c['problem_pattern']}",
            f"- Category / language: {c['category']} / {c['language']}", f"- Session: `{cr.get('session_id','not executed')}`",
            f"- Expected: `{json.dumps(c['expected'], ensure_ascii=False)}`"]
        if not turn_map[c["id"]]:
            lines += [f"- Question: {c['turns'][0]['question']}", f"- Result: {cr['failure_reason']}", ""]
        for t in turn_map[c["id"]]:
            lines += [f"- Turn {t['turn_index']} question: {t['question']}", f"- Standalone: {t.get('standalone_question','')}",
                f"- Sources: `{json.dumps(t.get('sources',[]), ensure_ascii=False)}`", f"- Provider: `{t.get('provider','not_detectable')}`", f"- Actual answer: {t.get('answer','')}",
                f"- Result: **{t['status']}** — {t.get('failure_reason','') or 'no failure'}", f"- Latency: {t['latency_ms']:.1f} ms", ""]
    groups=defaultdict(list)
    for r in case_results:
        if r["status"] in {"FAIL", "UNSUPPORTED_DATA_REQUIREMENT"}: groups[root_cause(r)].append(r)
    lines += ["## Failure Analysis", ""]
    for group in ["retrieval","query rewriting","session memory","prompt behavior","unsupported data","model generation","timeout/network","other"]:
        rows=groups.get(group,[]); lines += [f"### {group}", "", f"{len(rows)} case(s): " + (", ".join(r["case_id"] for r in rows) or "None."), ""]
    lines += ["## Unsupported Product Requirements", "", "| Original question | Missing data | Recommended future integration |", "|---|---|---|"]
    for c in cases:
        if c["support_status"] != "SUPPORTED":
            e=c["expected"]; lines.append(f"| {esc(c['turns'][0]['question'])} | {esc(e['missing_data'])} | {esc(e['recommended_integration'])} |")
    top = [r for r in case_results if r["status"] == "FAIL"][:10]
    lines += ["", "## Recommendations", "", "- **P0:** Investigate failed retrieval/top-source checks and exact fallback violations before a demo.",
        "- **P1:** Fix follow-up rewrite, dialect, and factual-generation failures shown above; rerun the same immutable corpus.",
        "- **P2:** Integrate SIS, timetable, registration, and fees services for unsupported product requirements.", "",
        "### Top 10 failures", ""] + [f"{i}. `{r['case_id']}` — {r['failure_reason']}" for i,r in enumerate(top,1)]
    output.write_text("\n".join(lines), encoding="utf-8")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--cases", type=Path, default=CASES_PATH); p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--include-unsupported", action="store_true", help="Call unsupported cases too (normally classified without wasting API calls)")
    p.add_argument("--limit", type=int); p.add_argument("--resume", action="store_true")
    p.add_argument("--report-only", action="store_true", help="Generate honest pending/unsupported reports without API calls")
    p.add_argument("--json-name", default="evaluation_results.json")
    p.add_argument("--csv-name", default="evaluation_results.csv")
    p.add_argument("--report-name", default="FULL_EVALUATION_REPORT.md")
    args=p.parse_args()
    cases=json.loads(args.cases.read_text(encoding="utf-8")); cases=cases[:args.limit] if args.limit else cases
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint=args.output_dir/"evaluation_checkpoint.json"
    turns=[]; case_results=[]
    if args.resume and checkpoint.exists():
        saved=json.loads(checkpoint.read_text(encoding="utf-8")); turns=saved.get("turns",[]); case_results=saved.get("cases",[])
    completed={r["case_id"] for r in case_results}
    if args.report_only:
        turns=[]; case_results=[]
        for c in cases:
            unsupported=c["support_status"] != "SUPPORTED"
            case_results.append({"case_id":c["id"],"category":c["category"],"language":c["language"],
                "support_status":c["support_status"],"status":"UNSUPPORTED_DATA_REQUIREMENT" if unsupported else "NEEDS_MANUAL_REVIEW",
                "failure_reason":c["expected"].get("missing_data", "not executed: upstream Gemini quota exhausted"),"session_id":"not executed"})
        raw={"generated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"base_url":args.base_url,
             "run_status":"BLOCKED_UPSTREAM_QUOTA","cases":case_results,"turns":turns}
        (args.output_dir/args.json_name).write_text(json.dumps(raw,ensure_ascii=False,indent=2),encoding="utf-8")
        fields=["case_id","category","language","support_status","session_id","turn_index","question","http_status","standalone_question","answer","sources","history_size","latency_ms","provider","retries","status","failure_reason"]
        with (args.output_dir/args.csv_name).open("w",encoding="utf-8-sig",newline="") as f:
            csv.DictWriter(f,fieldnames=fields).writeheader()
        report_markdown(cases,turns,case_results,args.output_dir/args.report_name)
        print(f"Pending reports written to {args.output_dir}"); return
    for ordinal,c in enumerate(cases,1):
        if c["id"] in completed:
            print(f"[{ordinal}/{len(cases)}] {c['id']}: already completed", flush=True); continue
        session=f"eval-{ordinal:04d}-{uuid.uuid4().hex[:12]}"
        if c["support_status"] != "SUPPORTED" and not args.include_unsupported:
            case_results.append({"case_id":c["id"],"category":c["category"],"language":c["language"],"support_status":c["support_status"],"status":"UNSUPPORTED_DATA_REQUIREMENT","failure_reason":c["expected"]["missing_data"],"session_id":"not executed"})
            checkpoint.write_text(json.dumps({"cases":case_results,"turns":turns},ensure_ascii=False,indent=2),encoding="utf-8"); continue
        statuses=[]; reasons=[]; sessions={"primary":session,"secondary":f"{session}-other"}; ti=0
        for turn in c["turns"]:
            active_session=sessions.get(turn.get("session", "primary"), session)
            if turn.get("action") == "delete":
                request("DELETE",f"{args.base_url.rstrip('/')}/session/{active_session}",timeout=30)
                continue
            ti += 1
            response,latency,retries=request("POST",f"{args.base_url.rstrip('/')}/chat",json={"session_id":active_session,"question":turn["question"]})
            code=response.status_code if response is not None else None
            try: payload=response.json() if response is not None else {}
            except ValueError: payload={"answer":response.text}
            status,reason=judge(c,ti,payload,code); statuses.append(status); reasons.append(reason)
            turns.append({"case_id":c["id"],"category":c["category"],"language":c["language"],"support_status":c["support_status"],"session_id":active_session,"turn_index":ti,"question":turn["question"],"http_status":code,"standalone_question":payload.get("standalone_question",""),"answer":payload.get("answer",payload.get("detail","")),"sources":payload.get("sources",[]),"history_size":payload.get("history_size"),"latency_ms":round(latency,2),"provider":payload.get("provider","not_detectable"),"retries":retries,"status":status,"failure_reason":reason})
        final="FAIL" if "FAIL" in statuses else "NEEDS_MANUAL_REVIEW" if "NEEDS_MANUAL_REVIEW" in statuses else "STYLE_CHECK" if "STYLE_CHECK" in statuses else "PASS"
        case_results.append({"case_id":c["id"],"category":c["category"],"language":c["language"],"support_status":c["support_status"],"status":final,"failure_reason":"; ".join(x for x in reasons if x),"session_id":session})
        request("DELETE",f"{args.base_url.rstrip('/')}/session/{session}",timeout=30)
        checkpoint.write_text(json.dumps({"cases":case_results,"turns":turns},ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"[{ordinal}/{len(cases)}] {c['id']}: {final}", flush=True)
    raw={"generated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"base_url":args.base_url,"cases":case_results,"turns":turns}
    (args.output_dir/args.json_name).write_text(json.dumps(raw,ensure_ascii=False,indent=2),encoding="utf-8")
    fields=["case_id","category","language","support_status","session_id","turn_index","question","http_status","standalone_question","answer","sources","history_size","latency_ms","provider","retries","status","failure_reason"]
    with (args.output_dir/args.csv_name).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for row in turns: w.writerow({**row,"sources":json.dumps(row["sources"],ensure_ascii=False)})
    report_markdown(cases,turns,case_results,args.output_dir/args.report_name)
    print(f"Reports written to {args.output_dir}")

if __name__ == "__main__": main()
