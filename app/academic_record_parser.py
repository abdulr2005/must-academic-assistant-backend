"""
Academic Record Vision Parser Module
Converts Student Academic Record Images (JPG, PNG) and PDFs into Structured JSON.
"""

import os
import re
import json
import time
import base64
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv()

logger = logging.getLogger("academic_record_parser")

DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

EXTRACTION_PROMPT = """You are a specialized Academic Record Information Extraction Parser.
Your SOLE task is to read the attached academic transcript / grade record image and extract the student's academic metrics into a strictly typed JSON object.

REQUIRED FIELDS:
1. "gpa": float or null
   - Extract the Cumulative GPA (المعدل التراكمي) on a 0.0 - 4.0 scale.
   - DO NOT extract Semester GPA (معدل الفصل).
   - If missing, unreadable, or ambiguous, return null.

2. "completed_hours": integer or null
   - Extract the Total Earned / Completed / Passed Credit Hours (الساعات المكتسبة أو المنجزة أو المجتازة).
   - DO NOT extract registered hours (الساعات المسجلة) or remaining hours.
   - If missing, unreadable, or ambiguous, return null.

3. "major": string or null
   - Must be EXACTLY one of these allowed values: "AI", "CS", "IS", or "General".
   - Egyptian / Arabic mappings:
     * "ذكاء اصطناعي" -> "AI"
     * "علوم حاسب" -> "CS"
     * "نظم معلومات" -> "IS"
     * "عام" / "سنة أولى" -> "General"
   - STRICT RULE: DO NOT INFER OR GUESS THE MAJOR FROM COURSE CODES!
   - If the major is not explicitly stated in the text/header, return null.

4. "completed_courses": list of strings
   - A list of course codes that the student has definitively PASSED / COMPLETED (e.g., passing grades A, B, C, D, or "Pass"/"ناجح").
   - Format each code with standard uppercase prefix and number (e.g., ["CS.101", "AI.201", "MATH101"]).
   - DO NOT include failed courses (grade F), withdrawn courses (W), or currently in-progress courses (IP/Registered).
   - If no completed courses are visible or clear, return [].

OUTPUT FORMAT:
Return ONLY a valid, raw JSON object with exactly these 4 keys:
{
  "gpa": float or null,
  "completed_hours": integer or null,
  "major": string or null,
  "completed_courses": ["COURSE.101", ...]
}
Do NOT include any conversational preamble, explanations, or markdown code fences."""


def _clean_and_validate_output(raw_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and normalizes the parsed dictionary to guarantee strict contract."""
    # 1. Validate GPA
    gpa = raw_dict.get("gpa")
    clean_gpa: Optional[float] = None
    if gpa is not None:
        try:
            gpa_float = float(gpa)
            if 0.0 <= gpa_float <= 4.0:
                clean_gpa = round(gpa_float, 2)
        except (ValueError, TypeError):
            clean_gpa = None

    # 2. Validate Completed Hours
    hours = raw_dict.get("completed_hours")
    clean_hours: Optional[int] = None
    if hours is not None:
        try:
            hours_int = int(float(hours))
            if 0 <= hours_int <= 300:
                clean_hours = hours_int
        except (ValueError, TypeError):
            clean_hours = None

    # 3. Validate Major
    major = raw_dict.get("major")
    clean_major: Optional[str] = None
    if isinstance(major, str):
        major_upper = major.strip().upper()
        if "ARTIFICIAL" in major_upper or major_upper == "AI":
            clean_major = "AI"
        elif "COMPUTER" in major_upper or major_upper == "CS":
            clean_major = "CS"
        elif "INFORMATION" in major_upper or major_upper == "IS":
            clean_major = "IS"
        elif "GENERAL" in major_upper or major_upper == "عام":
            clean_major = "General"

    # 4. Validate Completed Courses
    courses = raw_dict.get("completed_courses")
    clean_courses: List[str] = []
    if isinstance(courses, list):
        for c in courses:
            if isinstance(c, str):
                c_clean = c.strip().upper()
                c_norm = re.sub(r"\s+", "", c_clean)
                if re.match(r"^[A-Z]{2,5}\.?\d{2,4}$", c_norm):
                    m = re.match(r"^([A-Z]{2,5})\.?(\d{2,4})$", c_norm)
                    if m:
                        clean_courses.append(f"{m.group(1)}.{m.group(2)}")
                    else:
                        clean_courses.append(c_norm)
                elif c_norm:
                    clean_courses.append(c_norm)

    return {
        "gpa": clean_gpa,
        "completed_hours": clean_hours,
        "major": clean_major,
        "completed_courses": clean_courses
    }


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extracts JSON object from model text output safely."""
    if not text:
        raise ValueError("Model returned empty text.")

    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        return json.loads(json_str)

    return json.loads(text)


def _convert_pdf_to_image_bytes(pdf_bytes: bytes) -> bytes:
    """Converts the first page of a PDF document to PNG image bytes in memory."""
    try:
        import pymupdf as fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) == 0:
            raise ValueError("PDF document is empty.")
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        return pix.tobytes("png")
    except ImportError:
        logger.warning("PyMuPDF is not installed; attempting raw bytes.")
        return pdf_bytes


def parse_academic_record_openrouter(
    image_bytes: bytes,
    mime_type: str,
    api_key: str,
    model: str
) -> Dict[str, Any]:
    """Calls OpenRouter Vision API with image payload, with automatic retry."""
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{b64_image}"

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://must.edu.eg",
        "X-Title": "MUST Academic Record Parser"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }
        ],
        "temperature": 0.0
    }

    last_err = None
    for attempt in range(2):
        try:
            with httpx.Client(timeout=45.0) as client:
                response = client.post(OPENROUTER_API_URL, json=payload, headers=headers)

            if response.status_code != 200:
                last_err = RuntimeError(f"OpenRouter API Error {response.status_code}: {response.text}")
                time.sleep(1)
                continue

            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                last_err = ValueError("OpenRouter returned empty choices list.")
                continue

            message = choices[0].get("message", {})
            message_content = message.get("content") or message.get("reasoning") or ""
            if not message_content.strip():
                last_err = ValueError("Model returned blank message content.")
                continue

            parsed_dict = _extract_json_from_text(message_content)
            return _clean_and_validate_output(parsed_dict)
        except Exception as e:
            last_err = e
            time.sleep(1)
            continue

    raise RuntimeError(f"OpenRouter Vision failed: {last_err}")


def parse_academic_record_gemini_fallback(
    image_bytes: bytes,
    mime_type: str,
    gemini_api_key: str,
    model: str = "gemini-3-flash-preview"
) -> Dict[str, Any]:
    """Calls Gemini Vision API as an immediate fallback or local alternative."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=gemini_api_key)
    candidates = ["gemini-3-flash-preview", "gemini-3.6-flash"]
    last_err = None
    for cand in candidates:
        try:
            response = client.models.generate_content(
                model=cand,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    EXTRACTION_PROMPT
                ],
                config=types.GenerateContentConfig(temperature=0.0)
            )
            parsed = _extract_json_from_text(response.text)
            return _clean_and_validate_output(parsed)
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Gemini Vision fallback failed: {last_err}")


def parse_academic_record(
    file_bytes: bytes,
    content_type: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point for Academic Record Vision Parser.
    Accepts JPG, PNG, PDF and returns structured profile dictionary.
    """
    if not file_bytes:
        return {
            "gpa": None,
            "completed_hours": None,
            "major": None,
            "completed_courses": []
        }

    normalized_mime = content_type.lower().strip()

    if "pdf" in normalized_mime:
        file_bytes = _convert_pdf_to_image_bytes(file_bytes)
        normalized_mime = "image/png"
    elif normalized_mime in ("image/jpg", "jpg"):
        normalized_mime = "image/jpeg"

    openrouter_key = api_key or os.getenv("OPENROUTER_API_KEY", "").strip()
    openrouter_model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    chosen_provider = provider.lower() if provider else ("openrouter" if openrouter_key else "gemini")

    if chosen_provider == "openrouter":
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment or passed to parse_academic_record.")
        try:
            return parse_academic_record_openrouter(
                image_bytes=file_bytes,
                mime_type=normalized_mime,
                api_key=openrouter_key,
                model=openrouter_model
            )
        except Exception as e:
            logger.warning(f"OpenRouter call failed: {e}. Checking for Gemini fallback...")
            if gemini_key:
                return parse_academic_record_gemini_fallback(
                    image_bytes=file_bytes,
                    mime_type=normalized_mime,
                    gemini_api_key=gemini_key
                )
            raise e

    elif chosen_provider == "gemini":
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY is not set in environment.")
        return parse_academic_record_gemini_fallback(
            image_bytes=file_bytes,
            mime_type=normalized_mime,
            gemini_api_key=gemini_key
        )
    else:
        raise ValueError(f"Unknown provider '{chosen_provider}'. Must be 'openrouter' or 'gemini'.")
