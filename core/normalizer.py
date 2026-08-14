"""LLM response normalizer.

This module is the *only* place that knows about the raw text that LLMs
return.  Everything downstream (ui.py, render_service_card) receives a
clean, typed ``ServiceResponse`` and never touches raw model output.

Pipeline
--------
raw LLM text
  → normalize_line_breaks()   strip <br>, .br>, escaped \\n …
  → extract_json()            find the first {...} block in the text
  → json.loads()
  → _coerce_fields()          normalize_list() every list field
  → ServiceResponse           typed dataclass

All steps are tolerant: if any step fails ``normalize_response`` returns
``None`` and the caller falls back to plain markdown rendering.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def normalize_line_breaks(text: str) -> str:
    """Convert every line-break variant a model might emit to a real newline.

    Handles:
    ``<br>``, ``<br/>``, ``<br />``, ``<BR>``, ``<BR/>``, ``<BR />``
    ``.br>``, ``br>``, ``<br``,
    literal ``\\n`` (escaped backslash-n from JSON-in-text),
    and any mix of carriage-returns.
    """
    if not text:
        return text

    # 1. Literal escaped newline that some models write as \\n inside strings
    text = text.replace("\\n", "\n")

    # 2. All <br> variants (case-insensitive, with or without slash/space)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

    # 3. Truncated / malformed variants the vector-store content contains
    #    e.g.  ".br>", "br>", "<br" (without closing >)
    text = re.sub(r"\.br>", "\n", text)
    text = re.sub(r"\bbr>", "\n", text)
    text = re.sub(r"<br\b", "\n", text, flags=re.IGNORECASE)

    # 4. Windows-style carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 5. Collapse 3+ consecutive newlines to 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def normalize_list(value) -> List[str]:
    """Coerce any representation of a list that a model might return.

    Accepts:
    - Already a ``list`` → strip each item, drop empties, clean line-breaks
    - A ``str`` with numbered items  ``1. ... 2. ...``
    - A ``str`` with bullet items    ``• ... - ... * ...``
    - A ``str`` with ``<br>`` / ``\\n`` separators
    - ``None`` / empty              → ``[]``
    """
    if not value:
        return []

    if isinstance(value, list):
        result = []
        for item in value:
            cleaned = normalize_line_breaks(str(item)).strip()
            if cleaned:
                result.append(cleaned)
        return result

    # value is a string — normalise line-breaks first
    text = normalize_line_breaks(str(value)).strip()
    if not text:
        return []

    # Split on newlines (covers <br>-separated and \n-separated)
    lines = [ln.strip() for ln in text.splitlines()]

    # Remove leading numbering  "1." "1-" "١." "أ." etc.
    def _strip_marker(s: str) -> str:
        # Arabic and Latin numbered lists
        s = re.sub(r"^[\d٠-٩]+[\.\-\)]\s*", "", s)
        # Bullet characters
        s = re.sub(r"^[•\-\*▪️◦◆►]+\s*", "", s)
        return s.strip()

    result = [_strip_marker(ln) for ln in lines if ln]
    return [r for r in result if r]


def extract_json(text: str) -> Optional[dict]:
    """Extract the first valid JSON object from raw LLM output.

    Tolerates:
    - Markdown code fences  ```json ... ``` or ``` ... ```
    - Prose before/after the JSON block
    - Trailing commas (via a light pre-clean)
    - Single-quoted keys (via a light pre-clean)
    """
    if not text:
        return None

    # Strip code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)

    # Find the outermost { ... }
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        return None

    json_str = text[start : end + 1]

    # Light repairs
    # 1. Trailing commas before } or ]
    json_str = re.sub(r",\s*([\}\]])", r"\1", json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Structured response container
# ---------------------------------------------------------------------------

@dataclass
class ServiceResponse:
    """Model-independent container for a government service response."""

    service_name: str = ""
    description: str = ""
    conditions: List[str] = field(default_factory=list)
    required_documents: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    notes: str = ""
    support: str = ""
    similar_services: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """True when no meaningful content was extracted."""
        return not any(
            [
                self.service_name,
                self.description,
                self.conditions,
                self.required_documents,
                self.steps,
                self.notes,
                self.similar_services,
            ]
        )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

_LIST_FIELDS = ("conditions", "required_documents", "steps", "similar_services")
_STR_FIELDS  = ("service_name", "description", "notes", "support")

# Possible key aliases models might use (Arabic or English)
_FIELD_ALIASES: dict[str, list[str]] = {
    "service_name": [
        "service_name", "اسم الخدمة", "name", "الخدمة",
    ],
    "description": [
        "description", "وصف الخدمة", "وصف", "service_description",
    ],
    "conditions": [
        "conditions", "الشروط", "الشروط والأحكام", "الشروط والمتطلبات",
        "requirements", "متطلبات",
    ],
    "required_documents": [
        "required_documents", "المستندات المطلوبة", "المستندات",
        "documents", "الوثائق",
    ],
    "steps": [
        "steps", "خطوات التقديم", "خطوات", "الخطوات",
        "application_steps", "خطوات الاستخدام", "الخطوات اللازمة",
    ],
    "notes": [
        "notes", "ملاحظة", "ملاحظات", "note",
    ],
    "support": [
        "support", "الدعم", "دعم", "contact", "تواصل",
    ],
    "similar_services": [
        "similar_services", "خدمات مشابهة", "خدمات ذات صلة",
        "related_services",
    ],
}


def _resolve(raw: dict, canonical_key: str):
    """Return the value for *canonical_key* trying all known aliases."""
    for alias in _FIELD_ALIASES[canonical_key]:
        if alias in raw:
            return raw[alias]
    return None


def normalize_response(raw_text: str) -> Optional[ServiceResponse]:
    """Full normalisation pipeline.

    Returns a ``ServiceResponse`` if the model returned structured data,
    or ``None`` if the text is not a service card (e.g. a general greeting).
    """
    if not raw_text or not raw_text.strip():
        return None

    # Step 1 — clean line-break variants in the raw text first so JSON
    #           string values are clean before parsing.
    cleaned = normalize_line_breaks(raw_text)

    # Step 2 — extract JSON
    raw = extract_json(cleaned)
    if not raw or not isinstance(raw, dict):
        return None

    # Step 3 — coerce fields
    resp = ServiceResponse()

    for key in _STR_FIELDS:
        val = _resolve(raw, key)
        if val is not None:
            resp.__setattr__(key, normalize_line_breaks(str(val)).strip())

    for key in _LIST_FIELDS:
        val = _resolve(raw, key)
        if val is not None:
            resp.__setattr__(key, normalize_list(val))

    # Step 4 — reject if nothing meaningful came through
    if resp.is_empty():
        return None

    return resp
