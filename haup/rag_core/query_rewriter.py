"""
File Summary:
Query expansion and reformulation for the HAUP RAG engine. Improves retrieval recall
by expanding a single user query into multiple variants using heuristic synonym maps
and optional LLM-based semantic rephrasings. Multiple query vectors cast a wider net
across the ChromaDB collection.

====================================================================
SYSTEM PIPELINE FLOW (Architecture + Object Interaction)
====================================================================

QueryRewriter()  [Class → Object]
||
├── __init__()  [Function] -------------------------------> Store llm_client and max_variations
│
└── expand()  [Function] ---------------------------------> Return deduplicated list of query variants
        │
        ├── Start with original query --------------------> Always first in result list
        │
        ├── _heuristic_expand()  [Function] --------------> Zero-latency synonym/pattern expansion
        │       │
        │       ├── Country name → code map --------------> "india" → "IN"
        │       ├── Active synonyms ----------------------> Append "is_active 1"
        │       ├── Inactive synonyms --------------------> Append "is_active 0"
        │       ├── Recent synonyms ----------------------> Append "created_at recent registration"
        │       ├── Email domain extraction --------------> "@gmail.com email"
        │       ├── Phone prefix extraction --------------> "phone_number +91"
        │       └── Strip question words -----------------> Remove "who is / show me / find" etc.
        │
        ├── [Conditional Branch] llm_client is not None -> _llm_expand()
        │       │
        │       ├── _llm_expand()  [Function] ------------> LLM semantic rephrasings (best-effort)
        │       │       │
        │       │       ├── Build structured prompt ------> Ask for 3 keyword-aligned alternatives
        │       │       ├── llm.complete() ---------------> Call LLM with low temperature
        │       │       └── [Early Exit Branch] failure --> Return empty list silently
        │       │
        │       └── [Exception Block] --------------------> Swallow LLM errors, continue
        │
        └── Deduplicate preserving order -----------------> Return at most max_variations + 1 strings

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_core.llm_client import LLMClient


# ─── Common abbreviation/synonym maps ────────────────────────────────────────

_COUNTRY_MAP = {
    "india":         "IN",
    "us":            "US", "usa": "US", "united states": "US", "america": "US",
    "uk":            "GB", "united kingdom": "GB", "britain": "GB",
    "germany":       "DE", "france": "FR", "japan": "JP", "china": "CN",
    "brazil":        "BR", "canada": "CA", "australia": "AU",
}

_ACTIVE_SYNONYMS   = {"active", "enabled", "live", "online", "current"}
_INACTIVE_SYNONYMS = {"inactive", "disabled", "deactivated", "banned", "suspended"}

_RECENT_SYNONYMS = {
    "recent", "recently", "latest", "new", "newest", "just joined",
    "just registered", "fresh", "last week", "last month", "today",
}


"""================= Startup class QueryRewriter ================="""
class QueryRewriter:
    """
    Produces a list of query strings from a single user input.
    The list always starts with the original query.
    """

    """================= Startup method __init__ ================="""
    def __init__(self, llm_client: "LLMClient | None" = None, max_variations: int = 3):
        self._llm = llm_client
        self._max = max_variations
    """================= End method __init__ ================="""

    """================= Startup method expand ================="""
    def expand(self, query: str) -> list[str]:
        """
        Returns [original, *heuristic_variants, *llm_variants] deduplicated.
        Always returns at least the original query.
        """
        variants: list[str] = [query]
        variants.extend(self._heuristic_expand(query))

        if self._llm is not None:
            try:
                llm_variants = self._llm_expand(query)
                variants.extend(llm_variants)
            except Exception:
                pass  # LLM expansion is best-effort

        # Deduplicate preserving order, return at most max+1 strings
        seen:   set[str]  = set()
        result: list[str] = []
        for v in variants:
            key = v.strip().lower()
            if key not in seen:
                seen.add(key)
                result.append(v.strip())
            if len(result) >= self._max + 1:
                break

        return result
    """================= End method expand ================="""

    """================= Startup method _heuristic_expand ================="""
    def _heuristic_expand(self, query: str) -> list[str]:
        q        = query.lower()
        variants: list[str] = []

        # Country name → country code
        for name, code in _COUNTRY_MAP.items():
            if name in q:
                replaced = re.sub(re.escape(name), code, q, flags=re.IGNORECASE)
                variants.append(replaced)
                break

        # "active users" → include is_active=1
        if any(s in q for s in _ACTIVE_SYNONYMS):
            variants.append(f"{query} is_active 1")

        # "inactive users" → include is_active=0
        if any(s in q for s in _INACTIVE_SYNONYMS):
            variants.append(f"{query} is_active 0")

        # "recent" → add created_at context
        if any(s in q for s in _RECENT_SYNONYMS):
            variants.append(f"{query} created_at recent registration")

        # email domain extraction: "gmail users" → "@gmail.com"
        gmail_match = re.search(r"\b(\w+)\s+(?:email|account|mail)\b", q)
        if gmail_match:
            domain = gmail_match.group(1)
            variants.append(f"@{domain}.com email")

        # phone prefix: "+91" → indian phone
        phone_match = re.search(r"\+(\d{1,3})", query)
        if phone_match:
            variants.append(f"phone_number {phone_match.group(0)}")

        # Strip question words for embedding
        stripped = re.sub(
            r"^(who is|what is|find|show me|get|list|fetch|give me)\s+",
            "", q, flags=re.IGNORECASE,
        ).strip()
        if stripped and stripped != q:
            variants.append(stripped)

        return variants
    """================= End method _heuristic_expand ================="""

    """================= Startup method _llm_expand ================="""
    def _llm_expand(self, query: str) -> list[str]:
        """
        Ask the LLM to produce rephrasings optimised for keyword/semantic
        similarity to structured database rows.
        Returns an empty list on any failure.
        """
        if self._llm is None:
            return []

        prompt = (
            "You are a query expansion engine for a structured database search system.\n"
            "Given a user query, produce exactly 3 alternative phrasings that are more\n"
            "likely to match embedded database rows.  Rows look like:\n"
            "  name: John | email: john@example.com | country_code: US | is_active: 1\n\n"
            f"User query: {query}\n\n"
            "Return ONLY the 3 alternatives, one per line, no numbering, no explanation."
        )

        raw   = self._llm.complete(prompt, max_tokens=120, temperature=0.3)
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        return lines[:3]
    """================= End method _llm_expand ================="""

"""================= End class QueryRewriter ================="""