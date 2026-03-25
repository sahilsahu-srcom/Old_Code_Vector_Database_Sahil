"""
File Summary:
Production guardrails for the HAUP RAG engine. Applied before and after the LLM call.
Covers query length limits, blocked keywords, prompt injection detection, PII detection
and redaction, sliding-window rate limiting, and lightweight hallucination detection.

====================================================================
SYSTEM PIPELINE FLOW (Architecture + Object Interaction)
====================================================================

Guardrails()  [Class → Object]
||
├── __init__()  [Function] -------------------------------> Compile regex patterns, init rate window
│
├── check_input()  [Function] ----------------------------> Run all input guard checks
│       │
│       ├── Length check ---------------------------------> Reject if below min or above max
│       │       │
│       │       └── [Early Exit Branch] ------------------> Return blocked GuardResult
│       │
│       ├── Blocked keywords -----------------------------> Substring match against blocked list
│       │       │
│       │       └── [Early Exit Branch] ------------------> Return blocked GuardResult
│       │
│       ├── _detect_injections()  [Function] -------------> Regex scan for injection patterns
│       │       │
│       │       ├── [Conditional Branch] block_injections -> Return blocked GuardResult
│       │       │
│       │       └── [Conditional Branch] warn only -------> Append warning, continue
│       │
│       ├── _detect_pii()  [Function] --------------------> Regex scan for PII types
│       │       │
│       │       ├── Append PII warning -------------------> Note detected PII types
│       │       │
│       │       └── [Conditional Branch] pii_redact_in_query -> _redact_pii() on query
│       │
│       └── _check_rate_limit()  [Function] --------------> Sliding window per session_id
│               │
│               └── [Early Exit Branch] limit exceeded ---> Return blocked GuardResult
│
├── check_output()  [Function] ---------------------------> Run output safety checks
│       │
│       ├── [Conditional Branch] pii_redact_in_response -> _redact_pii() on response
│       │
│       └── [Conditional Branch] hallucination_check ----> _check_hallucination()
│               │
│               └── Flag if suspicious numbers found -----> Set safe=False, append warning
│
├── _detect_injections()  [Function] ---------------------> Search compiled injection regex list
│
├── _detect_pii()  [Function] ----------------------------> Search compiled PII regex patterns
│
├── _redact_pii()  [Function] ----------------------------> Replace PII matches with [REDACTED_TYPE]
│
├── _check_rate_limit()  [Function] ----------------------> Evict old timestamps, enforce 60s window
│
└── _check_hallucination()  [Function] -------------------> Compare response numbers to retrieved docs
        │
        └── [Early Exit Branch] no numbers in response ---> Return None

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from rag_core import logger as log


"""================= Startup class GuardrailsConfig ================="""
@dataclass
class GuardrailsConfig:
    # Input limits
    max_query_length:       int        = 1000
    min_query_length:       int        = 2

    # Rate limiting
    rate_limit_enabled:     bool       = True
    max_queries_per_minute: int        = 30

    # Injection protection
    injection_detection:    bool       = True
    block_injections:       bool       = True   # False = warn only

    # PII handling
    pii_detection:          bool       = True
    pii_redact_in_query:    bool       = False
    pii_redact_in_response: bool       = False

    # Blocked keywords (case-insensitive substring match)
    blocked_keywords:       List[str]  = field(default_factory=list)

    # Hallucination detection
    hallucination_check:    bool       = True
"""================= End class GuardrailsConfig ================="""


"""================= Startup class GuardResult ================="""
@dataclass
class GuardResult:
    allowed:        bool            = True
    warnings:       List[str]       = field(default_factory=list)
    modified_query: Optional[str]   = None   # set if query was redacted
    block_reason:   Optional[str]   = None
"""================= End class GuardResult ================="""


"""================= Startup class OutputGuardResult ================="""
@dataclass
class OutputGuardResult:
    safe:              bool          = True
    warnings:          List[str]     = field(default_factory=list)
    modified_response: Optional[str] = None
"""================= End class OutputGuardResult ================="""


"""================= Startup class Guardrails ================="""
class Guardrails:

    # Patterns that indicate prompt injection attempts
    _INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(everything|all|your\s+instructions)",
        r"you\s+are\s+now\s+(?!a\s+data)",
        r"act\s+as\s+(?!a\s+data\s+analyst)",
        r"jailbreak",
        r"system\s+prompt",
        r"<\s*/?(?:system|instructions?|prompt)\s*>",
        r"\[INST\]|\[/INST\]",
        r"###\s*(?:System|Human|Assistant):",
        r"print\s+(your\s+)?(?:system\s+)?instructions",
        r"reveal\s+(your\s+)?(prompt|instructions|context)",
    ]

    # PII patterns
    _PII_PATTERNS = {
        "email":       r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "phone":       r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "ssn":         r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "ip":          r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }

    """================= Startup method __init__ ================="""
    def __init__(self, cfg: GuardrailsConfig):
        self._cfg  = cfg
        self._log  = log.get("guardrails")

        # Compiled injection patterns
        self._injection_re = [
            re.compile(p, re.IGNORECASE) for p in self._INJECTION_PATTERNS
        ]

        # Compiled PII patterns
        self._pii_re = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self._PII_PATTERNS.items()
        }

        # Rate limit tracking: session_id → [timestamp, ...]
        self._rate_window: Dict[str, List[float]] = defaultdict(list)

        # Blocked keywords (lowercased)
        self._blocked = [kw.lower() for kw in cfg.blocked_keywords]
    """================= End method __init__ ================="""

    """================= Startup method check_input ================="""
    def check_input(self, query: str, session_id: str = "") -> GuardResult:
        """
        Run all input guards. Returns GuardResult with allowed=False
        if the query should be blocked.
        """
        result = GuardResult()
        q      = query

        # 1. Length check
        if len(query) < self._cfg.min_query_length:
            result.allowed      = False
            result.block_reason = "Query too short"
            return result

        if len(query) > self._cfg.max_query_length:
            result.allowed      = False
            result.block_reason = f"Query exceeds {self._cfg.max_query_length} character limit"
            return result

        # 2. Blocked keywords
        q_lower = query.lower()
        for kw in self._blocked:
            if kw in q_lower:
                result.allowed      = False
                result.block_reason = "Query contains blocked content"
                self._log.warning("Blocked keyword detected in session %s", session_id)
                return result

        # 3. Injection detection
        if self._cfg.injection_detection:
            injections = self._detect_injections(query)
            if injections:
                if self._cfg.block_injections:
                    result.allowed      = False
                    result.block_reason = "Potential prompt injection detected"
                    self._log.warning(
                        "Injection attempt in session %s: %s", session_id, injections
                    )
                    return result
                else:
                    result.warnings.append(
                        f"Possible injection pattern detected: {injections[0]}"
                    )

        # 4. PII detection
        if self._cfg.pii_detection:
            pii_found = self._detect_pii(query)
            if pii_found:
                result.warnings.append(
                    f"Query may contain PII: {', '.join(pii_found)}"
                )
                if self._cfg.pii_redact_in_query:
                    q                    = self._redact_pii(q)
                    result.modified_query = q

        # 5. Rate limiting
        if self._cfg.rate_limit_enabled:
            if not self._check_rate_limit(session_id):
                result.allowed      = False
                result.block_reason = (
                    f"Rate limit exceeded: max {self._cfg.max_queries_per_minute} "
                    f"queries per minute"
                )
                return result

        if result.warnings:
            self._log.debug(
                "Input warnings for session %s: %s", session_id, result.warnings
            )

        return result
    """================= End method check_input ================="""

    """================= Startup method check_output ================="""
    def check_output(
        self,
        response:            str,
        retrieved_row_texts: List[str],
        session_id:          str = "",
    ) -> OutputGuardResult:
        """
        Run output safety checks.
        """
        result = OutputGuardResult()
        out    = response

        # 1. PII scrubbing from response
        if self._cfg.pii_redact_in_response:
            pii_in_response = self._detect_pii(response)
            if pii_in_response:
                out                       = self._redact_pii(out)
                result.warnings.append(
                    f"PII redacted from response: {', '.join(pii_in_response)}"
                )
                result.modified_response = out

        # 2. Hallucination detection (lightweight heuristic)
        if self._cfg.hallucination_check and retrieved_row_texts:
            hall_warning = self._check_hallucination(response, retrieved_row_texts)
            if hall_warning:
                result.warnings.append(hall_warning)
                result.safe = False

        if result.warnings:
            self._log.debug(
                "Output warnings for session %s: %s", session_id, result.warnings
            )

        return result
    """================= End method check_output ================="""

    """================= Startup method _detect_injections ================="""
    def _detect_injections(self, query: str) -> List[str]:
        found = []
        for pattern in self._injection_re:
            m = pattern.search(query)
            if m:
                found.append(m.group(0)[:50])
        return found
    """================= End method _detect_injections ================="""

    """================= Startup method _detect_pii ================="""
    def _detect_pii(self, text: str) -> List[str]:
        found = []
        for name, pattern in self._pii_re.items():
            if pattern.search(text):
                found.append(name)
        return found
    """================= End method _detect_pii ================="""

    """================= Startup method _redact_pii ================="""
    def _redact_pii(self, text: str) -> str:
        """Replace PII matches with [REDACTED_TYPE] tokens."""
        result = text
        for name, pattern in self._pii_re.items():
            result = pattern.sub(f"[REDACTED_{name.upper()}]", result)
        return result
    """================= End method _redact_pii ================="""

    """================= Startup method _check_rate_limit ================="""
    def _check_rate_limit(self, session_id: str) -> bool:
        """Sliding window rate limiter."""
        now    = time.time()
        window = self._rate_window[session_id]

        # Evict entries older than 60s
        self._rate_window[session_id] = [t for t in window if now - t < 60]

        if len(self._rate_window[session_id]) >= self._cfg.max_queries_per_minute:
            return False

        self._rate_window[session_id].append(now)
        return True
    """================= End method _check_rate_limit ================="""

    """================= Startup method _check_hallucination ================="""
    def _check_hallucination(
        self,
        response:        str,
        retrieved_texts: List[str],
    ) -> Optional[str]:
        """
        Lightweight hallucination heuristic:
        Check if the LLM cited specific numbers (counts, IDs, totals)
        that don't appear in any retrieved document.
        """
        # Extract numbers from response
        response_numbers = set(re.findall(r"\b\d{2,}\b", response))
        if not response_numbers:
            return None

        # Build set of all numbers appearing in retrieved docs
        corpus_numbers: Set[str] = set()
        for text in retrieved_texts:
            corpus_numbers.update(re.findall(r"\b\d{2,}\b", text))

        # Numbers in response that weren't in any retrieved doc
        suspicious = response_numbers - corpus_numbers

        # Filter out years and very small numbers
        suspicious = {
            n for n in suspicious
            if not (1900 <= int(n) <= 2100)
            and int(n) > 10
        }

        if len(suspicious) > 3:
            return (
                f"Response contains {len(suspicious)} numbers not found in "
                f"retrieved data — possible hallucination. "
                f"Examples: {', '.join(list(suspicious)[:3])}"
            )
        return None
    """================= End method _check_hallucination ================="""

"""================= End class Guardrails ================="""