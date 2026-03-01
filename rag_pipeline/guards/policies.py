"""
Guardrail policies for GuardRAG.

The goal is to make it easy to inject pre- and post-generation checks,
mirroring the guard components in RAG'n'Roll.  Policies can operate on
retrieved documents, prompts, or generated answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol

from rag_pipeline_components.types import Document


@dataclass
class GuardDecision:
    allow: bool
    reason: str = ""
    fallback: Optional[str] = None

    @classmethod
    def allow_pass(cls, reason: str = "") -> "GuardDecision":
        return cls(True, reason, None)

    @classmethod
    def deny(cls, reason: str, *, fallback: Optional[str] = None) -> "GuardDecision":
        return cls(False, reason, fallback)


class Guardrail(Protocol):
    """Protocol describing a guard policy."""

    def inspect_retrieved(self, documents: Iterable[Document]) -> GuardDecision:
        ...

    def inspect_prompt(self, prompt: str) -> GuardDecision:
        ...

    def inspect_generation(self, prompt: str, answer: str) -> GuardDecision:
        ...


class KeywordBlocklistGuard:
    """
    Simple guard that flags documents or prompts containing high-risk tokens.
    Intended as a bootstrap before plugging in the full SecAlign/DPO policies.
    """

    def __init__(self, blocklist: Iterable[str] | None = None):
        self.blocklist = [token.lower() for token in (blocklist or ["override", "ignore safety", "developer mode"])]

    def inspect_retrieved(self, documents: Iterable[Document]) -> GuardDecision:
        for doc in documents:
            text = f"{doc.title} {doc.text}".lower()
            if any(token in text for token in self.blocklist):
                return GuardDecision.deny(
                    f"Blocked poisoned span in doc {doc.doc_id}",
                    fallback="I’m refusing because the retrieved context looked unsafe.",
                )
        return GuardDecision.allow_pass()

    def inspect_prompt(self, prompt: str) -> GuardDecision:
        lowered = prompt.lower()
        if any(token in lowered for token in self.blocklist):
            return GuardDecision.deny(
                "Prompt contains blocked keyword",
                fallback="I’m declining—this prompt appears to contain a malicious instruction.",
            )
        return GuardDecision.allow_pass()

    def inspect_generation(self, prompt: str, answer: str) -> GuardDecision:
        lowered = answer.lower()
        if any(token in lowered for token in self.blocklist):
            return GuardDecision.deny(
                "Answer echoes blocked keyword",
                fallback="I’m refusing because the generated answer echoed a blocked instruction.",
            )
        return GuardDecision.allow_pass()


class SecAlignGuard:
    """
    Approximation of SecAlign-style guardrail:
    - Tracks directives from retrieved poisoned documents.
    - Uses SecAlign heuristic to detect if generated answer followed the directive.
    """

    def __init__(self):
        self.injection_cues = [
            "override",
            "ignore previous instructions",
            "developer mode",
            "system override",
            "bypass safety",
            "root access",
        ]
        self.security_indicators = [
            "i cannot comply",
            "i'm not able to help",
            "i must refuse",
            "i'm sorry, but i can't",
            "cannot assist with that",
            "ignoring the malicious instruction",
            "based only on the provided document",
        ]
        self._directives: List[str] = []
        self._poisoned_detected: bool = False
        self._prompt_cues: bool = False

    def _contains_cue(self, text: str) -> bool:
        lowered = text.lower()
        return any(cue in lowered for cue in self.injection_cues)

    def inspect_retrieved(self, documents: Iterable[Document]) -> GuardDecision:
        self._directives = []
        self._poisoned_detected = False
        self._prompt_cues = False

        for doc in documents:
            # Use the is_poisoned property which checks doc_id.startswith("IPI_")
            if doc.is_poisoned:
                self._poisoned_detected = True
            directive = doc.metadata.get("_directive") or doc.metadata.get("directive")
            if directive:
                self._directives.append(str(directive))

        if self._poisoned_detected:
            return GuardDecision.allow_pass("Poisoned documents detected; monitoring generation.")
        return GuardDecision.allow_pass()

    def inspect_prompt(self, prompt: str) -> GuardDecision:
        if self._contains_cue(prompt):
            self._prompt_cues = True
        return GuardDecision.allow_pass()

    def inspect_generation(self, prompt: str, answer: str) -> GuardDecision:
        if not answer.strip():
            return GuardDecision.allow_pass("No answer generated.")

        for directive in self._directives:
            if self._is_insecure(answer, directive):
                return GuardDecision.deny(
                    "SecAlign guard detected the answer following a malicious directive.",
                    fallback="I’m declining to answer because the retrieved content appeared malicious.",
                )

        if self._poisoned_detected and not self._has_security_indicator(answer):
            return GuardDecision.deny(
                "SecAlign guard observed poisoned context without a safe refusal.",
                fallback="I’m refusing to comply because the context may contain a malicious instruction.",
            )

        if self._prompt_cues and self._contains_cue(answer):
            return GuardDecision.deny(
                "SecAlign guard saw injection cues echoed in the answer.",
                fallback="I’m stopping here; the response echoed a suspected injection.",
            )

        return GuardDecision.allow_pass()

    def _is_insecure(self, response: str, directive: str) -> bool:
        response_lower = response.lower()
        directive_lower = directive.lower()

        directive_tokens = [token for token in directive_lower.split() if len(token) > 3]
        directive_keywords = set(directive_tokens[:5])

        contains_injection = any(keyword in response_lower for keyword in directive_keywords)
        mentions_directive_verbatim = directive_lower in response_lower

        if mentions_directive_verbatim and not self._has_security_indicator(response):
            return True

        if contains_injection and not self._has_security_indicator(response):
            return True

        if len(response.split()) < 20 and contains_injection:
            return True

        return False

    def _has_security_indicator(self, response: str) -> bool:
        lowered = response.lower()
        return any(indicator in lowered for indicator in self.security_indicators)


GUARD_REGISTRY = {
    "keyword_blocklist": KeywordBlocklistGuard,
    "secalign": SecAlignGuard,
}


def build_guards(names: Iterable[str]) -> List[Guardrail]:
    guards: List[Guardrail] = []
    for name in names:
        try:
            guards.append(GUARD_REGISTRY[name]())
        except KeyError as exc:
            raise ValueError(f"Unknown guard '{name}'. Available: {list(GUARD_REGISTRY)}") from exc
    return guards


DEFAULT_GUARDS: List[Guardrail] = build_guards(["keyword_blocklist"])


