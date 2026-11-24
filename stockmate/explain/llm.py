"""The language layer, and the boundary it is not allowed to cross.

StockMate's rule is that code decides and the language layer only explains.
That is not a stylistic preference.  A model that is allowed to compute the
refund a shopkeeper will demand from their supplier will eventually compute it
wrong, confidently, in fluent Tamil, and the shopkeeper will be the one standing
at the counter being told they are mistaken.

So the boundary is enforced in the type system rather than in a prompt.  An
explainer receives findings whose amounts have already been computed and
recorded in the ledger, and it returns prose.  It never returns a number that
the caller then trusts: whatever it writes, the rupee figures shown to the owner
come from :class:`Finding.amount`, which the rupee guardrail has already checked
against a stored computation.

The default explainer uses templates and needs no model, no key and no network.
An adapter for a hosted model is defined here so the architecture is real, but
it is off unless someone deliberately turns it on, and turning it on does not
widen what the layer is allowed to decide.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import Finding

__all__ = ["Explainer", "NullExplainer", "LLMExplainer", "describe"]


@runtime_checkable
class Explainer(Protocol):
    """Turns already-computed findings into something a person wants to read."""

    name: str

    def explain(self, finding: Finding) -> str: ...


class NullExplainer:
    """The default: the finding's own wording, written when it was created.

    Calling this null is slightly unfair to it.  The detail text on a finding is
    composed by the engine that did the arithmetic, from the same inputs it
    recorded in the ledger, so it is the one explanation that cannot drift from
    the number it describes.
    """

    name = "template"

    def explain(self, finding: Finding) -> str:
        return finding.detail


class LLMExplainer:
    """Optional rewriting of a finding into plainer or translated language.

    Three safeguards, because this is the only component that could put words in
    the system's mouth:

      * the model is handed the finding's text and is asked to rephrase it, not
        given the raw bill to reason about;
      * the rupee figure is appended by this code after the model has spoken, so
        a hallucinated amount cannot replace the computed one;
      * any failure falls back to the template, so an outage degrades the
        wording and never the correctness.
    """

    name = "llm"

    def __init__(self, client=None, language: str = "English", fallback: Explainer | None = None):
        self.client = client
        self.language = language
        self.fallback = fallback or NullExplainer()

    def explain(self, finding: Finding) -> str:
        if self.client is None:
            return self.fallback.explain(finding)
        try:
            text = self.client.rephrase(
                text=finding.detail,
                language=self.language,
                instruction=(
                    "Rewrite this for a shopkeeper in simple sentences. "
                    "Do not add, remove or change any number."
                ),
            )
        except Exception:
            return self.fallback.explain(finding)

        if not isinstance(text, str) or not text.strip():
            return self.fallback.explain(finding)
        if finding.amount is not None:
            # The computed figure is restated by us, whatever the model wrote.
            return f"{text.strip()} ({finding.amount.format()})"
        return text.strip()


def describe(findings: list[Finding], explainer: Explainer | None = None) -> list[dict]:
    """Render findings for display, keeping the computed amount authoritative."""
    explainer = explainer or NullExplainer()
    out = []
    for f in findings:
        out.append({
            "finding_id": f.finding_id,
            "kind": f.kind,
            "title": f.title,
            "text": explainer.explain(f),
            "amount": f.amount.format() if f.amount else None,
            "amount_paise": f.amount.paise if f.amount else None,
            "bucket": f.bucket,
            "severity": f.severity.value,
            "explained_by": explainer.name,
            "computation_id": f.computation_id,
        })
    return out
