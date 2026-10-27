"""Mechanical grounding gate for simulation reports.

Why this exists: the Telegram E2E report invented quotes and a
``$10-20/mo`` price that appear in ZERO simulation database rows. The
report agent's only behavioral evidence channel (interviews) requires a
live simulation environment, and reports are generated after the run
stops -- so the LLM filled the gap from the brief's own questions. A
prompt rule ("don't fabricate") did not stop it; only a mechanical
check can.

What it does, on the finished markdown BEFORE it is saved as the
report:

* Every quoted span must match observed evidence. A quote attributed
  to users/agents must match simulation behavior (posts, comments,
  interview transcripts). A non-attributed quote (e.g. the launch
  announcement) may match the brief.
* Every money/percent figure must match observed evidence. A figure in
  a sentence attributing it to users ("users would pay $X") must match
  simulation behavior; a purely descriptive figure ("free tier covers
  5 tickers") may match the brief.
* Unverified quotes are removed; sentences carrying unverified figures
  are dropped. The gate records everything it removed, and the report
  carries a grounding verdict: ``grounded`` (nothing removed),
  ``grounded_after_stripping`` (something removed, but at least one
  behavior-verified claim remains), or ``ungrounded`` (no
  behavior-verified claim remains -- do not use for decisions).

Matching is normalized-substring: quotes may be excerpts of longer
posts, but a report quote that EXTENDS a post (invented continuation)
does not match and is stripped. This is v1 precision: it cannot catch
a paraphrase presented as fact -- that is what the sim_posts tool and
the verbatim-quote prompt rule are for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Double-style quotes: straight, curly, and Chinese corner brackets.
_QUOTE_RE = re.compile(r'[\"“「]([^\"”」]{1,2000})[\"”」]')

# Money: $10, $10-20, $10/mo, 10 dollars. Percent: 10%, 10 percent.
_MONEY_RE = re.compile(
    r"\$\s*\d[\d,]*(?:\.\d+)?(?:\s*[-–—]\s*\$?\s*\d[\d,]*(?:\.\d+)?)?"
    r"(?:\s*/\s*(?:mo|month))?"
    r"|\b\d[\d,]*(?:\.\d+)?\s+dollars?\b"
)
_PERCENT_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|percent)\b")
# Chinese money forms, so a Chinese report (or a locale regression)
# cannot smuggle a price past the gate: 10美元, 每月10至20美元.
_CHINESE_MONEY_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:至|到|[-–—])\s*\d[\d,]*(?:\.\d+)?\s*美元"
    r"|\d[\d,]*(?:\.\d+)?\s*美元"
)

# Words that make a sentence a claim ABOUT users/agents. Such sentences
# must ground in behavior, never in the brief alone.
_ATTRIBUTION_RE = re.compile(
    r"\b(user|users|agent|agents|persona|personas|participant|participants|"
    r"respondent|respondents|said|says|say|consider|considers|willing|"
    r"would pay|expect|expects|feel|feels|think|thinks|believe|believes|"
    r"prefer|prefers|told|mentioned|noted|reported|interviewed)\b"
    r"|用户|认为|表示|愿意|觉得|说",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def normalize_text(text: str) -> str:
    """Lowercase, unify quote marks, drop most punctuation, collapse space.

    Sentence-final periods are dropped (only decimal points survive), so
    a report quote ending in "." still matches the post it came from.
    """
    s = text.lower()
    for a, b in (("“", '"'), ("”", '"'), ("「", '"'), ("」", '"'),
                 ("‘", "'"), ("’", "'")):
        s = s.replace(a, b)
    s = re.sub(r"(?<!\d)\.(?!\d)", " ", s)
    s = re.sub(r"[^\w\s$%.\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def canonical_number(token: str) -> str:
    """Canonical form of a money/percent token for set membership."""
    s = token.lower().strip()
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    s = s.replace("/mo", "").replace("/month", "")
    s = s.replace("dollars", "").replace("dollar", "")
    s = s.replace("percent", "").replace("%", "")
    s = re.sub(r"[-–—]", "-", s)
    return s


def _canonical_chinese_money(token: str) -> str:
    s = token.replace("美元", "").replace(" ", "").replace(",", "")
    s = re.sub(r"[至到–—]", "-", s)
    return s


def _number_tokens(text: str) -> set[str]:
    out = set()
    for m in _MONEY_RE.finditer(text):
        out.add(canonical_number(m.group(0)))
    for m in _PERCENT_RE.finditer(text):
        out.add(canonical_number(m.group(0)))
    for m in _CHINESE_MONEY_RE.finditer(text):
        out.add(_canonical_chinese_money(m.group(0)))
    return out


def _split_sentences(line: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(line)
    return [p for p in parts if p.strip()]


@dataclass
class Claim:
    kind: str  # "quote" or "number"
    text: str
    line_idx: int
    span: tuple[int, int] | None = None  # char span within the line
    sentence: str = ""
    status: str = "unverified"  # "verified" or "unverified"
    source: str = ""  # "behavior" or "brief" when verified


@dataclass
class GroundingResult:
    claims: list[Claim] = field(default_factory=list)
    removed_quotes: list[str] = field(default_factory=list)
    removed_sentences: list[str] = field(default_factory=list)
    verdict: str = "ungrounded"

    @property
    def quotes_checked(self) -> int:
        return sum(1 for c in self.claims if c.kind == "quote")

    @property
    def quotes_verified(self) -> int:
        return sum(1 for c in self.claims
                   if c.kind == "quote" and c.status == "verified")

    @property
    def numbers_checked(self) -> int:
        return sum(1 for c in self.claims if c.kind == "number")

    @property
    def numbers_verified(self) -> int:
        return sum(1 for c in self.claims
                   if c.kind == "number" and c.status == "verified")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "quotes_checked": self.quotes_checked,
            "quotes_verified": self.quotes_verified,
            "numbers_checked": self.numbers_checked,
            "numbers_verified": self.numbers_verified,
            "removed_quotes": self.removed_quotes,
            "removed_sentences": self.removed_sentences,
            "claims": [
                {"kind": c.kind, "text": c.text, "status": c.status,
                 "source": c.source, "sentence": c.sentence}
                for c in self.claims
            ],
        }


def verify_report(
    markdown: str,
    behavior_texts: list[str],
    brief_texts: list[str],
) -> GroundingResult:
    """Check every quote/figure in the markdown against the evidence."""
    behavior_norm = [normalize_text(t) for t in behavior_texts]
    brief_norm = [normalize_text(t) for t in brief_texts]
    behavior_numbers: set[str] = set()
    brief_numbers: set[str] = set()
    for t in behavior_texts:
        behavior_numbers |= _number_tokens(t)
    for t in brief_texts:
        brief_numbers |= _number_tokens(t)

    result = GroundingResult()
    lines = markdown.split("\n")
    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped in ("---", "***"):
            continue
        # Quotes: check the sentence holding each quote for attribution.
        for m in _QUOTE_RE.finditer(line):
            quote = m.group(1).strip()
            if not quote:
                continue
            sentence = _sentence_holding(line, m.start())
            attributed = bool(_ATTRIBUTION_RE.search(sentence))
            nq = normalize_text(quote)
            in_behavior = any(nq and nq in b for b in behavior_norm)
            in_brief = any(nq and nq in b for b in brief_norm)
            if in_behavior:
                status, source = "verified", "behavior"
            elif in_brief and not attributed:
                status, source = "verified", "brief"
            else:
                status, source = "unverified", ""
            result.claims.append(Claim(
                kind="quote", text=quote, line_idx=line_idx,
                span=(m.start(), m.end()), sentence=sentence,
                status=status, source=source,
            ))
        # Numbers: one claim per distinct figure per line (deduped by
        # canonical form) to avoid counting the same figure repeatedly.
        seen: set[str] = set()
        number_matches = (
            [(m, "std") for m in _MONEY_RE.finditer(line)]
            + [(m, "std") for m in _PERCENT_RE.finditer(line)]
            + [(m, "cny") for m in _CHINESE_MONEY_RE.finditer(line)]
        )
        for m, kind in number_matches:
            token = m.group(0)
            canon = (
                _canonical_chinese_money(token)
                if kind == "cny"
                else canonical_number(token)
            )
            if canon in seen or not canon:
                continue
            seen.add(canon)
            sentence = _sentence_holding(line, m.start())
            attributed = bool(_ATTRIBUTION_RE.search(sentence))
            if canon in behavior_numbers:
                status, source = "verified", "behavior"
            elif canon in brief_numbers and not attributed:
                status, source = "verified", "brief"
            else:
                status, source = "unverified", ""
            result.claims.append(Claim(
                kind="number", text=token, line_idx=line_idx,
                span=(m.start(), m.end()), sentence=sentence,
                status=status, source=source,
            ))
    return result


def _sentence_holding(line: str, pos: int) -> str:
    for sent in _split_sentences(line):
        start = line.find(sent)
        if start <= pos < start + len(sent):
            return sent
    return line


def strip_unverified(markdown: str, verification: GroundingResult) -> str:
    """Remove unverified quotes; drop sentences with unverified figures.

    Order matters: sentences carrying unverified figures are dropped
    FIRST on the original text (a figure-bearing quote line goes away
    whole, quote included); unverified quote spans are then removed
    from surviving lines. Returns cleaned markdown and fills
    verification.removed_* + verdict.
    """
    out_lines: list[str] = []
    lines = markdown.split("\n")
    for line_idx, line in enumerate(lines):
        # Step 1: drop sentences carrying unverified figures.
        bad_sentences = {
            c.sentence for c in verification.claims
            if c.line_idx == line_idx and c.kind == "number"
            and c.status == "unverified"
        }
        if bad_sentences:
            kept = [s for s in _split_sentences(line) if s not in bad_sentences]
            for s in _split_sentences(line):
                if s in bad_sentences:
                    verification.removed_sentences.append(s)
            if not kept:
                continue
            # Rebuild the line from kept sentences, preserving a leading
            # blockquote/bullet marker.
            stripped = line.lstrip()
            marker = ""
            for m in ("> ", "- ", "* ", "- ", "1. "):
                if stripped.startswith(m):
                    marker = m
                    break
            line = marker + " ".join(s.strip() for s in kept)
        # Step 2: remove unverified quote spans (end-first: spans stay valid).
        spans = sorted(
            (c.span for c in verification.claims
             if c.line_idx == line_idx and c.kind == "quote"
             and c.status == "unverified" and c.span),
            reverse=True,
        )
        new_line = line
        for start, end in spans:
            verification.removed_quotes.append(new_line[start:end])
            new_line = new_line[:start] + new_line[end:]
        # A line left as a bare marker/whitespace is dropped, not kept.
        if new_line.strip().strip(">* -") == "":
            continue
        if new_line.strip():
            out_lines.append(new_line.rstrip())
    cleaned = "\n".join(out_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"

    behavior_verified = any(
        c.status == "verified" and c.source == "behavior"
        for c in verification.claims
    )
    removed_any = bool(verification.removed_quotes
                       or verification.removed_sentences)
    if not removed_any:
        verification.verdict = "grounded"
    elif behavior_verified:
        verification.verdict = "grounded_after_stripping"
    else:
        verification.verdict = "ungrounded"
    return cleaned
