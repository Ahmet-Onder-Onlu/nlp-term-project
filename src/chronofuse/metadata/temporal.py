"""Rule-based temporal expression extraction and normalisation.

Produces a :class:`TemporalProfile` for any text: explicit years, year ranges,
decades/centuries, month-year mentions, ISO dates, version-like release markers
and relative expressions ("two weeks ago", "currently"). Everything is mapped to
a set of *years* so queries and documents can be compared on one timeline.

This is deliberately dependency-free and fast enough to run over ~1.2M
documents; the LLM-based analyzer (``chronofuse.query``) refines query profiles.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

MIN_YEAR, MAX_YEAR = 1000, 2035

_MONTHS = ("january|february|march|april|may|june|july|august|september|october|"
           "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")

# NB: order matters; ranges are matched before single years.
_RE_RANGE = re.compile(r"\b(1[0-9]{3}|20[0-3][0-9])\s*(?:-|–|—|to|until|through)\s*(1[0-9]{3}|20[0-3][0-9])\b", re.I)
_RE_ISO = re.compile(r"\b(1[0-9]{3}|20[0-3][0-9])-(0[1-9]|1[0-2])(?:-(0[1-9]|[12][0-9]|3[01]))?\b")
_RE_MONTH_YEAR = re.compile(rf"\b(?:{_MONTHS})\.?\s+(?:\d{{1,2}}(?:st|nd|rd|th)?,?\s+)?(1[0-9]{{3}}|20[0-3][0-9])\b", re.I)
_RE_DECADE = re.compile(r"\b(1[0-9]{2}|20[0-2])0'?s\b")
_RE_CENTURY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)[\s-]+century\b", re.I)
_RE_YEAR = re.compile(r"(?<![\d.,$#/-])(1[0-9]{3}|20[0-3][0-9])(?![\d.,%/-]|\s*(?:bytes|btc|usd|kb|mb|gb|px|ms))", re.I)
_RE_RELATIVE = re.compile(
    r"\b(today|yesterday|tomorrow|now|currently|recently|nowadays|at present|so far|"
    r"(?:last|next|this|past|coming|previous)\s+(?:year|month|week|decade|century|quarter)|"
    r"(?:a|one|two|three|four|five|six|seven|eight|nine|ten|few|several|\d+)\s+"
    r"(?:years?|months?|weeks?|days?|decades?)\s+(?:ago|later|before|after|earlier)|"
    r"since|until|before|after|during|historically|originally|initially|formerly|anymore|"
    r"still|no longer|used to|any more)\b", re.I)


@dataclass
class TemporalProfile:
    years: list[int] = field(default_factory=list)           # all normalised years (sorted, unique)
    explicit: list[str] = field(default_factory=list)        # surface forms of absolute expressions
    relative: list[str] = field(default_factory=list)        # surface forms of relative expressions
    granularity: str = "none"                                # day | month | year | decade | century | none

    @property
    def has_time(self) -> bool:
        return bool(self.years)

    @property
    def span(self) -> tuple[int, int] | None:
        return (self.years[0], self.years[-1]) if self.years else None

    def to_dict(self) -> dict:
        return asdict(self)


def _valid(y: int) -> bool:
    return MIN_YEAR <= y <= MAX_YEAR


def extract_temporal(text: str, max_chars: int = 20000) -> TemporalProfile:
    """Extract a temporal profile from ``text`` (truncated to ``max_chars``)."""
    text = text[:max_chars]
    years: set[int] = set()
    explicit: list[str] = []
    gran_rank = {"none": 0, "century": 1, "decade": 2, "year": 3, "month": 4, "day": 5}
    gran = "none"

    def bump(g: str) -> None:
        nonlocal gran
        if gran_rank[g] > gran_rank[gran]:
            gran = g

    consumed: list[tuple[int, int]] = []

    def free(m: re.Match) -> bool:
        return not any(a <= m.start() < b for a, b in consumed)

    for m in _RE_RANGE.finditer(text):
        a, b = int(m.group(1)), int(m.group(2))
        if _valid(a) and _valid(b) and a <= b and b - a <= 200:
            years.update(range(a, b + 1)) if b - a <= 50 else years.update({a, b})
            explicit.append(m.group(0)); consumed.append(m.span()); bump("year")
    for m in _RE_ISO.finditer(text):
        if free(m) and _valid(int(m.group(1))):
            years.add(int(m.group(1))); explicit.append(m.group(0)); consumed.append(m.span())
            bump("day" if m.group(3) else "month")
    for m in _RE_MONTH_YEAR.finditer(text):
        if free(m) and _valid(int(m.group(1))):
            years.add(int(m.group(1))); explicit.append(m.group(0)); consumed.append(m.span()); bump("month")
    for m in _RE_DECADE.finditer(text):
        start = int(m.group(1)) * 10
        if _valid(start):
            years.update(range(start, start + 10)); explicit.append(m.group(0)); consumed.append(m.span())
            bump("decade")
    for m in _RE_CENTURY.finditer(text):
        c = int(m.group(1))
        if 10 <= c <= 21:
            start = (c - 1) * 100
            years.update({start, start + 50, start + 99}); explicit.append(m.group(0)); bump("century")
    for m in _RE_YEAR.finditer(text):
        if free(m):
            y = int(m.group(1))
            if _valid(y):
                years.add(y); explicit.append(m.group(0)); bump("year")

    relative = [m.group(0).lower() for m in _RE_RELATIVE.finditer(text)]
    return TemporalProfile(years=sorted(years), explicit=explicit[:50],
                           relative=sorted(set(relative))[:50], granularity=gran)


def year_overlap(q_years: list[int], d_years: list[int], tolerance: int = 1) -> float:
    """Soft overlap in [0, 1]: fraction of query years matched by a document year within ``tolerance``."""
    if not q_years or not d_years:
        return 0.0
    dset = set(d_years)
    hit = sum(1 for y in q_years if any((y + k) in dset for k in range(-tolerance, tolerance + 1)))
    return hit / len(q_years)
