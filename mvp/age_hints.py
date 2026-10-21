import re


# --------------------------------------------------
# Rough age hint for a ChildPlay source video, from
# its YouTube title / description / keywords.
#
# ChildPlay does not annotate age, so this is only a
# starting suggestion for the labeller to confirm by
# looking at the child.
# --------------------------------------------------

YOUNG_PATTERNS = [
    r"\bbab(y|ies)\b",
    r"\binfants?\b",
    r"\btoddlers?\b",
    r"\b\d{1,2}\s*-\s*\d{1,2}\s*months?\b",
    r"\b\d{1,2}\s*months?\b",
    r"\bpre-?school(er)?s?\b",
    r"\bnursery\b",
    r"\bday\s?care\b",
    r"\bearly years\b",
    r"\bbabbl(e|ing)\b",
    r"\bfirst words?\b",
    # Infant language project (UPF "Before first words")
    r"beforefirstwords?",
]

OLDER_PATTERNS = [
    r"\belementary\b",
    r"\bschool[- ]age\b",
    r"\b(1st|2nd|3rd|[4-9]th) grade\b",
    r"\bteenagers?\b",
]

# "3-year-old", "4 years old", "(2-4 years old)", "3yrs".
# Plain "N years" is skipped: in descriptions it is
# usually staff experience or company age.
AGE_IN_YEARS = re.compile(
    r"\b(\d{1,2})(?:\s*-\s*\d{1,2})?\s*-?\s*(?:yrs?\b|years?[- ]old\b)",
    re.IGNORECASE
)


def age_hint(title, description="", keywords=""):

    # The title is the most specific; only fall back to
    # description / keywords when it says nothing.
    hint, reasons = _hint_for(title)

    if hint != "unknown":
        return hint, reasons

    return _hint_for(description, keywords)


def _hint_for(*texts):

    text = " ".join(t for t in texts if isinstance(t, str))

    young = [
        m.group(0) for p in YOUNG_PATTERNS
        for m in [re.search(p, text, re.IGNORECASE)] if m
    ]

    older = [
        m.group(0) for p in OLDER_PATTERNS
        for m in [re.search(p, text, re.IGNORECASE)] if m
    ]

    for match in AGE_IN_YEARS.finditer(text):
        (young if int(match.group(1)) < 5 else older).append(match.group(0))

    if young and not older:
        hint = "likely_under5"
    elif older and not young:
        hint = "likely_5plus"
    elif young and older:
        hint = "mixed"
    else:
        hint = "unknown"

    return hint, sorted(set(young + older), key=str.lower)
