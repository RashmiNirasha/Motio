import re


# --------------------------------------------------
# Safety rules that do NOT depend on the LLM.
#
# 1. Before the LLM call: notes about visible objects for the
#    child's age (e.g. small parts under 3), sent with the
#    observation so suggestions take them into account.
# 2. After the LLM call: the suggestion is checked for items
#    that are hazardous at this age and for diagnostic wording.
#    A failing suggestion is retried once, then replaced with a
#    safe default.
#
# These lists are deliberately conservative starting points for
# a research prototype, not a complete child-safety standard.
# --------------------------------------------------

UNDER_3 = 36  # months
UNDER_5 = 60

# Visible-object notes, by detector label: (applies below age in months, note)
OBJECT_NOTES = {
    "block": (UNDER_3, "Use only blocks too large to fit in the mouth (no small bricks)."),
    "puzzle piece": (UNDER_3, "Small puzzle pieces are a choking hazard; use large chunky pieces."),
    "ball": (UNDER_3, "Balls smaller than about 4.5 cm across are a choking hazard."),
    "crayon": (UNDER_3, "Use chunky crayons with close supervision; children this age mouth them."),
    "pencil": (UNDER_5, "Pointed; supervise closely and prefer chunky crayons."),
    "pen": (UNDER_5, "Pointed, and caps are a choking hazard; supervise closely."),
    "bottle": (UNDER_3, "Remove small caps and lids."),
    "play dough": (UNDER_3, "Supervise so it is not eaten; it can be a choking hazard."),
    "toy car": (UNDER_3, "Check wheels and small parts are secure."),
    "doll": (UNDER_3, "Check there are no small detachable parts."),
    "stuffed toy": (UNDER_3, "Check eyes and buttons are secure."),
}

# Items a suggestion must not include, by age (months): matched as whole words,
# singular or plural.
HAZARDS = [
    (UNDER_3, [
        "coin", "button battery", "battery", "magnet", "marble", "balloon",
        "bead", "button", "small ball", "dried bean", "grape", "nut",
        "popcorn", "hard candy", "cord", "string", "ribbon", "plastic bag",
        "small part", "lego",
        # Wording that invites small items (e.g. "a small toy block")
        "small block", "small toy", "small piece", "small object", "small item",
        "tiny"
    ]),
    (UNDER_5, [
        "knife", "button battery", "magnet", "balloon", "plastic bag"
    ]),
]

# Wording that would present observations as a developmental assessment.
DIAGNOSTIC = re.compile(
    r"\b("
    r"delay(?:ed|s)?|diagnos\w*|disorder\w*|autis\w*|adhd|deficit\w*|"
    r"impair\w*|abnormal\w*|red flags?|below (?:average|age)|"
    r"behind (?:other|for (?:his|her|their) age)|not meeting milestones?|"
    r"developmental concern\w*"
    r")\b",
    re.IGNORECASE
)


# Caregiver questions about development, delay or milestones. These get a
# fixed answer prefix whatever the model says (see DEVELOPMENT_NOTE).
DEVELOPMENT_QUESTION = re.compile(
    r"\b("
    r"(?:is|are|am|seem|seems|falling|fall|be|being) behind|behind (?:for|other|his|her|their|in (?:his|her|their))|"
    r"delay(?:ed|s)?|milestones?|on track|"
    r"normal for (?:his|her|their|my|an?)\b[\w\s-]{0,15}\bage|"
    r"(?:is|are) (?:this|that|it|they|he|she) normal|"
    r"developing (?:normally|properly|well|ok|okay|slowly)|"
    r"(?:his|her|their|my child'?s|my baby'?s|child'?s) development|developmental|"
    r"autis\w*|adhd|"
    r"should (?:he|she|they|my child|my baby|my son|my daughter) (?:be able|be doing|be using)|"
    r"assess (?:his|her|their|my)|worried about (?:his|her|their|my)"
    r")\b",
    re.IGNORECASE
)

DEVELOPMENT_NOTE = (
    "I can't assess how your child is developing from a few moments of camera "
    "observation, and I don't compare children with milestones. If you have "
    "questions about development, your health visitor, paediatrician or an "
    "early-childhood professional is the right person to ask."
)

AGE_TIP = (
    "Tip: set your child's age in the app so suggestions fit their age."
)


def is_development_question(question):

    return bool(DEVELOPMENT_QUESTION.search(question or ""))


def object_notes(labels, age_months):

    notes = []

    for label in sorted(set(labels)):

        rule = OBJECT_NOTES.get(label)

        if rule and (age_months is None or age_months < rule[0]):
            notes.append({"object": label, "note": rule[1]})

    return notes


def hazards_for_age(age_months):

    # Unknown age: apply the strictest rules.
    words = set()

    for below, items in HAZARDS:
        if age_months is None or age_months < below:
            words.update(items)

    return sorted(words)


def _find_words(text, words):

    found = []

    for word in words:

        pattern = r"\b" + re.escape(word) + r"(?:s|es)?\b"

        if re.search(pattern, text, re.IGNORECASE):
            found.append(word)

    return found


def check_reply(reply, age_months):

    # reply: dict with observation_summary, confidence_note, reply, activity{...}
    activity = reply.get("activity") or {}

    activity_text = " ".join(
        [
            activity.get("title", ""),
            " ".join(activity.get("objects_needed", [])),
            " ".join(activity.get("instructions", [])),
            activity.get("easier_version", ""),
            activity.get("harder_version", "")
        ]
    )

    problems = []

    hazards = _find_words(activity_text, hazards_for_age(age_months))

    if hazards:
        problems.append(
            "The activity includes items that are unsafe at this age: "
            + ", ".join(hazards)
        )

    # The conversational reply may legitimately say "I can't assess
    # delays" when asked, so diagnostic wording is only checked in the
    # observation summary and the activity.
    assessed = " ".join(
        [
            reply.get("observation_summary", ""),
            activity_text,
            activity.get("why_it_helps", "")
        ]
    )

    diagnostic = sorted({m.group(0).lower() for m in DIAGNOSTIC.finditer(assessed)})

    if diagnostic:
        problems.append(
            "The summary or activity uses assessment wording: " + ", ".join(diagnostic)
        )

    return problems


def safe_default_activity(age_months):

    # Used only when a generated suggestion fails the checks twice.
    if age_months is not None and age_months < 18:

        return {
            "title": "Put in, take out",
            "objects_needed": ["a large bowl or box", "3-4 large soft toys or big blocks"],
            "uses_only_visible_objects": False,
            "instructions": [
                "Sit facing your child with the bowl between you.",
                "Drop one toy into the bowl and say \"in!\".",
                "Hand your child a toy and let them try; tip the toys out together and repeat."
            ],
            "why_it_helps": "Picking up and letting go of objects gives practice with grasping and releasing.",
            "skills_practised": ["grasp and release", "hand-eye coordination"],
            "safety_notes": ["Use only items too large to fit in the mouth.", "Stay with your child."],
            "easier_version": "Hold the bowl close to their hand so the toy only needs to be let go.",
            "harder_version": "Use a box with a hole in the lid to post the toys through."
        }

    return {
        "title": "Build a tower together",
        "objects_needed": ["4-6 large blocks or stacking cups"],
        "uses_only_visible_objects": False,
        "instructions": [
            "Place two large blocks in front of your child.",
            "Stack one on the other and invite them to add the next.",
            "Count the blocks together, then knock the tower down and start again."
        ],
        "why_it_helps": "Placing one block on another gives practice with controlled grasping, careful release and hand-eye coordination.",
        "skills_practised": ["grasp and release", "hand-eye coordination", "two-handed play"],
        "safety_notes": ["Use only blocks too large to fit in the mouth.", "Stay with your child."],
        "easier_version": "Start with two big blocks and help steady the bottom one.",
        "harder_version": "Try building a taller tower or a bridge with three blocks."
    }
