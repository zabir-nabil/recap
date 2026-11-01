"""Generate five template-based paraphrase candidates per FinanceBench question."""

import json
import re
from pathlib import Path


QUARTER_TO_WORD = {"Q1": "first", "Q2": "second", "Q3": "third", "Q4": "fourth"}
WORD_TO_QUARTER = {v: k for k, v in QUARTER_TO_WORD.items()}


def expand_quarter(text):
    """Q2 2023 → the second quarter of 2023"""
    def replacer(m):
        q, yr = m.group(1), m.group(2)
        return f"the {QUARTER_TO_WORD.get(q.upper(), q)} quarter of {yr}"
    return re.sub(r"\b(Q[1-4])\s+(?:of\s+)?(\d{4})\b", replacer, text, flags=re.IGNORECASE)


def contract_quarter(text):
    """the second quarter of 2023 → Q2 2023"""
    def replacer(m):
        word, yr = m.group(1).lower(), m.group(2)
        return f"{WORD_TO_QUARTER.get(word, word)} {yr}"
    return re.sub(
        r"\bthe\s+(first|second|third|fourth)\s+quarter\s+of\s+(\d{4})\b",
        replacer, text, flags=re.IGNORECASE
    )


# Matches year ranges like FY2015 - FY2017, FY2015 to FY2017, FY2015–FY2017
RANGE_RE = re.compile(r"FY\d{4}\s*(?:[-–]|to)\s*FY\d{4}", re.IGNORECASE)

TIME_PATTERNS = [
    r"as of (?:the\s+)?(?:Q[1-4]\s+(?:of\s+)?\d{4}|the \w+ quarter of \d{4}|\d{4})",
    r"for (?:fiscal year |fy|the year )?(?:ended\s+\S+\s+\d+,?\s+)?\d{4}",
    r"in (?:fiscal year |fy)?\d{4}",
    r"during (?:fiscal year |fy)?\d{4}",
    r"for the (?:first|second|third|fourth) quarter of \d{4}",
    r"as of (?:the )?(?:first|second|third|fourth) quarter of \d{4}",
    r"in (?:Q[1-4]) (?:of )?\d{4}",
    r"\bFY\d{4}\b",
    r"\bQ[1-4]\s+(?:of\s+)?(?:FY)?\d{4}\b",
    r"based on (?:fy|fiscal year )?\d{4}(?: data)?",
]
TIME_RE = re.compile("|".join(TIME_PATTERNS), re.IGNORECASE)


def extract_time_phrase(q):
    """Return (time_phrase, q_without_time) or (None, q).
    Also consumes the preceding preposition (as of, in, during, for) if present.
    IMPORTANT: If the question contains a year range (FY2015-FY2017), return (None, q)
    to avoid breaking the range by extracting just one year.
    """
    if RANGE_RE.search(q):
        return None, q
    m = TIME_RE.search(q)
    if not m:
        return None, q
    phrase = m.group(0)
    start, end = m.start(), m.end()
    # Expand phrase backward to include a preceding preposition
    prefix = q[:start]
    prep_m = re.search(r"\b(as\s+of(?:\s+the)?|in\s+(?:the\s+)?|during\s+|for\s+)\s*$", prefix, re.IGNORECASE)
    if prep_m:
        phrase = prep_m.group(0).rstrip() + " " + phrase
        start = prep_m.start()
    rest = (q[:start] + q[end:]).strip()
    # Remove leading/trailing commas and whitespace
    rest = re.sub(r"^[,\s]+|[,\s]+$", "", rest).strip()
    return phrase.strip(), rest


# ── Patterns that strip META-INSTRUCTIONS from questions ──
# ORDER MATTERS: broader/leading patterns first, trailing patterns last.
INSTRUCTION_PATTERNS = [
    # ── LEADING preambles (indices 0-11) ──────────────────────────────────────
    re.compile(r"^Assume\s+that\s+you\s+are\s+.+?\.\s*", re.IGNORECASE | re.DOTALL),
    re.compile(r"^Answer\s+the\s+following\s+question\s+.+?:\s*", re.IGNORECASE | re.DOTALL),
    re.compile(r"^Using\s+(?:only\s+)?the\s+(?:information\s+(?:within|provided\s+in)|cash\s+flow\s+statement|balance\s+sheet|details\s+\w+\s+outlined\s+within)\b.+?[,]\s*", re.IGNORECASE | re.DOTALL),
    re.compile(r"^According\s+to\s+the\s+(?:information\s+provided\s+in|details\s+\w+\s+outlined\s+within)\b.+?[,]\s*", re.IGNORECASE | re.DOTALL),
    # ALL "Basing your X ..." forms — ends with comma OR period
    re.compile(r"^Basing\s+your\s+.+?[,\.]\s*", re.IGNORECASE | re.DOTALL),
    re.compile(r"^Give\s+a\s+response\s+to\s+the\s+question\s+.+?:\s*", re.IGNORECASE | re.DOTALL),
    # "We want/need to calculate..." preambles ending with period or question mark
    re.compile(r"^We\s+(?:want|need)\s+to\s+(?:calculate|compute|determine)\s+.+?[.?]\s*", re.IGNORECASE | re.DOTALL),
    re.compile(r"^Please\s+help\s+us\s+(?:compute|calculate|determine)\s+.+?[.?]\s*", re.IGNORECASE | re.DOTALL),
    re.compile(r"^(?:We\s+need\s+to|We\s+want\s+to|Please)\s+calculate\s+.+?[.?]\s*", re.IGNORECASE | re.DOTALL),
    # Generic "We ..." meta-instruction sentence ending with period before real question
    re.compile(r"^We\s+(?:want|need|are\s+trying|would\s+like)\s+.+?[.]\s*(?=[A-Z])", re.IGNORECASE | re.DOTALL),
    # "Please help us compute it by basing your..." whole sentence
    re.compile(r"Please\s+help\s+us\s+compute\s+it\s+by\s+basing\s+your\s+.+?[.?]\s*", re.IGNORECASE | re.DOTALL),
    # Any sentence containing "basing your judgment/answer/..." ending with comma or period
    re.compile(r"^.{0,300}?basing\s+your\s+(?:judgment|answer|calculation|response)\s+.+?[,\.]\s*", re.IGNORECASE | re.DOTALL),
    # Strip "Here's the question:" and "Please answer the following question:" introducers
    re.compile(r"^(?:Here's|Here\s+is)\s+the\s+question:\s*", re.IGNORECASE | re.DOTALL),
    re.compile(r"^Please\s+answer\s+the\s+following\s+question:\s*", re.IGNORECASE | re.DOTALL),
    re.compile(r"^In\s+agreement\s+with\s+.+?[,]\s*", re.IGNORECASE | re.DOTALL),
    re.compile(r"^Taking\s+into\s+account\s+.+?[,]\s*", re.IGNORECASE | re.DOTALL),
    re.compile(r"^Based\s+on\s+the\s+information\s+provided\s+.+?[,]\s*", re.IGNORECASE | re.DOTALL),
    # ── TRAILING instructions (indices 17+) ───────────────────────────────────
    re.compile(r"[.?]?\s*Define\s+\w[\w\s-]+\s+as\s+.+$", re.IGNORECASE | re.DOTALL),
    re.compile(r"[.?]?\s*Answer\s+in\s+(?:USD|EUR|GBP|CAD|\$|millions|billions|thousands)\b.+$", re.IGNORECASE | re.DOTALL),
    re.compile(r"[.?]?\s*Please\s+state\s+(?:the\s+)?answer\s+in\s+.+$", re.IGNORECASE | re.DOTALL),
    re.compile(r"[.?]?\s*Provide\s+(?:the\s+)?answer\s+in\s+.+$", re.IGNORECASE | re.DOTALL),
    re.compile(r"[.?]?\s*Express\s+(?:the\s+)?answer\s+in\s+.+$", re.IGNORECASE | re.DOTALL),
    re.compile(r"[.?]?\s*Give\s+a\s+response\s+to\s+.+$", re.IGNORECASE | re.DOTALL),
    re.compile(r"[.?]?\s*(?:Provide|State)\s+in\s+(?:USD|EUR|billions|millions|thousands)\b.+$", re.IGNORECASE | re.DOTALL),
]

# Number of LEADING patterns (update this if you add/remove leading patterns above)
_N_LEADING = 17

# ── Patterns that strip CONDITIONAL CLAUSES used only for template building ──
# These are NOT stripped from the original question — only from paraphrase templates.
CONDITIONAL_STRIP_RE = re.compile(
    r"\s*[.?]?\s*If\s+(?:\w+\s+){1,6}(?:is|are|was|were)\s+not\s+.+$",
    re.IGNORECASE | re.DOTALL
)


def clean_question(q: str) -> str:
    """
    Remove leading preambles and trailing unit/definitional instructions.
    Preserves 'If X is not useful, explain why' — that IS part of the question.
    Applies trailing patterns to text WITHOUT question mark so regex anchors work cleanly.
    """
    q = q.strip()
    LEADING = INSTRUCTION_PATTERNS[:_N_LEADING]
    TRAILING = INSTRUCTION_PATTERNS[_N_LEADING:]
    # Apply leading patterns repeatedly until stable (handles chained multi-sentence preambles)
    for _ in range(4):
        prev = q
        for pat in LEADING:
            q = pat.sub("", q).strip()
        if q == prev:
            break
    # For trailing patterns, strip question mark temporarily so "$" anchors work
    q_noq = q.rstrip("?").rstrip().rstrip(".")
    for pat in TRAILING:
        q_noq = pat.sub("", q_noq).strip()
    q = q_noq
    # Normalize quarter format: q2 → Q2, Q2 of 2023 → Q2 2023
    q = re.sub(r"\b(q[1-4])\b", lambda m: m.group(1).upper(), q)
    q = re.sub(r"\b(Q[1-4])\s+of\s+(\d{4})\b", r"\1 \2", q)
    # Remove double spaces
    q = re.sub(r"\s{2,}", " ", q).strip()
    return q


def core_for_templates(q_clean: str) -> str:
    """
    Strip conditional 'If X is not useful...' clause for use INSIDE paraphrase templates only.
    The original q_clean is preserved for the first paraphrase (exact question).
    """
    stripped = CONDITIONAL_STRIP_RE.sub("", q_clean).strip().rstrip("?").strip()
    return stripped if stripped else q_clean


def ensure_q(text: str) -> str:
    """Capitalize first letter and ensure ends with exactly one '?'.
    Also fixes mid-sentence capitalization after commas in fronted questions."""
    text = text.strip()
    text = text.rstrip("?.!").strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    # Fix "As of FY2022, What drove..." → "As of FY2022, what drove..."
    # i.e. lowercase the first word after a comma if it's a question word
    text = re.sub(
        r"(,\s+)(What|How|Which|Why|Where|When|Who)\b",
        lambda m: m.group(1) + m.group(2).lower(),
        text
    )
    return text + "?"


def strip_q(text: str) -> str:
    return text.strip().rstrip("?").strip()


def clean_double_articles(text: str) -> str:
    """Fix 'the the', 'a a', 'an an' and double spaces."""
    text = re.sub(r"\b(the|a|an)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return text


# Subject-verb agreement

# Companies treated as singular subjects
COMPANY_PAT = re.compile(
    r"\b(3M|Apple|Google|Microsoft|Amazon|Meta|Tesla|Nike|Boeing|Pfizer|"
    r"Walmart|Target|Costco|CVS|Intel|AMD|Nvidia|Cisco|Oracle|IBM|"
    r"Chevron|ExxonMobil|ConocoPhillips|Shell|BP|"
    r"JPMorgan|Goldman|Wells Fargo|Citigroup|"
    r"Johnson|Merck|Abbott|Amgen|Gilead|"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Ltd|LLC|Co|Group|Holdings)\.?)\b"
)

# Maps (wrong_verb → correct_verb) when subject is a singular company
SVA_FIXES = {
    " have ": " has ",
    " are ": " is ",
    " maintain ": " maintains ",
    " operate ": " operates ",
    " generate ": " generates ",
    " report ": " reports ",
    " show ": " shows ",
    " experience ": " experiences ",
}


def fix_sva(text: str) -> str:
    """
    Fix subject-verb agreement when a singular company name is the subject
    of a declarative clause (not a question inversion).
    Only fires when company name immediately precedes the verb phrase.
    """
    for wrong, right in SVA_FIXES.items():
        # Pattern: Company<wrong_verb> — but NOT after "Does/Is/Has" inversion
        pattern = re.compile(
            rf"({COMPANY_PAT.pattern}){re.escape(wrong)}",
            re.IGNORECASE
        )
        def replacer(m, right=right):
            # Don't fix if this is already a question inversion (verb before subject)
            before = text[:m.start()].lower()
            if re.search(r"\b(does|is|has|do|are|have|did|was|were|can|could|would|should)\b", before):
                return m.group(0)
            return m.group(1) + right
        text = pattern.sub(replacer, text)
    return text


def normalize_for_dedup(text: str) -> str:
    """Aggressive normalization for near-duplicate detection."""
    t = text.lower().strip().rstrip("?")
    t = re.sub(r"\b(the|a|an|its|their|this|that|these|those)\b", " ", t)
    t = re.sub(r"\bfy(\d{4})\b", r"fiscal year \1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def deduplicate(paraphrases):
    seen_exact = set()
    seen_norm  = set()
    out = []
    for p in paraphrases:
        if not p or len(p) < 8:
            continue
        exact = p.lower().strip()
        norm  = normalize_for_dedup(p)
        if exact not in seen_exact and norm not in seen_norm:
            out.append(p)
            seen_exact.add(exact)
            seen_norm.add(norm)
    return out


# These functions take a CLEAN question (no instructions, no double articles)

TENSE_FLIP = {
    "are": "were", "were": "are", "is": "was", "was": "is",
    "have": "had", "had": "have", "do": "did", "did": "do",
    "does": "did", "has": "had"
}


def paraphrase_what_is(q_clean: str, original: str):
    """
    Handles: "What is/was/are/were <NP> for <company>?"
    Generates: temporal fronting, NP-fronting, natural interrogatives.
    """
    verb_m = re.match(r"(?i)what (is|was|are|were)\b", q_clean)
    verb   = verb_m.group(1) if verb_m else "is"
    core   = strip_q(re.sub(r"(?i)^what (is|was|are|were)\s+", "", q_clean))

    # Use template-safe core (strips "If X is not useful..." conditional)
    core_tmpl = core_for_templates(core)
    time_phrase, core_no_time = extract_time_phrase(core_tmpl)
    core_no_time = core_no_time.strip().rstrip(",").strip()

    if time_phrase:
        tp_exp = expand_quarter(time_phrase)
        tp_con = contract_quarter(time_phrase)
        if re.match(r"^(as of|for|during|in)\b", time_phrase, re.IGNORECASE):
            fronted = f"{time_phrase.capitalize()}, what {verb} {core_no_time}"
        else:
            fronted = f"In {tp_exp}, what {verb} {core_no_time}"
        core_bare = re.sub(r"^(the|a|an)\s+", "", core_no_time, flags=re.IGNORECASE)
        paraphrases = [
            ensure_q(f"What {verb} {core_tmpl}"),
            ensure_q(fronted),
            ensure_q(f"How much {verb} {core_no_time} {time_phrase}"),
            ensure_q(f"What {verb} the reported {core_bare} for {tp_con}"),
            ensure_q(f"What {verb} the reported {core_bare} {tp_con}"),
        ]
    else:
        verb_alt = TENSE_FLIP.get(verb.lower(), verb)
        paraphrases = [
            ensure_q(f"What {verb} {core_tmpl}"),
            ensure_q(f"What {verb_alt} {core_tmpl}"),
            ensure_q(f"How much {verb} {core_tmpl}"),
            ensure_q(f"What figure {verb} reported for {core_tmpl}"),
            ensure_q(f"What amount {verb} recorded for {core_tmpl}"),
        ]
    return paraphrases, core_no_time, time_phrase


def paraphrase_how_much(q_clean: str, original: str):
    core_raw = strip_q(re.sub(r"(?i)^how (much|many)\s+", "", q_clean))
    core = core_for_templates(core_raw)
    time_phrase, core_no_time = extract_time_phrase(core)
    core_no_time = core_no_time.strip().rstrip(",").strip()
    tp_suffix = f" {time_phrase}" if time_phrase else ""

    # If core contains "did X have/report/record Y", extract the NP cleanly
    # e.g. "total assets did Corning have in FY2021" → noun="total assets", subj="Corning"
    didx_m = re.match(r"^(.+?)\s+did\s+(\S+(?:\s+\S+){0,2}?)\s+(?:have|report|record|hold|carry)\b", core_no_time or core, re.IGNORECASE)
    if didx_m:
        noun = didx_m.group(1).strip()  # e.g. "total assets"
        subj = didx_m.group(2).strip()  # e.g. "Corning"
        if time_phrase:
            paraphrases = [
                ensure_q(q_clean),
                ensure_q(f"What was {subj}'s {noun}{tp_suffix}"),
                ensure_q(f"What {noun} was reported for {subj}{tp_suffix}"),
                ensure_q(f"How much {noun} did {subj} report{tp_suffix}"),
                ensure_q(f"What did {subj} record as {noun}{tp_suffix}"),
            ]
        else:
            paraphrases = [
                ensure_q(q_clean),
                ensure_q(f"What was {subj}'s {noun}"),
                ensure_q(f"What is the value of {subj}'s {noun}"),
                ensure_q(f"What figure was reported for {subj}'s {noun}"),
                ensure_q(f"What did {subj} record as {noun}"),
            ]
        return paraphrases, noun, time_phrase

    if time_phrase:
        has_prep = re.match(r"^(in|as of|for|during)\b", time_phrase, re.IGNORECASE)
        if has_prep:
            fronted = ensure_q(f"{time_phrase.capitalize()}, how much {core_no_time}")
        else:
            fronted = ensure_q(f"In {expand_quarter(time_phrase)}, how much {core_no_time}")
        # Avoid "total total X" — strip leading "total" from core_no_time before prepending
        core_no_total = re.sub(r"^total\s+", "", core_no_time, flags=re.IGNORECASE)
        paraphrases = [
            ensure_q(q_clean),
            ensure_q(f"What was the total {core_no_total}{tp_suffix}"),
            ensure_q(f"What figure was reported for {core_no_time}{tp_suffix}"),
            fronted,
            ensure_q(f"What did the company record as {core_no_time}{tp_suffix}"),
        ]
    else:
        core_no_total = re.sub(r"^total\s+", "", core, flags=re.IGNORECASE)
        paraphrases = [
            ensure_q(q_clean),
            ensure_q(f"What was the total {core_no_total}"),
            ensure_q(f"What is the value of {core}"),
            ensure_q(f"What figure was reported for {core}"),
            ensure_q(f"What did the company record as {core}"),
        ]
    return paraphrases, core_no_time, time_phrase


def paraphrase_which(q_clean: str, original: str):
    q_tmpl = core_for_templates(q_clean)
    time_phrase, _ = extract_time_phrase(strip_q(q_tmpl))
    m = re.match(
        r"(?i)which\s+(.+?)\s+"
        r"(are|is|were|was|have|has|do|does|did|can|could|would|will|should)\b(.*)",
        q_tmpl
    )
    if not m:
        core = strip_q(q_tmpl[5:])
        paraphrases = [
            ensure_q(f"Which {core}"),
            ensure_q(f"What {core}"),
            ensure_q(f"Identify the {core}"),
            ensure_q(f"Name the {core}"),
            ensure_q(f"Can you specify which {core}"),
        ]
        return paraphrases, core, time_phrase

    noun     = m.group(1).strip()
    verb     = m.group(2).strip()
    pred_raw = strip_q(m.group(3).strip())
    verb_alt = TENSE_FLIP.get(verb.lower(), verb)
    pred_no_time = TIME_RE.sub("", pred_raw).strip().rstrip(",").strip() if time_phrase else pred_raw

    if time_phrase:
        tp_exp = expand_quarter(time_phrase)
        tp_con = contract_quarter(time_phrase)
        if re.match(r"^(as of|for|during)\b", time_phrase, re.IGNORECASE):
            front = f"{time_phrase.capitalize()}, what {noun} {verb} {pred_no_time}"
        else:
            front = f"In {tp_exp}, what {noun} {verb} {pred_no_time}"
        paraphrases = [
            ensure_q(f"Which {noun} {verb} {pred_raw}"),
            ensure_q(front),
            ensure_q(f"Identify the {noun} that {verb} {pred_no_time} {time_phrase}"),
            ensure_q(f"Name the {noun} that {verb_alt} {pred_no_time} {tp_con}"),
            ensure_q(f"Can you specify which {noun} {verb} {pred_no_time} {time_phrase}"),
        ]
    else:
        paraphrases = [
            ensure_q(f"Which {noun} {verb} {pred_raw}"),
            ensure_q(f"What {noun} {verb} {pred_raw}"),
            ensure_q(f"Identify the {noun} that {verb} {pred_raw}"),
            ensure_q(f"Name the {noun} that {verb_alt} {pred_raw}"),
            ensure_q(f"Can you specify which {noun} {verb} {pred_raw}"),
        ]
    return paraphrases, pred_no_time, time_phrase


def paraphrase_yes_no(q_clean: str, original: str):
    """
    Yes/No questions: produce structurally diverse paraphrases.
    Uses core_for_templates() so 'If X is not useful...' never leaks into templates.
    """
    # Strip the conditional clause for all template use
    q_tmpl = core_for_templates(strip_q(q_clean))

    lead_m = re.match(
        r"(?i)^(does|is|are|did|has|have|was|were|can|could|would|should|will)\s+(.*)",
        q_tmpl
    )
    if not lead_m:
        paraphrases = [
            ensure_q(q_clean),                                              # original (full)
            ensure_q(f"To what extent {q_tmpl}"),
            ensure_q(f"How would you assess whether {q_tmpl}"),
            ensure_q(f"What evidence supports whether {q_tmpl}"),
            ensure_q(f"Based on the financials, {q_tmpl}"),
        ]
        return paraphrases, None, None

    lead_verb  = lead_m.group(1)
    rest       = lead_m.group(2).strip()   # e.g. "Paypal have positive working capital based on FY2022 data"
    lead_lower = lead_verb.lower()

    if lead_lower in ("does", "did", "do"):
        # rest = "<subj> <verb_base> <pred>"
        # Match any verb (not just a curated list) — take first word(s) as subj, next word as verb
        subj_verb_m = re.match(
            r"^(.+?)\s+(have|maintain|report|generate|show|exceed|include|carry|hold|"
            r"grow|increase|decrease|decline|improve|produce|operate|represent|reflect|"
            r"consist|contain|record|achieve|experience|exhibit|demonstrate|indicate)\b\s*(.*)",
            rest, re.IGNORECASE
        )
        if not subj_verb_m:
            # Generic verb: take first token(s) as subject, rest as predicate
            generic_m = re.match(r"^(\S+(?:\s+\S+){0,2}?)\s+(\w+)\s+(.*)", rest)
            if generic_m:
                subj, vb, pred = generic_m.group(1).strip(), generic_m.group(2).strip(), generic_m.group(3).strip()
            else:
                subj, vb, pred = rest, "", ""
        else:
            subj = subj_verb_m.group(1).strip()
            vb   = subj_verb_m.group(2).strip()
            pred = subj_verb_m.group(3).strip()

        # Map base verb to 3rd-person singular for "Does/Has" contexts
        _vb3 = {"have":"has","maintain":"maintains","report":"reports","generate":"generates",
                "show":"shows","exceed":"exceeds","include":"includes","carry":"carries",
                "hold":"holds","grow":"grows","increase":"increases","decrease":"decreases",
                "decline":"declines","improve":"improves","produce":"produces","operate":"operates",
                "represent":"represents","reflect":"reflects","record":"records",
                "achieve":"achieves","experience":"experiences","exhibit":"exhibits",
                "demonstrate":"demonstrates","indicate":"indicates","consist":"consists",
                "contain":"contains"}.get(vb.lower(), vb + "s")
        _vb_pres = _vb3 if lead_lower in ("does",) else vb
        if pred:
            # For "did" questions use past tense; for "does" use 3rd-person present
            _PAST = {"have":"had","maintain":"maintained","report":"reported","generate":"generated",
                     "show":"showed","exceed":"exceeded","include":"included","carry":"carried",
                     "hold":"held","grow":"grew","increase":"increased","decrease":"decreased",
                     "decline":"declined","improve":"improved","produce":"produced","operate":"operated",
                     "record":"recorded","achieve":"achieved","experience":"experienced"}
            _vb_stmt = _PAST.get(vb.lower(), vb + "ed") if lead_lower == "did" else _vb3
            paraphrases = [
                ensure_q(f"{lead_verb} {rest}"),
                ensure_q(f"Would you say that {subj} {_vb_stmt} {pred}"),
                ensure_q(f"Can it be confirmed that {subj} {_vb_stmt} {pred}"),
                ensure_q(f"Is it the case that {subj} {_vb_stmt} {pred}"),
                ensure_q(f"Is it true that {subj} {_vb_stmt} {pred}"),
            ]
        else:
            paraphrases = [
                ensure_q(f"{lead_verb} {rest}"),
                ensure_q(f"Would you say that {lead_verb} {rest}"),
                ensure_q(f"Can it be confirmed that {lead_verb} {rest}"),
                ensure_q(f"Is it true that {lead_verb} {rest}"),
                ensure_q(f"Is it the case that {lead_verb} {rest}"),
            ]

    elif lead_lower in ("is", "was", "are", "were"):
        # Try: "Best Buy's gross margins historically consistent (...)"
        # Extract subject + predicate cleanly
        subj_m = re.match(
            r"^([A-Z0-9][A-Za-z0-9&'\-\.]*(?:\s+[A-Z][A-Za-z0-9&'\-\.]*)*(?:'s)?)\s+((?:a|an|the|not|[a-z\(]).+)",
            rest
        )
        if subj_m:
            subj = subj_m.group(1).strip()
            pred = subj_m.group(2).strip()
            # Strip possessive apostrophe-s from subj for "subj's pred" form
            subj_base = re.sub(r"'s$", "", subj)
            # Truncate pred at opening paren for cleaner template (keep the paren in original)
            pred_short = re.sub(r"\s*\(.*$", "", pred).strip()
            # Determine connector: if pred starts with "a/an" it's a classification → "as a/an"
            # e.g. pred="a capital-intensive business" → "Would you classify X as a capital-intensive business"
            pred_article_m = re.match(r"^(a|an)\b(.+)", pred_short, re.IGNORECASE)
            if pred_article_m:
                art  = pred_article_m.group(1)
                rest_pred = pred_article_m.group(2).strip()
                paraphrases = [
                    ensure_q(f"{lead_verb} {rest}"),
                    ensure_q(f"Would you consider {subj_base} to be {art} {rest_pred}"),
                    ensure_q(f"Can {subj_base} be considered {art} {rest_pred}"),
                    ensure_q(f"Is it reasonable to classify {subj_base} as {art} {rest_pred}"),
                    ensure_q(f"Would you say {subj_base} is {art} {rest_pred}"),
                ]
            else:
                # pred does NOT start with article — e.g. "improving", "consistent", "healthy"
                paraphrases = [
                    ensure_q(f"{lead_verb} {rest}"),
                    ensure_q(f"Would you say that {subj_base} {lead_lower} {pred_short}"),
                    ensure_q(f"Can it be said that {subj_base} {lead_lower} {pred_short}"),
                    ensure_q(f"Is it true that {subj_base} {lead_lower} {pred_short}"),
                    ensure_q(f"Does the data suggest that {subj_base} {lead_lower} {pred_short}"),
                ]
        else:
            # Fallback: subj not matched — use generic yes/no forms
            pred_article_m2 = re.match(r"^(a|an)\b(.+)", rest, re.IGNORECASE)
            if pred_article_m2:
                art2  = pred_article_m2.group(1)
                rest2 = pred_article_m2.group(2).strip()
                # Try to extract subject from rest before the article
                # e.g. rest = "3M a capital-intensive business..."
                subj2_m = re.match(r"^(\S+(?:\s+\S+){0,2}?)\s+(a|an)\b(.+)", rest, re.IGNORECASE)
                if subj2_m:
                    s2 = subj2_m.group(1).strip()
                    a2 = subj2_m.group(2)
                    r2 = subj2_m.group(3).strip()
                    paraphrases = [
                        ensure_q(f"{lead_verb} {rest}"),
                        ensure_q(f"Would you consider {s2} to be {a2} {r2}"),
                        ensure_q(f"Can {s2} be considered {a2} {r2}"),
                        ensure_q(f"Is it reasonable to classify {s2} as {a2} {r2}"),
                        ensure_q(f"Would you say {s2} is {a2} {r2}"),
                    ]
                else:
                    paraphrases = [
                        ensure_q(f"{lead_verb} {rest}"),
                        ensure_q(f"Would you consider this to be {art2} {rest2}"),
                        ensure_q(f"Can it be considered {art2} {rest2}"),
                        ensure_q(f"Is it reasonable to classify this as {art2} {rest2}"),
                        ensure_q(f"Would you say this is {art2} {rest2}"),
                    ]
            else:
                paraphrases = [
                    ensure_q(f"{lead_verb} {rest}"),
                    ensure_q(f"Would you say that {rest}"),
                    ensure_q(f"Can it be confirmed that {rest}"),
                    ensure_q(f"Is it true that {rest}"),
                    ensure_q(f"Is it the case that {rest}"),
                ]

    elif lead_lower in ("has", "have", "had"):
        # Extract subject greedily — take all capitalized words at the start
        # e.g. "CVS Health reported..." → subj="CVS Health", pred="reported..."
        subj_m = re.match(
            r"^([A-Z0-9][A-Za-z0-9&'\-\.]*(?:\s+[A-Z][A-Za-z0-9&'\-\.]*)*(?:'s)?)\s+(.*)",
            rest
        )
        if subj_m:
            subj = subj_m.group(1).strip()
            pred = subj_m.group(2).strip()
            # Clean double-possessive: "AMCOR's" + "'s" → keep just subj without 's
            subj_clean = re.sub(r"'s$", "", subj)
            paraphrases = [
                ensure_q(f"{lead_verb} {rest}"),
                ensure_q(f"Has {subj_clean} {pred}"),
                ensure_q(f"Would you say that {subj_clean} {lead_lower} {pred}"),
                ensure_q(f"Can it be confirmed that {subj_clean} {lead_lower} {pred}"),
                ensure_q(f"Is it true that {subj_clean} {lead_lower} {pred}"),
            ]
        else:
            paraphrases = [
                ensure_q(f"{lead_verb} {rest}"),
                ensure_q(f"Has there been {rest}"),
                ensure_q(f"Would you say that {rest}"),
                ensure_q(f"Can it be confirmed that {rest}"),
                ensure_q(f"Is it true that {rest}"),
            ]

    elif lead_lower in ("can", "could", "would", "should", "will"):
        paraphrases = [
            ensure_q(f"{lead_verb} {rest}"),
            ensure_q(f"Is it reasonable to say that {rest}"),
            ensure_q(f"What does the data suggest about {rest}"),
            ensure_q(f"How likely is it that {rest}"),
            ensure_q(f"Based on the financials, {rest}"),
        ]
    else:
        paraphrases = [ensure_q(q_clean)] * 5

    return paraphrases, None, None


def paraphrase_why(q_clean: str, original: str):
    q_tmpl = core_for_templates(q_clean)
    paraphrases = [
        ensure_q(q_clean),
        ensure_q(re.sub(r"(?i)^why\s+", "What is the reason ", q_tmpl)),
        ensure_q(re.sub(r"(?i)^why\s+", "What caused ", q_tmpl)),
        ensure_q(re.sub(r"(?i)^why\s+", "What explains ", q_tmpl)),
        ensure_q(re.sub(r"(?i)^why\s+", "What factors led to ", q_tmpl)),
    ]
    return paraphrases, None, None


def paraphrase_how_did(q_clean: str, original: str):
    q_tmpl = core_for_templates(q_clean)
    paraphrases = [
        ensure_q(q_clean),
        ensure_q(re.sub(r"(?i)^how did\s+",   "In what way did ", q_tmpl)),
        ensure_q(re.sub(r"(?i)^how did\s+",   "By what means did ", q_tmpl)),
        ensure_q(re.sub(r"(?i)^how\s+\w+\s+", "Explain how ", q_tmpl)),
        ensure_q(re.sub(r"(?i)^how\s+\w+\s+", "Describe how ", q_tmpl)),
    ]
    return paraphrases, None, None


def paraphrase_what_drove(q_clean: str, original: str):
    # Extract the conditional clause from the ORIGINAL (not the template-stripped version)
    cond_m = re.search(r"(\?\s*If\s+.+)$", q_clean, re.IGNORECASE | re.DOTALL)
    cond_suffix = ""
    if cond_m:
        cond_suffix = " " + cond_m.group(1).strip().lstrip("?").strip()

    q_tmpl = core_for_templates(q_clean)
    time_phrase, core_no_time = extract_time_phrase(strip_q(q_tmpl))
    if time_phrase:
        q_no_time = TIME_RE.sub("", q_tmpl).strip()
        q_no_time = re.sub(r"\bas\s+of(?:\s+the)?\s*$", "", q_no_time, flags=re.IGNORECASE).strip()
        q_no_time = re.sub(r"\bin\s+the\s*$", "", q_no_time, flags=re.IGNORECASE).strip()
        q_no_time = q_no_time.rstrip(",").strip()
    else:
        q_no_time = q_tmpl

    tp_suffix = f" {time_phrase}" if time_phrase else ""

    def _with_cond(text):
        """Append the conditional clause (if any) to a paraphrase."""
        base = ensure_q(text)
        if cond_suffix:
            # Ensure comma before "If"
            cond_clean = cond_suffix.strip()
            if not cond_clean.startswith(","):
                cond_clean = ", " + cond_clean.lstrip()
            base = base.rstrip("?") + cond_clean.rstrip(".?") + "?"
        return base

    # Use core_no_time (cleanly extracted) rather than q_no_time (has dangling prepositions)
    # Strip the leading "What drove" from core_no_time for use in templates
    core_verb_stripped = re.sub(r"\s{2,}", " ", re.sub(r"(?i)^what drove\s+", "", core_no_time).strip()) if core_no_time else ""

    if core_verb_stripped and time_phrase:
        paraphrases = [
            ensure_q(q_clean),
            _with_cond(f"What caused {core_verb_stripped} {time_phrase}"),
            _with_cond(f"What factors explain {core_verb_stripped} {time_phrase}"),
            _with_cond(f"What contributed to {core_verb_stripped} {time_phrase}"),
            _with_cond(f"What was the primary driver of {core_verb_stripped} {time_phrase}"),
        ]
    elif core_verb_stripped:
        paraphrases = [
            ensure_q(q_clean),
            _with_cond(f"What caused {core_verb_stripped}"),
            _with_cond(f"What factors explain {core_verb_stripped}"),
            _with_cond(f"What contributed to {core_verb_stripped}"),
            _with_cond(f"What was the primary driver of {core_verb_stripped}"),
        ]
    else:
        paraphrases = [
            ensure_q(q_clean),
            _with_cond(re.sub(r"(?i)^what drove\b", "What caused", q_tmpl)),
            _with_cond(re.sub(r"(?i)^what drove\b", "What factors explain", q_tmpl)),
            _with_cond(re.sub(r"(?i)^what drove\b", "What contributed to", q_tmpl)),
            _with_cond(re.sub(r"(?i)^what drove\b", "What was the primary driver of", q_tmpl)),
        ]
    return paraphrases, core_no_time, time_phrase


def paraphrase_what_generic(q_clean: str, original: str):
    q_tmpl = core_for_templates(q_clean)
    core = strip_q(re.sub(r"(?i)^what\s+", "", q_tmpl))
    time_phrase, core_no_time = extract_time_phrase(core)
    core_no_time = core_no_time.strip().rstrip(",").strip()
    tp_suffix = f" {time_phrase}" if time_phrase else ""

    # Build a safe "describe" variant — avoid "How would you characterize X" when
    # core starts with a noun phrase that needs "as" (e.g. "3M a capital-intensive business")
    # Instead rephrase as "How would you describe X" which doesn't need "as"
    describe_v = ensure_q(f"How would you describe {core}")
    paraphrases = [
        ensure_q(q_clean),
        describe_v,
        ensure_q(f"What does the financial data indicate about {core_no_time}{tp_suffix}"),
        ensure_q(f"How did {core_no_time} look{tp_suffix}"),
        ensure_q(f"How did {core_no_time} look{tp_suffix}"),
    ]
    return paraphrases, core_no_time, time_phrase


def paraphrase_complex(q_clean: str, original: str):
    """
    For complex/conditional questions: synonym swaps + temporal fronting.
    Uses core_for_templates so 'If X is not useful...' never leaks into variants.
    """
    q_tmpl = core_for_templates(q_clean)

    swap_map = [
        (r"\bdrove\b",       ["caused", "led to", "contributed to"]),
        (r"\bchange\b",      ["shift", "movement", "fluctuation"]),
        (r"\bgrew\b",        ["increased", "expanded", "rose"]),
        (r"\bdeclined\b",    ["fell", "decreased", "dropped"]),
        (r"\bimproved\b",    ["strengthened", "increased", "rose"]),
        (r"\breported\b",    ["disclosed", "posted", "recorded"]),
        (r"\bprimary\b",     ["main", "key", "principal"]),
        (r"\bexcluding\b",   ["excluding", "stripping out", "setting aside"]),
        (r"\bdragged down\b",["weighed on", "held back", "negatively impacted"]),
        (r"\bwent\b",        ["had gone", "were to go"]),
        (r"\bliquidated\b",  ["sold off", "disposed of"]),
        (r"\bsegment\b",     ["division", "business unit"]),
        (r"\boverall\b",     ["total", "aggregate"]),
        (r"\bgrowth\b",      ["expansion", "increase"]),
        (r"\bimpact\b",      ["effect", "influence"]),
        (r"\bexclude\b",     ["strip out", "remove"]),
        (r"\bbankrupted\b",  ["become insolvent", "filed for bankruptcy"]),
    ]

    candidates = [ensure_q(q_clean)]
    # Apply ALL matching swaps (not just first), to get more distinct variants
    for pattern, replacements in swap_map:
        if re.search(pattern, q_tmpl, flags=re.IGNORECASE):
            for rep in replacements[:2]:
                v = ensure_q(re.sub(pattern, rep, q_tmpl, flags=re.IGNORECASE))
                candidates.append(v)
            if len(candidates) >= 4:
                break

    # Temporal fronting — avoid double prepositions and broken "In 2022, If we exclude..."
    time_phrase, rest = extract_time_phrase(strip_q(q_tmpl))
    core_no_time = rest
    if time_phrase:
        tp_raw = time_phrase.strip()
        tp_exp = expand_quarter(tp_raw)
        # Clean dangling prepositions left in rest after time phrase removal
        rest_clean = rest.strip()
        rest_clean = re.sub(r"\bas\s+of(?:\s+the)?\s*$", "", rest_clean, flags=re.IGNORECASE).strip()
        rest_clean = re.sub(r"\bin\s+the\s*$", "", rest_clean, flags=re.IGNORECASE).strip()
        rest_clean = rest_clean.rstrip(",").strip()
        rest_is_conditional = re.match(r"^(if|unless|excluding|given|assuming)\b", rest_clean, re.IGNORECASE)
        has_prep = re.match(r"^(in|as of|for|during)\b", tp_raw, re.IGNORECASE)
        if not rest_is_conditional and rest_clean:
            if not has_prep:
                candidates.append(ensure_q(f"In {tp_exp}, {rest_clean}"))
                candidates.append(ensure_q(f"During {tp_raw}, {rest_clean}"))
            else:
                candidates.append(ensure_q(f"{tp_raw.capitalize()}, {rest_clean}"))
                tp_bare = re.sub(r"^(in|as of|for|during)\s+", "", tp_raw, flags=re.IGNORECASE)
                candidates.append(ensure_q(f"Looking at {tp_bare}, {rest_clean}"))

    # Light structural expansions — only when question doesn't start with if/unless/excluding
    _starts_conditional = re.match(r"(?i)^(if|unless|excluding|given|assuming|suppose)\b", q_tmpl)
    _starts_yesno = re.match(r"(?i)^(does|is|are|did|has|have|was|were|can|could|would|should|will)\b", q_tmpl)
    if not _starts_conditional and not _starts_yesno:
        if re.match(r"(?i)^how\b", q_tmpl):
            candidates.append(ensure_q(re.sub(r"(?i)^how\s+", "Explain how ", q_tmpl)))
        elif re.match(r"(?i)^what\b", q_tmpl):
            # Avoid "Describe what is X" — strip "what is/was/are/were" cleanly
            _desc = re.sub(r"(?i)^what\s+(is|was|are|were)\s+", "What was the reported ", q_tmpl)
            if _desc == q_tmpl:  # no is/was/are — safe to use describe
                _desc = re.sub(r"(?i)^what\s+", "Describe what ", q_tmpl)
            candidates.append(ensure_q(_desc))
        _base = re.sub(r"(?i)^(explain|describe)\s+", "How would you ", q_tmpl)
        if _base != q_tmpl:
            candidates.append(ensure_q(_base))

    return candidates, core_no_time if time_phrase else strip_q(q_tmpl), time_phrase


def is_complex(q: str) -> bool:
    return bool(re.search(
        r"(?i)\b(if|unless|assuming|given that|provided that|"
        r"excluding|except|other than|in addition|also explain|"
        r"please state|state that and explain)\b",
        q
    ))


# FALLBACK FILL — diverse, natural frames
# Avoid no "Provide the If...", no "State the If..."

def _yesno_to_stmt(q: str) -> str:
    """Strip leading auxiliary from a yes/no question to make a statement fragment.
    e.g. 'Does 3M have healthy liquidity' → '3M has healthy liquidity'
    Used so fill templates don't produce 'Is there evidence that does 3M...'
    """
    aux_map = {
        "does": "", "do": "", "did": "",
        "is": "", "are": "", "was": "", "were": "",
        "has": "", "have": "", "had": "",
        "can": "", "could": "", "would": "", "should": "", "will": "",
    }
    m = re.match(r"(?i)^(does|do|did|is|are|was|were|has|have|had|can|could|would|should|will)\s+(.*)", q.strip())
    if not m:
        return q
    # Return the rest with correct casing (capitalize first letter)
    rest = m.group(2).strip()
    return rest[0].upper() + rest[1:] if rest else q


NATURAL_FILL_FRAMES = [
    "What is the value of {core}",
    "How would you describe {core}",
    "What figure was reported for {core}",
    "How much was {core}",
    "What did the company record as {core}",
    "What amount was reported for {core}",
    "How did {core} compare to prior periods",
]


def fill_to_five(candidates: list, core: str, time_phrase: str) -> list:
    """Fill candidate list to 5 using natural frames against the core NP."""
    seen_norms = {normalize_for_dedup(x) for x in candidates}
    tp_suffix  = f" {time_phrase}" if time_phrase else ""

    # Strategy 1: NP-based frames (only if core is not a conditional clause)
    if core and not re.match(r"(?i)^(if|unless|excluding|given|assuming)\b", core):
        # Strip "What is/was/are..." prefix from core before applying frames
        _core_noun = re.sub(r"(?i)^what\s+(is|was|are|were|drove|caused)\s+", "", core.strip())
        _core_noun = re.sub(r"(?i)^(does|is|are|did|has|have|was|were)\s+\S+\s+", "", _core_noun).strip() or core.strip()
        for frame in NATURAL_FILL_FRAMES:
            if len(candidates) >= 5:
                break
            candidate = ensure_q(frame.format(core=_core_noun) + tp_suffix)
            candidate = fix_sva(clean_double_articles(candidate))
            norm = normalize_for_dedup(candidate)
            if norm not in seen_norms:
                candidates.append(candidate)
                seen_norms.add(norm)

    # Strategy 2: For complex/hypothetical questions — light lexical variation on the original
    if len(candidates) < 5 and candidates:
        base = strip_q(core_for_templates(candidates[0]))
        lexical_variants = [
            re.sub(r"\bhow much\b", "what amount", base, flags=re.IGNORECASE),
            re.sub(r"\bwhat\b", "which", base, count=1, flags=re.IGNORECASE),
            re.sub(r"\bcould\b", "would", base, flags=re.IGNORECASE),
            re.sub(r"\bdragged down\b", "weighed on", base, flags=re.IGNORECASE),
            re.sub(r"\bcompare\b", "differ", base, flags=re.IGNORECASE),
            re.sub(r"\beffective\b", "actual", base, flags=re.IGNORECASE),
            re.sub(r"\bgrowth\b", "expansion", base, flags=re.IGNORECASE),
            re.sub(r"\bincreased\b", "grew", base, flags=re.IGNORECASE),
            re.sub(r"\bdecreased\b", "fell", base, flags=re.IGNORECASE),
            f"How would you characterize {base}" if not re.match(r"(?i)^(does|is|are|did|has|have|was|were|can|could|would|should|will)\b", base) else ensure_q(f"Would you say that {_yesno_to_stmt(base)}"),
        ]
        for v in lexical_variants:
            if len(candidates) >= 5:
                break
            v = fix_sva(clean_double_articles(ensure_q(v)))
            norm = normalize_for_dedup(v)
            if norm not in seen_norms:
                candidates.append(v)
                seen_norms.add(norm)

    # Absolute last resort — use structural variants rather than repeating
    if len(candidates) < 5 and candidates:
        base = strip_q(candidates[0])
        extras = [
            ensure_q(f"Is it true that {base}"),
            ensure_q(f"Would you confirm that {base}"),
            ensure_q(f"Can we say that {base}"),
            ensure_q(f"Is it fair to say that {base}"),
        ]
        for ex in extras:
            if len(candidates) >= 5:
                break
            norm = normalize_for_dedup(ex)
            if norm not in seen_norms:
                candidates.append(ex)
                seen_norms.add(norm)
        # true last resort
        while len(candidates) < 5:
            candidates.append(candidates[0])

    return candidates[:5]


# POST-PROCESSING: apply to ALL paraphrases

def post_process(p: str) -> str:
    """Fix every common artifact: double articles, SVA, spacing, punctuation, quarters, tense."""
    p = clean_double_articles(p)
    p = fix_sva(p)
    p = re.sub(r"\s{2,}", " ", p)
    # Normalize quarter/fiscal year capitalization (e.g. fy2022 → FY2022, q2 → Q2)
    p = re.sub(r"\bfy(\d{4})\b", r"FY\1", p, flags=re.IGNORECASE)
    p = re.sub(r"\b(q[1-4])\b", lambda m: m.group(1).upper(), p)
    p = re.sub(r"\b(Q[1-4])\s+of\s+(\d{4})\b", r"\1 \2", p)
    # Fix "how much is" → "how much was" (financials are historical)
    p = re.sub(r"\bhow much is\b", "how much was", p, flags=re.IGNORECASE)
    # Fix "Is it the case that does/is/was/did X" → "Is it the case that X" (strip aux after "that")
    p = re.sub(
        r"\b(the case|it fair|reasonable|evidence)\s+that\s+(does|is|are|was|were|did|has|have|had)\s+",
        lambda m: m.group(0).split(" that ")[0] + " that ",
        p, flags=re.IGNORECASE
    )
    # Fix "How would you characterize does/is/was/did/has X" → "Is it the case that X"
    p = re.sub(
        r"\bHow would you (characterize|describe)\s+(does|is|was|are|were|did|has|have|had|can|could|would)\s+",
        "Is it the case that ",
        p, flags=re.IGNORECASE
    )
    # Fix "How would you characterize if..." → remove (already handled in complex branch)
    p = re.sub(r"\bHow would you characterize\s+if\b", "What happens if", p, flags=re.IGNORECASE)
    # Fix "the the", double articles after these substitutions
    p = re.sub(r"\b(the|a|an)\s+\1\b", r"\1", p, flags=re.IGNORECASE)
    # Fix "the X amount" → "the X" and "X amount" → "X"
    p = re.sub(r"\bthe (\w+) amount\b", r"the \1", p, flags=re.IGNORECASE)
    p = re.sub(r"\b(capital expenditure|capex) amount\b", r"\1", p, flags=re.IGNORECASE)
    # Fix possessive + article: "3M's a reasonably..." → "3M's reasonably..."
    p = re.sub(r"(\w+'s)\s+a\s+(?=[a-z])", r"\1 ", p)
    # Fix casing: 3m → 3M, m&a → M&A
    p = re.sub(r"\b3m\b", "3M", p, flags=re.IGNORECASE)
    p = re.sub(r"\bm&a\b", "M&A", p, flags=re.IGNORECASE)
    # Fix grammar: "did + 3rd-person verb" → "did + base form"
    # e.g. "Did 3M has" → "Did 3M have", "Did 3M maintains" → "Did 3M maintain"
    p = re.sub(r"\b(did|does)\s+(\w+)\s+(has|is|maintains|operates|generates|reports|shows|experiences|holds|carries|uses|makes|includes|contains|represents|reflects|records|achieves|improves|increases|decreases|declines|grows|rises|falls)\b",
               lambda m: m.group(1) + " " + m.group(2) + " " + {
                   "has":"have","is":"be","maintains":"maintain","operates":"operate",
                   "generates":"generate","reports":"report","shows":"show",
                   "experiences":"experience","holds":"hold","carries":"carry",
                   "uses":"use","makes":"make","includes":"include","contains":"contain",
                   "represents":"represent","reflects":"reflect","records":"record",
                   "achieves":"achieve","improves":"improve","increases":"increase",
                   "decreases":"decrease","declines":"decline","grows":"grow",
                   "rises":"rise","falls":"fall"
               }.get(m.group(3).lower(), m.group(3)),
               p, flags=re.IGNORECASE)
    # Fix doubled prepositions e.g. "for For X" → "for X"
    p = re.sub(r"\bfor\s+[Ff]or\b", "for", p)
    # Fix double-possessive e.g. "AMCOR's's" → "AMCOR's"
    p = re.sub(r"('s)'s\b", r"\1", p)
    # Fix wrong possessive split e.g. "CVS's Health" → "CVS Health" (company name split)
    p = re.sub(r"\b([A-Z]+)'s\s+([A-Z][a-z]+)\b", lambda m: m.group(1) + " " + m.group(2)
               if m.group(1)+m.group(2) in ("CVSHealth","MGMResorts","BestBuy") or
               len(m.group(1)) <= 4 and m.group(2)[0].isupper()
               else m.group(0), p)
    p = re.sub(r"\bin\s+[Ii]n\b", "in", p)
    p = re.sub(r"\bas of\s+as of\b", "as of", p, flags=re.IGNORECASE)
    p = re.sub(r"\bdid has\b", "did have", p, flags=re.IGNORECASE)
    p = re.sub(r"\bdid is\b", "was", p, flags=re.IGNORECASE)
    # Fix "Is/Was/Are X as a Y" → "Is/Was/Are X a Y" (wrong as-a after copula)
    p = re.sub(r"\b(Is|Was|Are|Were)(\s+\S+)\s+as\s+a\b", r"\1\2 a", p)
    # Fix "that X as a" → "that X is a"
    p = re.sub(r"(\bthat\s+\S+)\s+as\s+a\b", r"\1 is a", p)
    # Fix "What can be said about if..." → "What can be said about whether..."
    p = re.sub(r"\bWhat can be said about if\b", "What can be said about whether", p, flags=re.IGNORECASE)
    p = ensure_q(p)
    return p


# Assemble five candidates

def validate_and_fill(candidates: list, core: str, time_phrase: str) -> list:
    """Post-process, deduplicate, fill to 5.
    Ensures the 'If X is not useful...' conditional only appears in paraphrase[0].
    """
    cleaned = [post_process(p) for p in candidates if p and len(p.strip()) > 6]
    deduped = deduplicate(cleaned)
    deduped = fill_to_five(deduped, core or "", time_phrase)
    # Final safety: strip any leaked preamble text from ALL paraphrases
    result = []
    for p in deduped[:5]:
        p_clean = clean_question(p)
        if p_clean and len(p_clean) > 6:
            result.append(ensure_q(p_clean))
        else:
            result.append(p)
    return result


YES_NO_RE = re.compile(
    r"^(does|is|are|did|has|have|was|were|can|could|would|should|will)\b",
    re.IGNORECASE
)


def generate_paraphrases(question: str) -> list:
    """
    Full pipeline:
    1. Clean instruction text from original
    2. Route to appropriate handler
    3. Post-process + deduplicate + fill to 5
    """
    if not question or not question.strip():
        return [""] * 5

    # Clean — remove instructions, double articles, normalize spaces
    q_clean = clean_question(question)
    if not q_clean:
        q_clean = question.strip()

    q_clean = q_clean[0].upper() + q_clean[1:]
    q_lower  = q_clean.lower()
    original = ensure_q(q_clean)

    # Route
    if YES_NO_RE.match(q_lower):
        candidates, core, tp = paraphrase_yes_no(q_clean, original)

    elif re.match(r"what (drove|caused|led|contributed|resulted)\b", q_lower):
        candidates, core, tp = paraphrase_what_drove(q_clean, original)

    elif is_complex(q_clean):
        candidates, core, tp = paraphrase_complex(q_clean, original)

    elif re.match(r"which\b", q_lower):
        candidates, core, tp = paraphrase_which(q_clean, original)

    elif re.match(r"what (is|was|are|were)\b", q_lower):
        candidates, core, tp = paraphrase_what_is(q_clean, original)

    elif re.match(r"what\b", q_lower):
        candidates, core, tp = paraphrase_what_generic(q_clean, original)

    elif re.match(r"how much\b|how many\b", q_lower):
        candidates, core, tp = paraphrase_how_much(q_clean, original)

    elif re.match(r"how (did|has|does|do|is|are|was|were)\b", q_lower):
        candidates, core, tp = paraphrase_how_did(q_clean, original)

    elif re.match(r"why\b", q_lower):
        candidates, core, tp = paraphrase_why(q_clean, original)

    else:
        candidates, core, tp = paraphrase_complex(q_clean, original)

    # Post-process + fill
    return validate_and_fill(candidates, core, tp)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate template-based FinanceBench paraphrases.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSON array of FinanceBench records.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    args = parser.parse_args()
    try:
        with args.input.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot read input JSON: {exc}")
    if not isinstance(data, list) or not data or any(
        not isinstance(item, dict) or not isinstance(item.get("question"), str)
        or not item["question"].strip() for item in data
    ):
        parser.error("Input must be a nonempty JSON array of records with nonempty question strings.")
    if args.input is not None and args.input.resolve() == args.output.resolve():
        parser.error("Input and output must use different paths.")
    if not args.output.parent.is_dir():
        parser.error(f"Output directory does not exist: {args.output.parent}")
    try:
        from tqdm import tqdm
    except ImportError:
        parser.error("Install tqdm to generate paraphrases.")

    results = []
    failure_log = []

    for item in tqdm(data):
        if not isinstance(item, dict):
            continue
        question = item.get("question", "")
        if not question:
            continue

        try:
            paraphrases = generate_paraphrases(question)
        except Exception as e:
            failure_log.append({"question": question, "error": str(e)})
            paraphrases = [ensure_q(clean_question(question))] * 5

        # Final guard
        if len(paraphrases) != 5 or any(not p.strip() for p in paraphrases):
            core_fb = strip_q(clean_question(question))
            paraphrases = validate_and_fill(
                [ensure_q(core_fb)], core_fb, None
            )

        new_item = dict(item)
        new_item["paraphrases"] = paraphrases
        results.append(new_item)


    print("\n-- Self-validation --")
    issues = 0
    issue_details = {
        "wrong_count": 0,
        "near_duplicate": 0,
        "missing_q": 0,
        "double_the": 0,
        "instruction_leak": 0,
        "broken_sentence": 0,
        "bad_fronting": 0,
    }

    for r in results:
        ps = r.get("paraphrases", [])

        if len(ps) != 5:
            print(f"  Wrong count ({len(ps)}): {r['question'][:60]}")
            issue_details["wrong_count"] += 1
            issues += 1
            continue

        norms = [normalize_for_dedup(p) for p in ps]
        if len(set(norms)) < 5:
            print(f"   Near-duplicate: {r['question'][:60]}")
            issue_details["near_duplicate"] += 1
            issues += 1

        for p in ps:
            if not p.endswith("?"):
                print(f"  Missing '?': {p[:60]}")
                issue_details["missing_q"] += 1
                issues += 1

            if "the the" in p.lower():
                print(f"  Double 'the': {p[:80]}")
                issue_details["double_the"] += 1
                issues += 1

            # Preserve valid conditional instructions.
            _leak_check = p
            _conditional_re = re.compile(r"if\s+\w[\w\s]+?not\s+(?:a\s+)?useful.+?(?:then\s+)?please\s+state\s+that\s+and\s+explain", re.IGNORECASE)
            _leak_check_safe = _conditional_re.sub("__CONDITIONAL__", _leak_check)
            if re.search(r"(?i)(please state that and explain|give a response to|answer in usd|answer in eur|using only the|basing your|according to the information provided)", _leak_check_safe):
                print(f"  Instruction leaked: {p[:80]}")
                issue_details["instruction_leak"] += 1
                issues += 1

            if "then?" in p or re.search(r",\s*then\?$", p):
                print(f"  Broken sentence: {p[:80]}")
                issue_details["broken_sentence"] += 1
                issues += 1

            if re.match(r"^For in\b", p) or re.match(r"^Provide the If\b", p):
                print(f"  Bad fronting: {p[:80]}")
                issue_details["bad_fronting"] += 1
                issues += 1

    if issues == 0:
        print("  All items passed validation")
    else:
        details = ", ".join(f"{key}: {value}" for key, value in issue_details.items() if value)
        print(f"Review warnings ({issues} issues; {details})")
    if failure_log:
        print(f"Review warnings: fallback candidates used for {len(failure_log)} records.")
    try:
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except OSError as exc:
        parser.error(f"Cannot write output JSON: {exc}")
    print(f"Output written: {args.output} ({len(results)} records)")


if __name__ == "__main__":
    main()
