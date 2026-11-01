"""Generate numeric and multi-year condition candidates from FinanceBench records."""

import json
import re
import math
from pathlib import Path
from collections import OrderedDict


# Math helpers

def parse_numeric(s):
    """Return float from raw answer string, or None if non-numeric."""
    s = re.sub(r'[\$,]', '', str(s).strip()).rstrip('%').strip()
    try:
        return float(s)
    except ValueError:
        return None


def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0: return False
    return True


def is_fibonacci(n):
    """Perfect-square O(1) Fibonacci test."""
    if n < 0: return False
    def perf_sq(x):
        s = int(math.isqrt(x))
        return s * s == x
    return perf_sq(5 * n * n + 4) or perf_sq(5 * n * n - 4)


def nearest_prime(n):
    """Return the closest prime to n (ties go to the larger)."""
    if n < 2: return 2
    if is_prime(n): return n
    lo, hi = n - 1, n + 1
    while True:
        clo = lo if (lo >= 2 and is_prime(lo)) else None
        chi = hi if is_prime(hi) else None
        if clo is not None and chi is not None:
            return clo if (n - clo) < (chi - n) else chi
        if clo is not None: return clo
        if chi is not None: return chi
        lo -= 1; hi += 1


def nearest_fibonacci(n):
    """Return the closest Fibonacci number to n (ties go to the larger)."""
    if n <= 0: return 0
    a, b = 0, 1
    while b < n:
        a, b = b, a + b
    if (b - n) < (n - a): return b
    elif (n - a) < (b - n): return a
    else: return b  # tie → larger


PRIME_DIGITS = {2, 3, 5, 7}
EVEN_DIGITS  = {0, 2, 4, 6, 8}
ODD_DIGITS   = {1, 3, 5, 7, 9}


def digits_of_integer(val):
    """Digits of the INTEGER part only — 1577.00 → [1,5,7,7], 8.70 → [9]."""
    return [int(c) for c in str(abs(round(val)))]


def fmt(v):
    """Clean number formatting — no trailing .0 noise."""
    if v == int(v): return str(int(v))
    return f"{v:.4f}".rstrip('0').rstrip('.')


def ensure_q(s):
    """Exactly one '?' at end, trimmed."""
    return s.strip().rstrip('?').strip() + '?'


def clean_spaces(s):
    return re.sub(r' {2,}', ' ', s)


# Preamble / trailing-instruction stripper

PREAMBLE_PAT = re.compile(
    r'^(?:'
    r'Assume\s+that\s+you\s+are.+?(?::\s*|\.\s*)'
    r'|You\s+are\s+an?\s+.+?:\s*'
    r'|Using\s+(?:the\s+|only\s+).+?:\s*'
    r'|Basing\s+your.+?:\s*'
    r'|We\s+(?:need|want)\s+to\s+.+?:\s*'
    r'|Answer\s+the\s+following.+?:\s*'
    r"|Here's\s+the\s+question:\s*"
    r'|Please\s+(?:answer|help).+?:\s*'
    r'|Give\s+a\s+response.+?:\s*'
    r'|By\s+drawing\s+conclusions.+?:\s*'
    r')+'
    , re.IGNORECASE | re.DOTALL
)

TRAILING_PAT = re.compile(
    r'\s*(?:'
    r'Give\s+a\s+response.+?'
    r'|Relying\s+on.+?'
    r'|Using\s+the\s+.+?statement.+?'
    r'|Based\s+on\s+the\s+.+?statement.+?'
    r'|Answer\s+in\s+USD\s+\w+'
    r'|Compute\s+or\s+estimate.+?'
    r'|Address\s+the\s+question.+?'
    r'|Base\s+your\s+.+?'
    r')\.?$'
    , re.IGNORECASE | re.DOTALL
)


def clean_base_question(q):
    """Strip preamble + trailing instructions → core financial question, no '?'."""
    q = PREAMBLE_PAT.sub('', q.strip())
    q = TRAILING_PAT.sub('', q.strip())
    q = q.strip().rstrip('?').strip()
    q = clean_spaces(q)
    if q and q[0].islower():
        q = q[0].upper() + q[1:]
    return q


def metric_phrase(base_q):
    """
    Convert clean base question to noun phrase for mid-sentence embedding.
    "What is the FY2018 capital expenditure (in USD millions) for 3M"
    → "FY2018 capital expenditure (in USD millions) for 3M"
    """
    if not base_q or not base_q.strip():
        return "answer"
    phrase = re.sub(
        r'^(?:what\s+(?:is|was|were)\s+(?:the\s+)?)',
        '', base_q.strip(), flags=re.IGNORECASE
    ).strip().rstrip('?').strip()
    return phrase if phrase else base_q.rstrip('?').strip()


# Parts A & B — numeric property variants
#   Yes/No questions  → gold answer = "Yes" or "No" ONLY
#                        question must end with "Answer only Yes or No."
#                        full reasoning stored in "explanation" field
#   Open questions    → gold answer = the actual value/count
#                        full reasoning stays in answer

def make_numeric_variants(item):
    """Generate Parts A+B numeric-constraint questions. Returns [] if non-numeric."""
    ans_raw = str(item.get('answer', '')).strip()
    val     = parse_numeric(ans_raw)
    if val is None:
        return []

    base_q = clean_base_question(item['question'])
    mp     = metric_phrase(base_q)

    n    = round(val)
    digs = digits_of_integer(val)

    # Rounding explanation used in the "explanation" field
    rounding_explanation = f"The answer is {ans_raw}, which rounds to the closest integer {n}."

    prime_ok  = is_prime(n)
    fib_ok    = is_fibonacci(n)
    is_even   = (n % 2 == 0)
    near_p    = nearest_prime(n)
    near_fib  = nearest_fibonacci(n)
    near_even = n if is_even else (n - 1 if n > 0 else n + 1)
    near_odd  = n if not is_even else n + 1

    variants = []

    # A-1: Prime check  → YES/NO
    variants.append({
        "question": (
            f"Is the {mp} a prime number (rounded to the closest integer value)? "
            f"Answer only Yes or No."
        ),
        "answer":      "Yes" if prime_ok else "No",
        "explanation": (f"{rounding_explanation} "
                        f"{n} {'is' if prime_ok else 'is not'} a prime number."),
        "constraint_type": "numeric_prime_check",
    })

    # A-2: Fibonacci check  → YES/NO
    variants.append({
        "question": (
            f"Is the {mp} a Fibonacci number (rounded to the closest integer value)? "
            f"Answer only Yes or No."
        ),
        "answer":      "Yes" if fib_ok else "No",
        "explanation": (f"{rounding_explanation} "
                        f"{n} {'is' if fib_ok else 'is not'} a Fibonacci number."),
        "constraint_type": "numeric_fibonacci_check",
    })

    # A-3: Even/Odd  → open answer ("even" or "odd")
    # NOT a yes/no question — it's a choice question
    variants.append({
        "question": ensure_q(
            f"Is the {mp} even or odd (rounded to the closest integer value)"),
        "answer":      "even" if is_even else "odd",
        "explanation": f"{rounding_explanation} {n} is {'even' if is_even else 'odd'}.",
        "constraint_type": "numeric_even_odd_check",
    })

    # A-4 to A-7: Divisibility  → YES/NO
    for d in [3, 4, 5, 10]:
        div_ok = (n % d == 0)
        variants.append({
            "question": (
                f"Is the {mp} divisible by {d} (rounded to the closest integer value)? "
                f"Answer only Yes or No."
            ),
            "answer":      "Yes" if div_ok else "No",
            "explanation": (f"{rounding_explanation} "
                            f"{n} {'is' if div_ok else 'is not'} divisible by {d}."),
            "constraint_type": f"numeric_divisible_by_{d}",
        })

    # A-8: Closest prime  → open answer (the number)
    variants.append({
        "question": ensure_q(
            f"What is the closest prime number to the {mp} "
            f"(rounded to the closest integer value)"),
        "answer": str(near_p),
        "explanation": (f"{rounding_explanation} "
                        f"The closest prime to {n} is {near_p}"
                        f"{' (it is itself prime)' if prime_ok else ''}."),
        "constraint_type": "numeric_closest_prime",
    })

    # A-9: Closest Fibonacci  → open answer (the number)
    variants.append({
        "question": ensure_q(
            f"What is the closest Fibonacci number to the {mp} "
            f"(rounded to the closest integer value)"),
        "answer": str(near_fib),
        "explanation": (f"{rounding_explanation} "
                        f"The closest Fibonacci number to {n} is {near_fib}"
                        f"{' (it is itself a Fibonacci number)' if fib_ok else ''}."),
        "constraint_type": "numeric_closest_fibonacci",
    })

    # A-10: Closest even  → open answer (the number)
    variants.append({
        "question": ensure_q(
            f"What is the closest even number to the {mp} "
            f"(rounded to the closest integer value)"),
        "answer": str(near_even),
        "explanation": (f"{rounding_explanation} "
                        f"The closest even number to {n} is {near_even}"
                        f"{' (it is itself even)' if is_even else ''}."),
        "constraint_type": "numeric_closest_even",
    })

    # A-11: Closest odd  → open answer (the number)
    variants.append({
        "question": ensure_q(
            f"What is the closest odd number to the {mp} "
            f"(rounded to the closest integer value)"),
        "answer": str(near_odd),
        "explanation": (f"{rounding_explanation} "
                        f"The closest odd number to {n} is {near_odd}"
                        f"{' (it is itself odd)' if not is_even else ''}."),
        "constraint_type": "numeric_closest_odd",
    })

    # Part B: Digit checks  → open answers (counts / Yes/No)
    pdigs      = [d for d in digs if d in PRIME_DIGITS]
    edigs      = [d for d in digs if d in EVEN_DIGITS]
    odigs      = [d for d in digs if d in ODD_DIGITS]
    only_prime = bool(digs) and all(d in PRIME_DIGITS for d in digs)
    digit_explanation = (f"The answer rounds to the closest integer {n}. "
                         f"Its digits are {digs}.")

    # B-1: Prime digit count  → open answer (integer count)
    variants.append({
        "question": ensure_q(
            f"How many prime digits (2, 3, 5, 7) does the {mp} contain "
            f"(rounded to the closest integer value)"),
        "answer":      str(len(pdigs)),
        "explanation": f"{digit_explanation} Prime digits: {pdigs}. Count = {len(pdigs)}.",
        "constraint_type": "numeric_prime_digit_count",
    })

    # B-2: Only prime digits  → YES/NO
    variants.append({
        "question": (
            f"Does the {mp} contain only prime digits (2, 3, 5, 7) "
            f"(rounded to the closest integer value)? "
            f"Answer only Yes or No."
        ),
        "answer":      "Yes" if only_prime else "No",
        "explanation": (f"{digit_explanation} "
                        f"{'All digits are prime.' if only_prime else 'Not all digits are prime.'}"),
        "constraint_type": "numeric_only_prime_digits",
    })

    # B-3: Even digit count  → open answer (integer count)
    variants.append({
        "question": ensure_q(
            f"How many even digits does the {mp} contain "
            f"(rounded to the closest integer value)"),
        "answer":      str(len(edigs)),
        "explanation": f"{digit_explanation} Even digits: {edigs}. Count = {len(edigs)}.",
        "constraint_type": "numeric_even_digit_count",
    })

    # B-4: Odd digit count  → open answer (integer count)
    variants.append({
        "question": ensure_q(
            f"How many odd digits does the {mp} contain "
            f"(rounded to the closest integer value)"),
        "answer":      str(len(odigs)),
        "explanation": f"{digit_explanation} Odd digits: {odigs}. Count = {len(odigs)}.",
        "constraint_type": "numeric_odd_digit_count",
    })

    return variants


# Part C — evidence text processing utilities

def get_evidence_text(item):
    ev = item.get('evidence', [])
    if isinstance(ev, list):
        return ' '.join(e.get('evidence_text', '') for e in ev)
    return str(ev)


def detect_column_order(text, candidate_years):
    """Detect column order of years in the evidence table header."""
    year_positions = {}
    for yr in candidate_years:
        for m in re.finditer(r'(?<!\d)' + yr + r'(?!\d)', text):
            year_positions[yr] = m.start()
            break
    if not year_positions:
        return sorted(candidate_years, reverse=True)
    return sorted(year_positions.keys(), key=lambda y: year_positions[y])


def find_value_variants(val):
    """Generate all plausible string representations of a financial value."""
    v = abs(val)
    candidates = []
    if v == int(v):
        n = int(v)
        candidates.append(f"{n:,}")
        candidates.append(str(n))
    else:
        candidates.append(f"{v:.2f}")
        candidates.append(f"{v:.1f}")
        int_part = int(v)
        dec_part = f"{v:.2f}".split('.')[1]
        if int_part >= 1000:
            candidates.append(f"{int_part:,}.{dec_part}")
    return candidates


def extract_row_numbers(window_text):
    """Extract numeric values from a table-row window, skipping year-like numbers."""
    vals = []
    for raw in re.findall(r'\([\d,]+\.?\d*\)|[\-]?[\d,]+\.?\d*', window_text):
        clean = raw.strip().replace(',', '')
        if clean.startswith('(') and clean.endswith(')'):
            clean = '-' + clean[1:-1]
        try:
            v = float(clean)
            if 1800 <= abs(v) <= 2100:
                continue
            vals.append(v)
        except ValueError:
            pass
    return vals


def extract_multiyear_values_proximity(text, ref_val, col_order):
    """
    Column-order-aware proximity extractor for Part C.
    Finds ref_val in the evidence text, reads row numbers,
    maps them to years by column order. Returns {year: float} or {}.
    """
    n_cols = len(col_order)
    if n_cols == 0:
        return {}
    search_val = abs(ref_val)
    for variant in find_value_variants(search_val):
        pattern = r'[\(\-]?' + re.escape(variant) + r'\)?'
        for m in re.finditer(pattern, text):
            window = text[m.start(): min(len(text), m.start() + 600)]
            vals   = extract_row_numbers(window)
            if len(vals) < n_cols:
                continue
            first   = vals[0]
            ref_abs = abs(ref_val)
            if ref_abs > 0.001:
                if abs(abs(first) - ref_abs) / ref_abs > 0.02:
                    continue
            elif abs(first) > 0.01:
                continue
            if ref_val < 0 and first > 0: first = -first
            if ref_val > 0 and first < 0: continue
            sign   = -1 if ref_val < 0 else 1
            result = {}
            for i, yr in enumerate(col_order):
                if i < len(vals):
                    result[yr] = sign * abs(vals[i]) if sign < 0 else vals[i]
            return result
    return {}


def sanity_check_values(yvm, ref_val):
    """Remove outlier years implausibly far from ref_val in log scale."""
    if not ref_val or ref_val == 0:
        return yvm
    log_ref = math.log(abs(ref_val) + 1)
    return {y: v for y, v in yvm.items()
            if abs(math.log(abs(v) + 1) - log_ref) < 2.3}


# Hard-coded verified year→value maps for computed-ratio items

HARDCODED = {
    "financebench_id_03856": {"2016": 0.52, "2017": 0.66, "2018": 0.83},
    "financebench_id_09724": {"2019": 39.2, "2020": 40.7, "2021": 39.7},
    "financebench_id_06272": {"2020": 0.91, "2021": 0.74, "2022": 0.80},
    "financebench_id_05915": {"2017": 17.95, "2018": 17.15},
    "financebench_id_04980": {"2019": 4.23, "2020": 4.24, "2021": 4.62},
    "financebench_id_03620": {"2020": 8388, "2021": 9247, "2022": 9068},
    "financebench_id_04481": {"2020": 17.9, "2021": 17.5, "2022": 16.5},
    "financebench_id_06247": {"2017": 41.86, "2018": 45.07},
    "financebench_id_06655": {"2015": 74.31, "2016": 85.33, "2017": 93.86},
    "financebench_id_07966": {"2017": 1.8, "2018": 1.6, "2019": 2.3},
    "financebench_id_02987": {"2017": 26.88, "2018": 22.74, "2019": 24.26},
    "financebench_id_07507": {"2014": 48.2, "2015": 57.9, "2016": 65.4},
    "financebench_id_03473": {"2015": 0.08, "2016": 0.07, "2017": 0.01},
    "financebench_id_04103": {"2017": -5.2, "2018": -4.1, "2019": -3.7},
    "financebench_id_10499": {"2017": 5.92, "2018": 5.88, "2019": 6.25},
}

SKIP_C = {fid for fid, yv in HARDCODED.items() if len(yv) < 2}

_DIMENSIONLESS_PAT = re.compile(
    r'\b(?:ratio|margin|turnover|percentage|payout|yield|return|roe|roa|roi|'
    r'ccc|dpo|dso|dio|growth|change|rate|multiple|coverage|days|'
    r'capex\s+as\s+a\s+%|%\s+of\s+revenue)\b',
    re.IGNORECASE
)


def infer_unit(metric, answer_raw, explicit_unit):
    if _DIMENSIONLESS_PAT.search(metric): return ""
    if re.search(r'%', str(answer_raw)):  return ""
    return explicit_unit


# Part C — multi-year comparison variant generator

def extract_metric_and_company(qs):
    """Pull metric noun phrase and company name from a cleaned question string."""
    metric  = None
    company = None
    unit    = "USD millions"
    for pat in [
        r'(?i)what\s+(?:is|was)\s+(?:the\s+)?(?:FY\d{2,4}|Q\d\s*\d{4})[\s,]*(.+?)\s+for\s+(\w[\w\s&\'.\-]{1,30}?)(?:\?|$)',
        r"(?i)what\s+(?:is|was)\s+[\w\s&'.]+?'s\s+(?:FY\d{2,4})[\s,]*(.+?)(?:\?|$)",
        r'(?i)what\s+(?:is|was)\s+(?:the\s+)?(.+?)\s+for\s+(\w[\w\s&\'.\-]{1,30}?)(?:\?|$)',
    ]:
        m = re.match(pat, qs)
        if m:
            g1 = m.group(1).strip()
            g2 = m.group(2).strip() if len(m.groups()) >= 2 else None
            metric  = re.sub(r'\bFY\d{2,4}\b', '', g1, flags=re.IGNORECASE).strip().rstrip(',').strip()
            company = g2
            break
    if not metric:
        return None, unit, None
    unit_m = re.search(r'\(([^)]+)\)', metric)
    if unit_m:
        unit   = unit_m.group(1)
        metric = re.sub(r'\s*\([^)]+\)', '', metric).strip()
    metric = re.sub(r'\s{2,}', ' ', metric).rstrip(',').strip()
    if not company:
        m_c = re.search(r'\bfor\s+([A-Z][A-Za-z0-9&\'.\- ]{1,30}?)(?:\?|$|\s+in\s+|\s+as\s+|\.)', qs)
        if m_c: company = m_c.group(1).strip()
    if not company:
        return None, unit, None
    return metric, unit, company


def make_multiyear_variants(item):
    """Generate Part C multi-year comparison questions."""
    fid      = item['financebench_id']
    question = item['question']
    answer   = str(item.get('answer', '')).strip()
    val      = parse_numeric(answer)

    if fid in SKIP_C:   return []
    if val is None:     return []
    if not re.search(r'(?i)\bFY\d{2,4}\b', question): return []
    if val == 0 and fid not in HARDCODED: return []

    if fid in HARDCODED:
        yvm = dict(HARDCODED[fid])
    else:
        ev_text         = get_evidence_text(item)
        candidate_years = sorted(
            set(y for y in re.findall(r'\b(20\d{2})\b', ev_text) if 2010 <= int(y) <= 2024)
        )
        if len(candidate_years) < 2: return []
        col_order = detect_column_order(ev_text, candidate_years)
        yvm       = extract_multiyear_values_proximity(ev_text, val, col_order)
        orig_yr_m = re.search(r'\bFY(\d{4})\b', question, re.IGNORECASE)
        if not orig_yr_m: return []
        orig_year = orig_yr_m.group(1)
        if orig_year not in yvm:
            yvm[orig_year] = val
        yvm = sanity_check_values(yvm, val)

    if len(yvm) < 2: return []

    qs = clean_base_question(question)
    metric, explicit_unit, company = extract_metric_and_company(qs)
    if not metric or not company: return []

    unit        = infer_unit(metric, answer, explicit_unit)
    unit_clause = f" ({unit})" if unit else ""

    yvm         = OrderedDict(sorted(yvm.items()))
    valid_years = list(yvm.keys())
    ystr        = ", ".join(f"FY{y}" for y in valid_years)
    asc         = sorted(yvm.items(), key=lambda x: x[1])
    desc        = sorted(yvm.items(), key=lambda x: x[1], reverse=True)
    maxY, maxV  = desc[0]
    minY, minV  = asc[0]
    avg         = round(sum(yvm.values()) / len(yvm), 2)
    rng         = maxV - minV
    total       = sum(yvm.values())
    all_str     = ", ".join(f"FY{y}: {fmt(v)}" for y, v in yvm.items())
    mc          = metric

    vals_chron  = [yvm[y] for y in valid_years]
    diffs       = [vals_chron[i+1] - vals_chron[i] for i in range(len(vals_chron)-1)]
    n_up        = sum(1 for d in diffs if d > 0)
    n_down      = sum(1 for d in diffs if d < 0)
    if n_up == len(diffs):   trend_word = "consistently increased"
    elif n_down == len(diffs): trend_word = "consistently decreased"
    else:                    trend_word = "fluctuated (no single direction)"
    net_chg = vals_chron[-1] - vals_chron[0]
    net_pct = (f" ({fmt(abs(net_chg / vals_chron[0] * 100))}%)" if vals_chron[0] != 0 else "")

    variants = [
        {
            "question": ensure_q(f"List the {mc}{unit_clause} for {company} across {ystr}, sorted in ascending order"),
            "answer":   "Ascending: " + ", ".join(f"FY{y}: {fmt(v)}" for y, v in asc),
            "constraint_type": "multiyear_sorted_ascending",
            "years_used": valid_years, "year_values": dict(yvm),
        },
        {
            "question": ensure_q(f"List the {mc}{unit_clause} for {company} across {ystr}, sorted in descending order"),
            "answer":   "Descending: " + ", ".join(f"FY{y}: {fmt(v)}" for y, v in desc),
            "constraint_type": "multiyear_sorted_descending",
            "years_used": valid_years, "year_values": dict(yvm),
        },
        {
            "question": ensure_q(f"In which year was the {mc}{unit_clause} highest for {company}, based on data from {ystr}"),
            "answer":   f"FY{maxY} had the highest {mc} at {fmt(maxV)}{unit_clause}. All values: {all_str}.",
            "constraint_type": "multiyear_highest_year",
            "years_used": valid_years, "year_values": dict(yvm),
        },
        {
            "question": ensure_q(f"In which year was the {mc}{unit_clause} lowest for {company}, based on data from {ystr}"),
            "answer":   f"FY{minY} had the lowest {mc} at {fmt(minV)}{unit_clause}. All values: {all_str}.",
            "constraint_type": "multiyear_lowest_year",
            "years_used": valid_years, "year_values": dict(yvm),
        },
        {
            "question": ensure_q(f"What is the average {mc}{unit_clause} for {company} across {ystr}"),
            "answer":   f"Average = {fmt(avg)}{unit_clause}. Values: {all_str}. Sum = {fmt(total)}, N = {len(yvm)}.",
            "constraint_type": "multiyear_average",
            "years_used": valid_years, "year_values": dict(yvm),
        },
        {
            "question": ensure_q(f"What is the range (maximum minus minimum) of the {mc}{unit_clause} for {company} across {ystr}"),
            "answer":   f"Range = {fmt(rng)}{unit_clause}. Max: FY{maxY} = {fmt(maxV)}, Min: FY{minY} = {fmt(minV)}.",
            "constraint_type": "multiyear_range",
            "years_used": valid_years, "year_values": dict(yvm),
        },
        {
            "question": ensure_q(f"Rank the {mc}{unit_clause} for {company} across {ystr} from highest to lowest"),
            "answer":   "  |  ".join(f"#{r} FY{y}: {fmt(v)}" for r, (y, v) in enumerate(desc, 1)),
            "constraint_type": "multiyear_rank",
            "years_used": valid_years, "year_values": dict(yvm),
        },
        {
            "question": ensure_q(f"Did the {mc}{unit_clause} for {company} increase, decrease, or fluctuate from {valid_years[0]} to {valid_years[-1]}"),
            "answer":   (f"The {mc} {trend_word} from FY{valid_years[0]} to FY{valid_years[-1]}. "
                         f"Net change: {'+' if net_chg >= 0 else ''}{fmt(net_chg)}{net_pct}. "
                         f"Values: {all_str}."),
            "constraint_type": "multiyear_trend",
            "years_used": valid_years, "year_values": dict(yvm),
        },
        {
            "question": ensure_q(f"What is the total (sum) of {mc}{unit_clause} for {company} across {ystr}"),
            "answer":   f"Total = {fmt(total)}{unit_clause}. Values: {all_str}.",
            "constraint_type": "multiyear_cumulative_sum",
            "years_used": valid_years, "year_values": dict(yvm),
        },
    ]

    for ya, yb in zip(valid_years[:-1], valid_years[1:]):
        va, vb    = yvm[ya], yvm[yb]
        chg       = vb - va
        pct       = (f" ({fmt(abs(chg / va * 100))}%)" if va != 0 else "")
        direction = 'increased' if chg > 0 else ('decreased' if chg < 0 else 'unchanged')
        variants.append({
            "question": ensure_q(f"What was the year-over-year change in {mc}{unit_clause} for {company} from FY{ya} to FY{yb}"),
            "answer":   (f"The {mc} {direction} by {fmt(abs(chg))}{pct} "
                         f"from FY{ya} ({fmt(va)}) to FY{yb} ({fmt(vb)})."),
            "constraint_type": f"multiyear_yoy_{ya}_{yb}",
            "years_used": [ya, yb], "year_values": {ya: va, yb: vb},
        })

    return variants


# Yes/No constraint types (for validation)

YESNO_TYPES = {
    "numeric_prime_check", "numeric_fibonacci_check",
    "numeric_divisible_by_3", "numeric_divisible_by_4",
    "numeric_divisible_by_5", "numeric_divisible_by_10",
    "numeric_only_prime_digits",
}


# Validation

def validate_all(augmented):
    errors = 0
    for item in augmented:
        for v in item.get('constraint_variants', []):
            q  = v.get('question', '')
            a  = v.get('answer',   '')
            ct = v.get('constraint_type', '')
            errs = []
            if not q: errs.append("empty question")
            if not a: errs.append("empty answer")
            if '  ' in q: errs.append("double space in question")
            if re.search(r'answer to the following question', q, re.IGNORECASE):
                errs.append("old wrapper present")
            if re.search(r'Assume that you are', q, re.IGNORECASE):
                errs.append("preamble in question")

            # Yes/No type checks
            if ct in YESNO_TYPES:
                if a not in ("Yes", "No"):
                    errs.append(f"Yes/No answer must be 'Yes' or 'No', got: {a!r}")
                if "Answer only Yes or No" not in q:
                    errs.append("Yes/No question missing 'Answer only Yes or No.'")
                if not q.endswith("."):
                    # ends with "Answer only Yes or No." so ends with '.'
                    pass

            # Open-answer types should NOT be Yes/No
            if ct not in YESNO_TYPES and ct != 'numeric_even_odd_check':
                if a in ("Yes", "No"):
                    errs.append(f"Open-answer type has bare Yes/No answer: {ct}")

            if errs:
                errors += len(errs)
                print(f"  VALIDATION [{ct}] {errs}")
                print(f"    Q: {q[:80]}")
                print(f"    A: {a}")
    return errors


# Main

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate FinanceBench condition candidates.")
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
        or not item["question"].strip() or not item.get("financebench_id") for item in data
    ):
        parser.error("Input must be a nonempty array of records with financebench_id and question fields.")
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must use different paths.")
    if not args.output.parent.is_dir():
        parser.error(f"Output directory does not exist: {args.output.parent}")

    augmented = []
    stats     = dict(ab=0, c=0, iab=0, ic=0)

    for item in data:
        variants       = []
        ab_variants    = make_numeric_variants(item)
        c_variants     = make_multiyear_variants(item)
        if ab_variants:
            variants.extend(ab_variants)
            stats['iab'] += 1
            stats['ab']  += len(ab_variants)
        if c_variants:
            variants.extend(c_variants)
            stats['ic'] += 1
            stats['c']  += len(c_variants)
        aug = dict(item)
        aug['constraint_variants'] = variants
        augmented.append(aug)

    print("Running validation...")
    errs   = validate_all(augmented)
    status = "Validation passed - zero errors." if errs == 0 else f"{errs} issue(s) found."
    print(status, "\n")

    try:
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(augmented, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        parser.error(f"Cannot write output JSON: {exc}")
    print(f"Output written: {args.output}")

    # Count Yes/No vs open-answer variants
    yn_count   = sum(1 for item in augmented
                     for v in item.get('constraint_variants', [])
                     if v['constraint_type'] in YESNO_TYPES)
    open_count = sum(1 for item in augmented
                     for v in item.get('constraint_variants', [])
                     if v['constraint_type'] not in YESNO_TYPES)

    print("  SUMMARY")
    print(f"  Total items processed       : {len(augmented)}")
    print(f"  Items with numeric variants     : {stats['iab']}")
    print(f"  Total numeric variants          : {stats['ab']}")
    print(f"  Items with multi-year variants       : {stats['ic']}")
    print(f"  Total multi-year variants            : {stats['c']}")
    print(f"  Grand total variants        : {stats['ab'] + stats['c']}")
    print(f"  Yes/No answer variants      : {yn_count}")
    print(f"  Open answer variants        : {open_count}")


if __name__ == '__main__':
    main()
