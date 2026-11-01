"""Create synthetic FinanceBench records from explicit company and year mappings."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


FISCAL_YEAR = re.compile(r"(?i)(?<!\w)(FY\s*)(\d{4}|\d{2})(?!\w|[.,]\d)")
YEAR_CONTEXT = re.compile(
    r"(?i)(?:\b(?:in|during|since|as\s+of)\s+"
    r"|\b(?:fiscal\s+)?years?(?:\s+ended)?\s*[:,-]?\s*"
    r"|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s*)$"
)
YEAR_SUFFIX = re.compile(r"(?i)^\s+(?:(?:first|second|third|fourth)\s+quarter|fiscal\s+year)\b")
MONEY_PREFIX = re.compile(r"(?:[$\u00a3\u20ac]|\b(?:USD|EUR|GBP))(?:\s*[-+(]){0,2}\s*$", re.I)
MONEY_SUFFIX = re.compile(r"(?i)^\s*\)?\s*(?:%|\b(?:million|billion|thousand|dollars?|pounds?|euros?|shares?|units?|USD|EUR|GBP)\b)")
DERIVED_FIELDS = {"paraphrases", "constraint_variants", "mask_map", "masked_question"}
SOURCE_FIELDS = ("company", "doc_name", "doc_period", "doc_link", "gics_sector", "doc_type")


def _years(mapping):
    if not isinstance(mapping, dict):
        raise ValueError("Year mappings must be a dictionary.")
    years = {}
    for source, target in mapping.items():
        source, target = int(source), int(target)
        if source in years and years[source] != target:
            raise ValueError(f"Conflicting mappings for {source}.")
        years[source] = target
    if any(not 1000 <= year <= 9999 for pair in years.items() for year in pair):
        raise ValueError("Year mappings must use four-digit years.")
    return {source: target for source, target in years.items() if source != target}


def _mapped_year(value, years):
    if len(value) == 4:
        return years.get(int(value))
    matches = [target for source, target in years.items() if source % 100 == int(value)]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous short fiscal year: FY{value}")
    return matches[0] if matches else None


def _financial_number(text, start, end):
    before, after = text[:start], text[end:]
    if before and before[-1] in ".," and len(before) > 1 and before[-2].isdigit():
        return True
    if after and after[0] in ".," and len(after) > 1 and after[1].isdigit():
        return True
    return bool(MONEY_PREFIX.search(before[-20:]) or MONEY_SUFFIX.match(after[:25]))


def _whole_token(text, start, end):
    if start and re.match(r"\w", text[start - 1]):
        return False
    if end < len(text) and re.match(r"\w", text[end]):
        return False
    before, after = text[:start], text[end:]
    return not (
        len(before) > 1 and before[-1] in ".," and before[-2].isdigit()
        or len(after) > 1 and after[0] in ".," and after[1].isdigit()
    )


def _replace_text(text, company_pattern, target_company, years, path, explicit=None):
    edits = [
        (m.start(), m.end(), target_company)
        for m in company_pattern.finditer(text)
        if not _financial_number(text, m.start(), m.end())
    ]

    def overlaps(start, end):
        return any(start < stop and end > begin for begin, stop, _ in edits)

    recognized = []
    for match in FISCAL_YEAR.finditer(text):
        target = _mapped_year(match.group(2), years)
        if target is not None and not overlaps(*match.span()):
            recognized.append((match.start(), match.end(), match.group(1) + str(target)))

    bare_pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(str, sorted(years))) + r")(?!\w)") if years else None
    ambiguous = []
    for match in bare_pattern.finditer(text) if bare_pattern else ():
        start, end = match.span()
        if overlaps(start, end) or any(start < b and end > a for a, b, _ in recognized):
            continue
        if _financial_number(text, start, end):
            continue
        if YEAR_CONTEXT.search(text[max(0, start - 70):start]) or YEAR_SUFFIX.match(text[end:end + 40]):
            recognized.append((start, end, str(years[int(match.group())])))
        else:
            ambiguous.append((start, end, match.group()))

    if explicit is None:
        if ambiguous:
            start, end, value = ambiguous[0]
            raise ValueError(
                f"Ambiguous year-like value at {path}[{start}:{end}]: {value!r}. "
                "Supply complete year spans for this field; [] confirms it has no year references."
            )
        edits.extend(recognized)
    else:
        selected = []
        for span in explicit:
            if not isinstance(span, list) or len(span) != 3:
                raise ValueError(f"Expected [start, end, original_text] at {path}.")
            start, end, expected = span
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
                raise ValueError(f"Invalid year span at {path}: {span!r}")
            if text[start:end] != expected:
                raise ValueError(f"Year span no longer matches source text at {path}: {span!r}")
            if not _whole_token(text, start, end):
                raise ValueError(f"Year span is part of a larger token at {path}: {span!r}")
            match = re.fullmatch(r"(?i)(FY\s*)?(\d{4}|\d{2})", expected)
            if not match or (not match.group(1) and len(match.group(2)) != 4):
                raise ValueError(f"Expected a four-digit year or fiscal expression at {path}: {span!r}")
            if _financial_number(text, start, end):
                raise ValueError(f"Year span points into a financial amount at {path}: {span!r}")
            target = _mapped_year(match.group(2), years)
            if target is None:
                raise ValueError(f"Year span has no replacement mapping at {path}: {span!r}")
            selected.append((start, end, (match.group(1) or "") + str(target)))
        for start, end, _ in recognized:
            if not any(a <= start and b >= end for a, b, _ in selected):
                raise ValueError(f"Explicit year spans omit a recognized date at {path}[{start}:{end}].")
        edits.extend(selected)

    edits.sort()
    if any(left[1] > right[0] for left, right in zip(edits, edits[1:])):
        raise ValueError(f"Overlapping replacement spans at {path}.")
    result, offset = [], 0
    for start, end, replacement in edits:
        result.extend((text[offset:start], replacement))
        offset = end
    result.append(text[offset:])
    return "".join(result)


def relabel_record(original, target_company, year_map, company_aliases=(), year_spans=None):
    """Relabel originals; explicit field spans identify every date in that field."""
    if not isinstance(original, dict):
        raise ValueError("Each source record must be an object.")
    if original.get("synthetic") or original.get("source_financebench_id"):
        raise ValueError("Use original source records, not existing synthetic variants.")
    for key in ("financebench_id", "company", "question", "answer"):
        if not isinstance(original.get(key), str) or not original[key].strip():
            raise ValueError(f"Source {key} must be a nonempty string.")
    evidence_rows = original.get("evidence")
    if not isinstance(evidence_rows, list) or any(not isinstance(e, dict) for e in evidence_rows):
        raise ValueError("Source evidence must be a list of objects.")
    for evidence in evidence_rows:
        for key in ("evidence_text", "evidence_text_full_page"):
            if key in evidence and not isinstance(evidence[key], str):
                raise ValueError(f"Source {key} must be a string.")
    period = original.get("doc_period")
    if type(period) is not int or not 1000 <= period <= 9999:
        raise ValueError("Source doc_period must be a four-digit integer year.")
    if not isinstance(target_company, str) or not target_company.strip():
        raise ValueError("Target company must be a nonempty string.")
    source_id = original["financebench_id"].strip()
    source_company = original["company"].strip()
    target_company = target_company.strip()
    years = _years(year_map)
    if target_company.casefold() == source_company.casefold() and not years:
        raise ValueError("Replacement must change the company or a year.")
    if isinstance(company_aliases, str) or any(not isinstance(alias, str) for alias in company_aliases):
        raise ValueError("Company aliases must be a sequence of strings.")
    aliases = {source_company, *(alias.strip() for alias in company_aliases)}
    if "" in aliases:
        raise ValueError("Company aliases cannot be empty.")
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(a) for a in sorted(aliases, key=lambda a: (-len(a), a))) + r")(?!\w)", re.I)
    spans = {} if year_spans is None else year_spans
    if not isinstance(spans, dict):
        raise ValueError("Year spans must map field paths to lists of spans.")
    if any(not isinstance(value, list) for value in spans.values()):
        raise ValueError("Each year-span field must contain a list.")
    record = copy.deepcopy(original)
    for key in DERIVED_FIELDS:
        record.pop(key, None)
    fields = [(key, record, key) for key in ("question", "answer", "justification")]
    for index, evidence in enumerate(record.get("evidence", [])):
        for key in ("evidence_text", "evidence_text_full_page"):
            fields.append((f"evidence.{index}.{key}", evidence, key))
    paths = {path for path, container, key in fields if isinstance(container.get(key), str)}
    if spans.keys() - paths:
        raise ValueError(f"Unknown year-span fields: {sorted(spans.keys() - paths)}")
    year_changed = years.get(period, period) != period
    for path, container, key in fields:
        if isinstance(container.get(key), str):
            text = container[key]
            container[key] = _replace_text(text, pattern, target_company, years, path, spans.get(path))
            if target_company.casefold() == source_company.casefold():
                year_changed |= container[key] != _replace_text(text, pattern, target_company, {}, path)
    if target_company.casefold() == source_company.casefold() and not year_changed:
        raise ValueError("Replacement does not change any company or year reference.")

    period = years.get(period, period)
    recipe = {"source_id": source_id, "company": target_company, "years": years, "aliases": sorted(aliases), "year_spans": spans}
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]
    identifier = f"{source_id}__synthetic_{digest}"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", target_company).strip("_") or "company"
    doc_name = f"synthetic_{source_id}_{slug}_{period}_{original.get('doc_type', 'report')}"
    record.update({
        "financebench_id": identifier,
        "source_financebench_id": source_id,
        "company": target_company,
        "doc_period": period,
        "doc_name": doc_name,
        "synthetic": True,
        "source_metadata": {key: copy.deepcopy(original.get(key)) for key in SOURCE_FIELDS},
        "year_map": {str(source): target for source, target in years.items()},
    })
    record["source_metadata"]["financebench_id"] = source_id
    record["source_metadata"]["evidence_doc_names"] = [e.get("doc_name") for e in original.get("evidence", [])]
    for evidence in record.get("evidence", []):
        evidence["doc_name"] = doc_name
    record["doc_link"] = None
    record["gics_sector"] = None
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-id", action="append")
    parser.add_argument("--company", required=True)
    parser.add_argument("--company-alias", action="append", default=[])
    parser.add_argument("--year-map", action="append", default=[], metavar="SOURCE=TARGET")
    parser.add_argument("--year-spans", type=Path, help="JSON mapping source IDs to field paths and complete [start, end, original_text] year spans.")
    args = parser.parse_args()
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("Input and output paths must differ.")
        years = {}
        for expression in args.year_map:
            source, target = (int(part) for part in expression.split("="))
            if source in years and years[source] != target:
                raise ValueError(f"Conflicting mappings for {source}.")
            years[source] = target
        data = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Input must be a JSON list of original FinanceBench records.")
        span_map = json.loads(args.year_spans.read_text(encoding="utf-8")) if args.year_spans else {}
        if not isinstance(span_map, dict):
            raise ValueError("Year-span input must be a JSON object keyed by source ID.")
        selected = set(args.source_id) if args.source_id else None
        records, seen, found = [], set(), set()
        for original in data:
            if not isinstance(original, dict):
                raise ValueError("Each input record must be an object.")
            source_id = original.get("financebench_id", "")
            if selected is not None and source_id not in selected:
                continue
            found.add(source_id)
            record = relabel_record(original, args.company, years, args.company_alias, span_map.get(source_id))
            if record["financebench_id"] in seen:
                raise ValueError(f"Duplicate source/recipe: {source_id}")
            seen.add(record["financebench_id"])
            records.append(record)
        if selected is not None and selected - found:
            raise ValueError(f"Unknown source IDs: {sorted(selected - found)}")
        if not records:
            raise ValueError("No source records selected.")
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, args.output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"Wrote {len(records)} synthetic records to {args.output}")


if __name__ == "__main__":
    main()
