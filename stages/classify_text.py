"""Stage 7 — Text region classification.

Each raw text block from Stage 1 gets a classification:

- ``room_number``: a short alphanumeric like "204" or "203a"
- ``sheet_callout``: a sheet identifier like "A302" or "A7.1"
- ``grid_label``: a single letter (A, B, GG) or single number (1, 4.1) used as a structural grid tag
- ``dimension``: measurements like 28'-6", 4'-4 5/8"
- ``room_label``: longer free-form text matching a room-type pattern (e.g., "BATHROOM", "Studio Unit 1")
- ``wall_schedule_tag``: a compact type identifier like "W-3" or "C2"
- ``note``: multi-word text not matching any pattern
- ``title``: typically the longest text at a specific position (sheet title); rough heuristic
- ``unknown``: everything else

Also emits ``cross_references`` for any text that looks like a sheet callout
encountered inside a different entity's context (e.g., "see A302" inside a
note, or "A302/1" inline).
"""

from __future__ import annotations

import re
import uuid


_WALL_SCHEDULE_PATTERN = re.compile(r"^[WCSwcs][-]?[0-9]{1,3}[a-z]?$")


def _first_match_room_label(
    text: str, patterns: dict
) -> tuple[str | None, str | None]:
    """Return (room_type_if_matched, matched_substring) or (None, None)."""
    norm = re.sub(r"\s+", " ", text).strip().upper()
    for room_type, substrings in patterns.items():
        for substring in substrings:
            if substring in norm:
                return room_type, substring
    return None, None


def _classify_one(text: str, config: dict) -> tuple[str, list[str], str]:
    """Return (classification, sheet_references_found_in_text, rule_triggered)."""
    if not text or not text.strip():
        return "unknown", [], "empty_text"
    t = text.strip()

    # Collect any sheet-callout-looking substrings (for cross-reference resolution).
    # Run against the stripped text; could match "A302" even if it's embedded in a note.
    sheet_refs: list[str] = []
    sheet_pattern = re.compile(config["text_sheet_callout_pattern"])
    for token in re.split(r"[\s,;/]+", t):
        if sheet_pattern.match(token):
            sheet_refs.append(token)

    # Room number: short alphanumeric like 204, 203a
    if re.match(config["text_room_number_pattern"], t):
        return "room_number", sheet_refs, "room_number_pattern"

    # Sheet callout: whole text IS a sheet identifier
    if sheet_pattern.match(t):
        return "sheet_callout", sheet_refs, "sheet_callout_pattern"

    # Grid label: single letter / single digit / A.1 — but exclude common
    # fixture/callout abbreviations that match the same pattern (DW, DN, REF, UP...).
    if re.match(config["text_grid_label_pattern"], t):
        blocklist = {str(x).upper() for x in (config.get("text_grid_label_blocklist") or [])}
        if t.upper() not in blocklist:
            return "grid_label", sheet_refs, "grid_label_pattern"

    # Dimension
    if re.match(config["text_dimension_pattern"], t):
        return "dimension", sheet_refs, "dimension_pattern"

    # Wall-schedule tag (W-3, C2)
    if _WALL_SCHEDULE_PATTERN.match(t):
        return "wall_schedule_tag", sheet_refs, "wall_schedule_tag_pattern"

    # Room label — must match a room-type pattern
    room_type, _matched = _first_match_room_label(t, config.get("room_type_label_patterns") or {})
    if room_type is not None:
        return "room_label", sheet_refs, "room_label_pattern"

    # Title vs. note heuristic: long text with very large bbox was historically
    # a title, but we don't have bbox area here. Default to note for multi-word.
    if len(t.split()) >= 2:
        return "note", sheet_refs, "multiword_note_heuristic"

    return "unknown", sheet_refs, "no_pattern_match"


def classify_text_regions(
    extracted: dict, config: dict
) -> tuple[list[dict], list[dict]]:
    """Assign a text_region_id and classification to each text block.

    Returns (text_regions, cross_references). Mutates the input text_blocks to
    attach ``_text_region_id`` for downstream reference.
    """
    text_blocks = extracted.get("text_blocks") or []
    regions: list[dict] = []
    cross_refs: list[dict] = []

    for tb in text_blocks:
        region_id = tb.get("_text_region_id") or str(uuid.uuid4())
        tb["_text_region_id"] = region_id
        text = (tb.get("text") or "").strip()
        classification, sheet_refs, rule = _classify_one(text, config)
        bbox = tb.get("bbox") or [0, 0, 0, 0]
        # A text region whose role depends on an external sheet (sheet_callout
        # or any embedded sheet reference) can't be confirmed from this drawing
        # alone — that's H2's empirical footprint on the text side.
        requires_xdoc = bool(sheet_refs) or classification == "sheet_callout"
        regions.append(
            {
                "text_region_id": region_id,
                "text": text,
                "bbox": [float(v) for v in bbox],
                "classification": classification,
                "references": [ref for ref in sheet_refs] if sheet_refs else [],
                "enclosing_room_id": None,
                "linked_entity_ids": [],
                "rule_triggered": rule,
                "requires_cross_document_validation": requires_xdoc,
            }
        )
        # Emit cross-references for sheet callouts (even if the text is only
        # PART of a longer note — e.g., "see A302, detail 1")
        for ref in sheet_refs:
            # Try to extract a detail number if present nearby
            detail = None
            m = re.search(r"{}\s*[/,]?\s*([0-9]{{1,3}})".format(re.escape(ref)), text)
            if m:
                detail = m.group(1)
            cross_refs.append(
                {
                    "cross_reference_id": str(uuid.uuid4()),
                    "source_text_region_id": region_id,
                    "target_sheet": ref,
                    "target_detail": detail,
                    "context": classification,
                    "rule_triggered": "sheet_callout_token_in_text",
                    "requires_cross_document_validation": True,
                }
            )

    return regions, cross_refs
