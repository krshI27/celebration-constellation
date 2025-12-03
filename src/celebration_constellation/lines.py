"""Constellation stick-figure line support.

This module provides optional line segment data for constellations and
utilities to map line definitions (by HR id pairs) to indices in a star
DataFrame for rendering with existing visualization helpers.

If no data file is available, the loader returns an empty dictionary and
callers should handle the absence gracefully (e.g., disable lines toggle).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

DEFAULT_LINES_PATH = (
    Path(__file__).parent.parent.parent
    / "data"
    / "supplemental"
    / "constellation_lines.json"
)


def load_constellation_lines(
    path: Path | None = None,
) -> Dict[str, List[Tuple[int, int]]]:
    """Load constellation line definitions.

    The JSON file should map IAU 3-letter abbreviations (e.g., "ORI") to
    a list of pairs of HR ids, for example: {"ORI": [[1713, 1852], ...]}.

    Args:
        path: Optional custom path. Defaults to data/supplemental/constellation_lines.json

    Returns:
        Dict mapping constellation abbrev to list of (HR1, HR2) pairs.
        Returns empty dict if file does not exist or is invalid.
    """
    p = path or DEFAULT_LINES_PATH
    try:
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        result: Dict[str, List[Tuple[int, int]]] = {}
        for key, pairs in raw.items():
            clean_pairs: List[Tuple[int, int]] = []
            for pair in pairs:
                if (
                    isinstance(pair, (list, tuple))
                    and len(pair) == 2
                    and isinstance(pair[0], int)
                    and isinstance(pair[1], int)
                ):
                    clean_pairs.append((pair[0], pair[1]))
            if clean_pairs:
                result[key.upper()] = clean_pairs
        return result
    except Exception:
        return {}


def build_line_segments_for_region(
    stars_df: pd.DataFrame,
    constellation_abbrev: str | None,
    lines_map: Dict[str, List[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Map constellation HR-id line pairs to index pairs within a region.

    Args:
        stars_df: Region star DataFrame with a 'star_id' (HR) column
        constellation_abbrev: 3-letter IAU abbrev (e.g., 'ORI'); None allowed
        lines_map: Output from load_constellation_lines()

    Returns:
        List of (start_idx, end_idx) index pairs into stars_df rows. Empty if
        lines unknown or mapping fails.
    """
    if not isinstance(constellation_abbrev, str):
        return []
    key = constellation_abbrev.upper()
    if key not in lines_map:
        return []

    if "star_id" not in stars_df.columns:
        return []

    # Build lookup from HR id to index in this region
    id_to_idx = {int(hr): idx for idx, hr in enumerate(stars_df["star_id"].tolist())}

    segments: List[Tuple[int, int]] = []
    for hr1, hr2 in lines_map[key]:
        i = id_to_idx.get(int(hr1))
        j = id_to_idx.get(int(hr2))
        if i is not None and j is not None and i != j:
            segments.append((i, j))

    return segments
