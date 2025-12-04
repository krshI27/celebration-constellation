# Constellation Data Sources

This project uses public astronomical catalogs to enable constellation matching and visualization. Below is a concise overview of the sources, licensing, and how we use them.

## Core Catalogs

- Bright Stars (`V/50` – Yale Bright Star Catalog):
  - Contents: Bright star positions (RA/Dec), magnitudes, color index (B–V), identifiers (HR/HD/HIP).
  - Use: Matching and rendering star fields; magnitude-based point sizes; optional B–V colorization.
  - Access: Via `astroquery.vizier` or cached local files.
  - License: Public domain-like; distributed by VizieR. Attribution recommended.

- IAU Constellation Boundaries (`VI/49`):
  - Contents: Official IAU polygon boundaries for constellations.
  - Use: Identify which constellation a sampled sky region falls into; display basic info and area.
  - Access: Via `astroquery.vizier` or local cache.
  - License: Public reference data; attribution recommended.

## Optional “Stick Figure” Lines

- Dataset goal: Line segments connecting notable stars within each constellation for visual reference.
- Constraints: Avoid GPL-only datasets; prefer permissive (e.g., CC BY-SA) with clear attribution.

### Recommended Options

- HYG Database (cross-IDs + positions):
  - Use to reconcile identifiers (HIP/HD/Bayer) to HR where needed.
  - License: CC BY-SA.
  - Pair with an open constellation-lines JSON (IAU abbreviations → HR pairs).

- Custom `constellation_lines.json` (local file):
  - Format: `{ "ORI": [[1713,1852], ...], "UMA": [[...], ...] }` where numbers are HR IDs.
  - Location: `data/supplemental/constellation_lines.json`.
  - License: Ensure the source for these pairs is permissive; include attribution in this document.

## Implementation Notes

- Lines Loader: `celebration_constellation/lines.py`
  - `load_constellation_lines(path=None) -> dict[str, list[tuple[int,int]]]`
  - `build_line_segments_for_region(stars_df, abbrev, lines_map) -> list[tuple[int,int]]`
  - Behavior: Returns empty data if file missing; UI toggle handles gracefully.

- Matching Pipeline:
  - Circle detection → RANSAC similarity transform against bright stars in sampled sky windows.
  - Inverse transform maps catalog star positions back into image coordinates for rendering.
  - Constellation identification via `VI/49` boundary lookup.

## Attribution

When using external datasets, include a short attribution string here, for example:

- “Bright Star Catalog (V/50), courtesy of VizieR (CDS, Strasbourg).”
- “IAU Constellation Boundaries (VI/49), courtesy of VizieR.”
- “Constellation line segments derived from [SOURCE NAME], licensed under CC BY-SA.”

## Future Work

- Build a small ingestion script to merge line-segment datasets with HR IDs and cache to `data/supplemental/constellation_lines.json`.
- Add unit tests to verify HR mapping correctness for prominent constellations (e.g., ORI, UMA, SCO).

### Quick Ingestion

- Constellation metadata (names, areas) from `VI/49`:
  - Script: `scripts/ingest_constellation_info.py`
  - Output: `data/supplemental/constellations_meta.json`
  - Run:

    ```zsh
    conda activate drinking-galaxies
    python scripts/ingest_constellation_info.py
    ```
