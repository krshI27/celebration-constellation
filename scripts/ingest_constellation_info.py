"""Ingest IAU constellation metadata and cache locally.

Outputs:
- data/supplemental/constellations_meta.json with abbrev, full_name, area_sq_deg.

Requires: astroquery, pandas.
"""

import json
from pathlib import Path

import pandas as pd
from astroquery.vizier import Vizier

OUT_PATH = (
    Path(__file__).parent.parent / "data" / "supplemental" / "constellations_meta.json"
)


def fetch_vi49() -> pd.DataFrame:
    viz = Vizier(columns=["Name", "abbr", "Area"])
    viz.row_limit = -1
    cats = vizier_get_catalogs_safe(viz, "VI/49")
    if not cats:
        raise RuntimeError("Failed to download VI/49 from VizieR")
    df = cats[0].to_pandas()
    df = df.rename(
        columns={"Name": "full_name", "abbr": "abbrev", "Area": "area_sq_deg"}
    )
    # Clean types
    df["abbrev"] = df["abbrev"].astype(str).str.upper()
    df["full_name"] = df["full_name"].astype(str)
    df["area_sq_deg"] = pd.to_numeric(df["area_sq_deg"], errors="coerce")
    return df[["abbrev", "full_name", "area_sq_deg"]]


def vizier_get_catalogs_safe(viz: Vizier, catalog: str):
    try:
        return viz.get_catalogs(catalog)
    except Exception as e:
        print(f"Error fetching {catalog}: {e}")
        return []


def main():
    df = fetch_vi49()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = df.to_dict(orient="records")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} constellations to {OUT_PATH}")


if __name__ == "__main__":
    main()
