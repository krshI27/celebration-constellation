"""Star catalog data management using Yale Bright Star Catalog.

This module downloads, caches, and manages star catalog data for constellation
matching. Uses the Yale Bright Star Catalog (BSC5) which contains ~9,000 stars
and is very lightweight (< 1 MB).

Supports offline mode using local catalog files from data/supplemental/.
"""

import gzip
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier


def parse_local_catalog(catalog_path: Path) -> pd.DataFrame:
    """Parse Bright Star Catalogue V/50 from local fixed-width format file.

    Format specification from ReadMe:
    - Bytes 1-4: HR number (Harvard Revised Number)
    - Bytes 76-77: RAh (hours, J2000)
    - Bytes 78-79: RAm (minutes, J2000)
    - Bytes 80-83: RAs (seconds, J2000)
    - Byte 84: DE- (sign, J2000)
    - Bytes 85-86: DEd (degrees, J2000)
    - Bytes 87-88: DEm (arcminutes, J2000)
    - Bytes 89-90: DEs (arcseconds, J2000)
    - Bytes 103-107: Vmag (visual magnitude)

    Args:
        catalog_path: Path to catalog.gz file

    Returns:
        DataFrame with columns: star_id, ra, dec, magnitude

    Raises:
        FileNotFoundError: If catalog file doesn't exist
        ValueError: If catalog parsing fails
    """
    if not catalog_path.exists():
        raise FileNotFoundError(f"Local catalog not found: {catalog_path}")

    print(f"Loading local catalog from {catalog_path}...")

    records = []

    with gzip.open(catalog_path, "rt", encoding="ascii", errors="ignore") as f:
        for line in f:
            if len(line) < 107:
                continue

            try:
                # Extract HR number (bytes 1-4)
                hr_str = line[0:4].strip()
                if not hr_str:
                    continue
                star_id = int(hr_str)

                # Extract RA J2000 (bytes 76-83)
                ra_h = line[75:77].strip()
                ra_m = line[77:79].strip()
                ra_s = line[79:83].strip()

                # Extract Dec J2000 (bytes 84-90)
                dec_sign = line[83:84].strip()
                dec_d = line[84:86].strip()
                dec_m = line[86:88].strip()
                dec_s = line[88:90].strip()

                # Extract visual magnitude (bytes 103-107)
                vmag_str = line[102:107].strip()

                # Skip if any required field is missing
                if not all([ra_h, ra_m, ra_s, dec_d, dec_m, dec_s, vmag_str]):
                    continue

                # Convert RA from sexagesimal to decimal degrees
                ra_hours = float(ra_h) + float(ra_m) / 60.0 + float(ra_s) / 3600.0
                ra_deg = ra_hours * 15.0  # Convert hours to degrees

                # Convert Dec from sexagesimal to decimal degrees
                dec_deg = float(dec_d) + float(dec_m) / 60.0 + float(dec_s) / 3600.0
                if dec_sign == "-":
                    dec_deg = -dec_deg

                # Parse magnitude
                magnitude = float(vmag_str)

                records.append(
                    {
                        "star_id": star_id,
                        "ra": ra_deg,
                        "dec": dec_deg,
                        "magnitude": magnitude,
                    }
                )

            except (ValueError, IndexError):
                continue

    if not records:
        raise ValueError("No valid star records found in catalog")

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} stars from local catalog")

    return df


class StarCatalog:
    """Manager for star catalog data."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize star catalog manager.

        Args:
            cache_dir: Directory for caching catalog data
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".drinking_galaxies" / "star_cache"

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_file = self.cache_dir / "yale_bright_stars.csv"
        self._catalog = None

    def download_catalog(self, force_refresh: bool = False) -> pd.DataFrame:
        """Load Yale Bright Star Catalog from local file or download from VizieR.

        Priority order:
        1. Use cached CSV if available and not force_refresh
        2. Try loading from local catalog.gz in data/supplemental/
        3. Fall back to VizieR download (requires network)

        Args:
            force_refresh: Force re-download even if cache exists

        Returns:
            DataFrame with star positions and magnitudes

        Note:
            Yale BSC5 (V/50) contains ~9,000 stars visible to naked eye.
            Very lightweight at < 1 MB.
        """
        if self.catalog_file.exists() and not force_refresh:
            return pd.read_csv(self.catalog_file)

        # Try loading from local supplemental data first
        local_catalog_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "supplemental"
            / "catalog.gz"
        )

        if local_catalog_path.exists():
            try:
                catalog = parse_local_catalog(local_catalog_path)

                # Save to cache
                catalog.to_csv(self.catalog_file, index=False)
                print(f"Saved catalog to {self.catalog_file}")

                return catalog

            except Exception as e:
                print(f"Warning: Failed to parse local catalog: {e}")
                print("Falling back to VizieR download...")

        # Fall back to VizieR if local file not available
        print("Downloading Yale Bright Star Catalog from VizieR...")

        # Query Yale BSC5 catalog (V/50)
        vizier = Vizier(
            columns=["HR", "RAJ2000", "DEJ2000", "Vmag", "pmRA", "pmDE"],
            row_limit=-1,
        )

        catalog_list = vizier.get_catalogs("V/50")

        if not catalog_list:
            raise RuntimeError("Failed to download star catalog from VizieR")

        catalog = catalog_list[0].to_pandas()

        # Clean and standardize column names
        catalog = catalog.rename(
            columns={
                "HR": "star_id",
                "RAJ2000": "ra",
                "DEJ2000": "dec",
                "Vmag": "magnitude",
                "pmRA": "pm_ra",
                "pmDE": "pm_dec",
            }
        )

        # Remove stars without position data (including empty strings)
        catalog = catalog.replace("", np.nan)
        catalog = catalog.dropna(subset=["ra", "dec", "magnitude"])

        # Convert RA/Dec from sexagesimal to decimal degrees if needed
        if catalog["ra"].dtype == object:  # String format
            coords = SkyCoord(
                ra=list(catalog["ra"].values),
                dec=list(catalog["dec"].values),
                unit=(u.hourangle, u.deg),
            )
            catalog["ra"] = coords.ra.deg
            catalog["dec"] = coords.dec.deg

        # Convert other numeric columns
        numeric_cols = ["magnitude", "pm_ra", "pm_dec"]
        for col in numeric_cols:
            if col in catalog.columns:
                catalog[col] = pd.to_numeric(catalog[col], errors="coerce")

        # Remove any remaining rows with invalid data
        catalog = catalog.dropna(subset=["ra", "dec", "magnitude"])

        # Save to cache
        catalog.to_csv(self.catalog_file, index=False)
        print(f"Saved catalog to {self.catalog_file}")

        return catalog

    def load_catalog(self) -> pd.DataFrame:
        """Load catalog from cache or download if not available.

        Returns:
            DataFrame with star catalog data
        """
        if self._catalog is None:
            self._catalog = self.download_catalog()

        return self._catalog

    def get_stars_in_region(
        self,
        ra_center: float,
        dec_center: float,
        radius_deg: float = 20.0,
        max_magnitude: float = 6.0,
    ) -> pd.DataFrame:
        """Get stars within a circular region of the sky.

        Args:
            ra_center: Right ascension of region center (degrees)
            dec_center: Declination of region center (degrees)
            radius_deg: Radius of search region (degrees)
            max_magnitude: Maximum visual magnitude (dimmer stars excluded)

        Returns:
            DataFrame with stars in the specified region

        Example:
            >>> catalog = StarCatalog()
            >>> stars = catalog.get_stars_in_region(ra_center=83.8, dec_center=-5.4, radius_deg=15)
            >>> print(f"Found {len(stars)} stars near Orion")
        """
        catalog = self.load_catalog()

        # Filter by magnitude first (faster)
        bright_stars = catalog[catalog["magnitude"] <= max_magnitude].copy()

        # Create SkyCoord for vectorized angular separation
        center = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg)
        star_coords = SkyCoord(
            ra=bright_stars["ra"].values * u.deg,
            dec=bright_stars["dec"].values * u.deg,
        )

        separations = center.separation(star_coords)
        mask = separations <= radius_deg * u.deg

        return bright_stars[mask].copy()

    def get_all_bright_stars(self, max_magnitude: float = 6.0) -> pd.DataFrame:
        """Get all stars brighter than specified magnitude.

        Args:
            max_magnitude: Maximum visual magnitude

        Returns:
            DataFrame with bright stars
        """
        catalog = self.load_catalog()
        return catalog[catalog["magnitude"] <= max_magnitude].copy()

    def convert_to_stereographic(
        self,
        stars: pd.DataFrame,
        ra_center: float,
        dec_center: float,
    ) -> np.ndarray:
        """Convert star positions to stereographic projection.

        Stereographic projection maps the celestial sphere to a plane,
        preserving local angles and shapes.

        Args:
            stars: DataFrame with 'ra' and 'dec' columns
            ra_center: Right ascension of projection center (degrees)
            dec_center: Declination of projection center (degrees)

        Returns:
            Array of (x, y) coordinates in stereographic projection
        """
        # Convert to radians
        ra = np.deg2rad(stars["ra"].values)
        dec = np.deg2rad(stars["dec"].values)
        ra0 = np.deg2rad(ra_center)
        dec0 = np.deg2rad(dec_center)

        # Stereographic projection formulas
        cos_c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(
            ra - ra0
        )
        k = 2 / (1 + cos_c)

        x = k * np.cos(dec) * np.sin(ra - ra0)
        y = k * (
            np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)
        )

        return np.column_stack([x, y])

    def search_sky_regions(
        self,
        num_samples: int = 100,
        max_magnitude: float = 6.0,
    ) -> list[dict]:
        """Sample random sky regions for constellation matching.

        Args:
            num_samples: Number of sky regions to sample
            max_magnitude: Maximum star magnitude

        Returns:
            List of dicts with 'ra', 'dec', and 'stars' for each region
        """
        # Generate random RA/Dec samples
        ra_samples = np.random.uniform(0, 360, num_samples)
        dec_samples = np.random.uniform(-90, 90, num_samples)

        regions = []
        for ra, dec in zip(ra_samples, dec_samples):
            stars = self.get_stars_in_region(ra, dec, max_magnitude=max_magnitude)
            if len(stars) > 0:
                regions.append({"ra": ra, "dec": dec, "stars": stars})

        return regions
