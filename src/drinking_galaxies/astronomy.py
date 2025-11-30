"""Star catalog data management using Yale Bright Star Catalog.

This module downloads, caches, and manages star catalog data for constellation
matching. Uses the Yale Bright Star Catalog (BSC5) which contains ~9,000 stars
and is very lightweight (< 1 MB).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier


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
        """Download Yale Bright Star Catalog from VizieR.

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

        # Remove stars without position data
        catalog = catalog.dropna(subset=["ra", "dec"])

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
