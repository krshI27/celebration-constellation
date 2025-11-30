"""Constellation identification using IAU constellation boundaries.

This module downloads and manages IAU constellation boundary data,
providing constellation identification for celestial coordinates.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astroquery.vizier import Vizier


class ConstellationCatalog:
    """Manager for IAU constellation boundary data."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize constellation catalog manager.

        Args:
            cache_dir: Directory for caching boundary data
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".drinking_galaxies" / "constellation_cache"

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.boundary_file = self.cache_dir / "iau_boundaries.csv"
        self._boundaries = None

    def download_boundaries(
        self, force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """Download IAU constellation boundaries from VizieR.

        Args:
            force_refresh: Force re-download even if cache exists

        Returns:
            DataFrame with constellation boundary data, or None if download fails

        Note:
            Uses IAU constellation boundaries catalog (VI/49).
            Contains boundary polygons for all 88 constellations.
            Gracefully returns None if VizieR is unavailable (offline mode).
        """
        if self.boundary_file.exists() and not force_refresh:
            return pd.read_csv(self.boundary_file)

        print("Downloading IAU constellation boundaries from VizieR...")

        try:
            # Query IAU constellation boundaries (VI/49)
            vizier = Vizier(
                columns=["Constellation", "RAhr", "DEdeg"],
                row_limit=-1,
            )

            catalog_list = vizier.get_catalogs("VI/49")

            if not catalog_list:
                print("Warning: No constellation boundary data returned from VizieR")
                return None

            catalog = catalog_list[0].to_pandas()

            # Clean and standardize column names
            catalog = catalog.rename(
                columns={
                    "Constellation": "constellation",
                    "RAhr": "ra_hours",
                    "DEdeg": "dec_deg",
                }
            )

            # Convert RA from hours to degrees
            catalog["ra"] = catalog["ra_hours"] * 15.0  # 1 hour = 15 degrees

            # Use Dec directly
            catalog["dec"] = catalog["dec_deg"]

            # Remove any rows with missing data
            catalog = catalog.dropna(subset=["constellation", "ra", "dec"])

            # Keep only needed columns
            catalog = catalog[["constellation", "ra", "dec"]]

            # Save to cache
            catalog.to_csv(self.boundary_file, index=False)
            print(f"Saved constellation boundaries to {self.boundary_file}")

            return catalog

        except Exception as e:
            print(
                f"Warning: Failed to download constellation boundaries: {e}"
            )
            print("App will continue without constellation identification")
            return None

    def load_boundaries(self) -> Optional[pd.DataFrame]:
        """Load constellation boundaries from cache or download if not available.

        Returns:
            DataFrame with constellation boundary data, or None if unavailable
        """
        if self._boundaries is None:
            self._boundaries = self.download_boundaries()

        return self._boundaries

    def _prepare_polygon(self, boundary_points: pd.DataFrame) -> np.ndarray:
        """Prepare constellation boundary polygon for point-in-polygon test.

        Args:
            boundary_points: DataFrame with RA/Dec boundary vertices

        Returns:
            Array of polygon vertices (ra, dec) with polygon closed
        """
        # Sort by RA to get proper polygon order
        boundary_points = boundary_points.sort_values("ra")

        # Extract coordinates
        polygon = boundary_points[["ra", "dec"]].values

        # Close the polygon if not already closed
        if len(polygon) > 0 and not np.allclose(polygon[0], polygon[-1]):
            polygon = np.vstack([polygon, polygon[0:1]])

        return polygon

    def _point_in_polygon(
        self,
        ra: float,
        dec: float,
        polygon: np.ndarray,
    ) -> bool:
        """Test if point is inside polygon using ray-casting algorithm.

        Adapted for spherical coordinates with RA wrap-around handling.

        Args:
            ra: Right ascension in degrees (0-360)
            dec: Declination in degrees (-90 to 90)
            polygon: Array of (ra, dec) vertices

        Returns:
            True if point is inside polygon, False otherwise
        """
        n = len(polygon)
        inside = False

        # Normalize RA to [0, 360)
        ra = ra % 360

        # Ray-casting algorithm
        p1_ra, p1_dec = polygon[0]
        for i in range(1, n + 1):
            p2_ra, p2_dec = polygon[i % n]

            # Handle RA wrap-around
            # If polygon edge crosses 0°/360° boundary, shift point and polygon
            if abs(p2_ra - p1_ra) > 180:
                if p1_ra > 180:
                    p1_ra -= 360
                if p2_ra > 180:
                    p2_ra -= 360
                if ra > 180:
                    ra_test = ra - 360
                else:
                    ra_test = ra
            else:
                ra_test = ra

            # Standard ray-casting test
            if dec > min(p1_dec, p2_dec):
                if dec <= max(p1_dec, p2_dec):
                    if ra_test <= max(p1_ra, p2_ra):
                        if p1_dec != p2_dec:
                            x_inters = (dec - p1_dec) * (p2_ra - p1_ra) / (
                                p2_dec - p1_dec
                            ) + p1_ra
                            if p1_ra == p2_ra or ra_test <= x_inters:
                                inside = not inside

            p1_ra, p1_dec = polygon[i % n]

        return inside

    def identify_constellation(self, ra: float, dec: float) -> Optional[str]:
        """Identify constellation containing the given celestial coordinates.

        Args:
            ra: Right ascension in degrees (0-360)
            dec: Declination in degrees (-90 to 90)

        Returns:
            Constellation name (3-letter IAU abbreviation) or None if not found
            Returns None if constellation boundaries are unavailable (offline mode)

        Example:
            >>> catalog = ConstellationCatalog()
            >>> constellation = catalog.identify_constellation(83.8, -5.4)
            >>> print(constellation)
            Ori
        """
        boundaries = self.load_boundaries()

        # Return None if boundaries unavailable (offline mode)
        if boundaries is None:
            return None

        # Group by constellation
        for constellation_name, group in boundaries.groupby("constellation"):
            polygon = self._prepare_polygon(group)

            if self._point_in_polygon(ra, dec, polygon):
                return constellation_name

        return None

    def get_constellation_info(self, name: Optional[str]) -> Optional[dict]:
        """Get detailed information about a constellation.

        Args:
            name: Constellation name (3-letter IAU abbreviation)

        Returns:
            Dictionary with constellation metadata or None if not found

        Example:
            >>> catalog = ConstellationCatalog()
            >>> info = catalog.get_constellation_info("Ori")
            >>> print(info["full_name"])
            Orion
        """
        if name is None:
            return None

        # Constellation metadata dictionary
        # Based on IAU official constellation data
        metadata = CONSTELLATION_METADATA.get(name.upper())

        if metadata:
            return {
                "abbrev": name.upper(),
                **metadata,
            }

        return None


# IAU constellation metadata
# Full names, areas, and brief descriptions
CONSTELLATION_METADATA = {
    "AND": {
        "full_name": "Andromeda",
        "area_sq_deg": 722,
        "description": "The Chained Princess, home to the Andromeda Galaxy",
    },
    "ANT": {
        "full_name": "Antlia",
        "area_sq_deg": 239,
        "description": "The Air Pump, a faint southern constellation",
    },
    "APS": {
        "full_name": "Apus",
        "area_sq_deg": 206,
        "description": "The Bird of Paradise, a southern circumpolar constellation",
    },
    "AQR": {
        "full_name": "Aquarius",
        "area_sq_deg": 980,
        "description": "The Water Bearer, one of the zodiac constellations",
    },
    "AQL": {
        "full_name": "Aquila",
        "area_sq_deg": 652,
        "description": "The Eagle, featuring bright star Altair",
    },
    "ARA": {
        "full_name": "Ara",
        "area_sq_deg": 237,
        "description": "The Altar, a southern constellation",
    },
    "ARI": {
        "full_name": "Aries",
        "area_sq_deg": 441,
        "description": "The Ram, first constellation of the zodiac",
    },
    "AUR": {
        "full_name": "Auriga",
        "area_sq_deg": 657,
        "description": "The Charioteer, featuring bright star Capella",
    },
    "BOO": {
        "full_name": "Bootes",
        "area_sq_deg": 907,
        "description": "The Herdsman, home to bright star Arcturus",
    },
    "CAE": {
        "full_name": "Caelum",
        "area_sq_deg": 125,
        "description": "The Chisel, a small southern constellation",
    },
    "CAM": {
        "full_name": "Camelopardalis",
        "area_sq_deg": 757,
        "description": "The Giraffe, a large but faint northern constellation",
    },
    "CNC": {
        "full_name": "Cancer",
        "area_sq_deg": 506,
        "description": "The Crab, zodiac constellation with Beehive Cluster",
    },
    "CVN": {
        "full_name": "Canes Venatici",
        "area_sq_deg": 465,
        "description": "The Hunting Dogs, home to the Whirlpool Galaxy",
    },
    "CMA": {
        "full_name": "Canis Major",
        "area_sq_deg": 380,
        "description": "The Great Dog, featuring Sirius the brightest star",
    },
    "CMI": {
        "full_name": "Canis Minor",
        "area_sq_deg": 183,
        "description": "The Little Dog, featuring bright star Procyon",
    },
    "CAP": {
        "full_name": "Capricornus",
        "area_sq_deg": 414,
        "description": "The Sea Goat, a zodiac constellation",
    },
    "CAR": {
        "full_name": "Carina",
        "area_sq_deg": 494,
        "description": "The Keel, featuring Canopus the second brightest star",
    },
    "CAS": {
        "full_name": "Cassiopeia",
        "area_sq_deg": 598,
        "description": "The Queen, featuring a distinctive W-shape",
    },
    "CEN": {
        "full_name": "Centaurus",
        "area_sq_deg": 1060,
        "description": "The Centaur, home to Alpha Centauri",
    },
    "CEP": {
        "full_name": "Cepheus",
        "area_sq_deg": 588,
        "description": "The King, a northern circumpolar constellation",
    },
    "CET": {
        "full_name": "Cetus",
        "area_sq_deg": 1231,
        "description": "The Whale, fourth largest constellation",
    },
    "CHA": {
        "full_name": "Chamaeleon",
        "area_sq_deg": 132,
        "description": "The Chameleon, a small southern circumpolar constellation",
    },
    "CIR": {
        "full_name": "Circinus",
        "area_sq_deg": 93,
        "description": "The Compass, a small southern constellation",
    },
    "COL": {
        "full_name": "Columba",
        "area_sq_deg": 270,
        "description": "The Dove, a southern constellation",
    },
    "COM": {
        "full_name": "Coma Berenices",
        "area_sq_deg": 386,
        "description": "Berenice's Hair, home to the Coma Cluster",
    },
    "CRA": {
        "full_name": "Corona Australis",
        "area_sq_deg": 128,
        "description": "The Southern Crown, a small constellation",
    },
    "CRB": {
        "full_name": "Corona Borealis",
        "area_sq_deg": 179,
        "description": "The Northern Crown, a small but distinctive constellation",
    },
    "CRV": {
        "full_name": "Corvus",
        "area_sq_deg": 184,
        "description": "The Crow, a small southern constellation",
    },
    "CRT": {
        "full_name": "Crater",
        "area_sq_deg": 282,
        "description": "The Cup, a faint southern constellation",
    },
    "CRU": {
        "full_name": "Crux",
        "area_sq_deg": 68,
        "description": "The Southern Cross, smallest constellation",
    },
    "CYG": {
        "full_name": "Cygnus",
        "area_sq_deg": 804,
        "description": "The Swan, featuring the Northern Cross asterism",
    },
    "DEL": {
        "full_name": "Delphinus",
        "area_sq_deg": 189,
        "description": "The Dolphin, a small but distinctive constellation",
    },
    "DOR": {
        "full_name": "Dorado",
        "area_sq_deg": 179,
        "description": "The Dolphinfish, home to Large Magellanic Cloud",
    },
    "DRA": {
        "full_name": "Draco",
        "area_sq_deg": 1083,
        "description": "The Dragon, a large northern circumpolar constellation",
    },
    "EQU": {
        "full_name": "Equuleus",
        "area_sq_deg": 72,
        "description": "The Little Horse, second smallest constellation",
    },
    "ERI": {
        "full_name": "Eridanus",
        "area_sq_deg": 1138,
        "description": "The River, sixth largest constellation",
    },
    "FOR": {
        "full_name": "Fornax",
        "area_sq_deg": 398,
        "description": "The Furnace, a southern constellation",
    },
    "GEM": {
        "full_name": "Gemini",
        "area_sq_deg": 514,
        "description": "The Twins, featuring stars Castor and Pollux",
    },
    "GRU": {
        "full_name": "Grus",
        "area_sq_deg": 366,
        "description": "The Crane, a southern constellation",
    },
    "HER": {
        "full_name": "Hercules",
        "area_sq_deg": 1225,
        "description": "The Hero, fifth largest constellation",
    },
    "HOR": {
        "full_name": "Horologium",
        "area_sq_deg": 249,
        "description": "The Clock, a faint southern constellation",
    },
    "HYA": {
        "full_name": "Hydra",
        "area_sq_deg": 1303,
        "description": "The Water Snake, largest constellation",
    },
    "HYI": {
        "full_name": "Hydrus",
        "area_sq_deg": 243,
        "description": "The Little Water Snake, a southern constellation",
    },
    "IND": {
        "full_name": "Indus",
        "area_sq_deg": 294,
        "description": "The Indian, a southern constellation",
    },
    "LAC": {
        "full_name": "Lacerta",
        "area_sq_deg": 201,
        "description": "The Lizard, a small northern constellation",
    },
    "LEO": {
        "full_name": "Leo",
        "area_sq_deg": 947,
        "description": "The Lion, zodiac constellation featuring Regulus",
    },
    "LMI": {
        "full_name": "Leo Minor",
        "area_sq_deg": 232,
        "description": "The Little Lion, a small northern constellation",
    },
    "LEP": {
        "full_name": "Lepus",
        "area_sq_deg": 290,
        "description": "The Hare, a southern constellation",
    },
    "LIB": {
        "full_name": "Libra",
        "area_sq_deg": 538,
        "description": "The Scales, a zodiac constellation",
    },
    "LUP": {
        "full_name": "Lupus",
        "area_sq_deg": 334,
        "description": "The Wolf, a southern constellation",
    },
    "LYN": {
        "full_name": "Lynx",
        "area_sq_deg": 545,
        "description": "The Lynx, a large but faint northern constellation",
    },
    "LYR": {
        "full_name": "Lyra",
        "area_sq_deg": 286,
        "description": "The Lyre, featuring bright star Vega",
    },
    "MEN": {
        "full_name": "Mensa",
        "area_sq_deg": 153,
        "description": "The Table Mountain, faintest constellation",
    },
    "MIC": {
        "full_name": "Microscopium",
        "area_sq_deg": 210,
        "description": "The Microscope, a small southern constellation",
    },
    "MON": {
        "full_name": "Monoceros",
        "area_sq_deg": 482,
        "description": "The Unicorn, home to the Rosette Nebula",
    },
    "MUS": {
        "full_name": "Musca",
        "area_sq_deg": 138,
        "description": "The Fly, a southern constellation",
    },
    "NOR": {
        "full_name": "Norma",
        "area_sq_deg": 165,
        "description": "The Carpenter's Square, a small southern constellation",
    },
    "OCT": {
        "full_name": "Octans",
        "area_sq_deg": 291,
        "description": "The Octant, contains the south celestial pole",
    },
    "OPH": {
        "full_name": "Ophiuchus",
        "area_sq_deg": 948,
        "description": "The Serpent Bearer, sometimes called 13th zodiac",
    },
    "ORI": {
        "full_name": "Orion",
        "area_sq_deg": 594,
        "description": "The Hunter, one of the most recognizable constellations",
    },
    "PAV": {
        "full_name": "Pavo",
        "area_sq_deg": 378,
        "description": "The Peacock, a southern constellation",
    },
    "PEG": {
        "full_name": "Pegasus",
        "area_sq_deg": 1121,
        "description": "The Winged Horse, seventh largest constellation",
    },
    "PER": {
        "full_name": "Perseus",
        "area_sq_deg": 615,
        "description": "The Hero, home to the Perseus Double Cluster",
    },
    "PHE": {
        "full_name": "Phoenix",
        "area_sq_deg": 469,
        "description": "The Phoenix, a southern constellation",
    },
    "PIC": {
        "full_name": "Pictor",
        "area_sq_deg": 247,
        "description": "The Painter's Easel, a southern constellation",
    },
    "PSC": {
        "full_name": "Pisces",
        "area_sq_deg": 889,
        "description": "The Fish, a zodiac constellation",
    },
    "PSA": {
        "full_name": "Piscis Austrinus",
        "area_sq_deg": 245,
        "description": "The Southern Fish, featuring bright star Fomalhaut",
    },
    "PUP": {
        "full_name": "Puppis",
        "area_sq_deg": 673,
        "description": "The Stern, a large southern constellation",
    },
    "PYX": {
        "full_name": "Pyxis",
        "area_sq_deg": 221,
        "description": "The Compass, a small southern constellation",
    },
    "RET": {
        "full_name": "Reticulum",
        "area_sq_deg": 114,
        "description": "The Reticle, a small southern constellation",
    },
    "SGE": {
        "full_name": "Sagitta",
        "area_sq_deg": 80,
        "description": "The Arrow, third smallest constellation",
    },
    "SGR": {
        "full_name": "Sagittarius",
        "area_sq_deg": 867,
        "description": "The Archer, points toward galactic center",
    },
    "SCO": {
        "full_name": "Scorpius",
        "area_sq_deg": 497,
        "description": "The Scorpion, featuring red supergiant Antares",
    },
    "SCL": {
        "full_name": "Sculptor",
        "area_sq_deg": 475,
        "description": "The Sculptor, a faint southern constellation",
    },
    "SCT": {
        "full_name": "Scutum",
        "area_sq_deg": 109,
        "description": "The Shield, fifth smallest constellation",
    },
    "SER": {
        "full_name": "Serpens",
        "area_sq_deg": 637,
        "description": "The Serpent, only constellation split into two parts",
    },
    "SEX": {
        "full_name": "Sextans",
        "area_sq_deg": 314,
        "description": "The Sextant, a faint equatorial constellation",
    },
    "TAU": {
        "full_name": "Taurus",
        "area_sq_deg": 797,
        "description": "The Bull, featuring the Pleiades star cluster",
    },
    "TEL": {
        "full_name": "Telescopium",
        "area_sq_deg": 252,
        "description": "The Telescope, a small southern constellation",
    },
    "TRI": {
        "full_name": "Triangulum",
        "area_sq_deg": 132,
        "description": "The Triangle, home to the Triangulum Galaxy",
    },
    "TRA": {
        "full_name": "Triangulum Australe",
        "area_sq_deg": 110,
        "description": "The Southern Triangle, a small but bright constellation",
    },
    "TUC": {
        "full_name": "Tucana",
        "area_sq_deg": 295,
        "description": "The Toucan, home to Small Magellanic Cloud",
    },
    "UMA": {
        "full_name": "Ursa Major",
        "area_sq_deg": 1280,
        "description": "The Great Bear, third largest constellation",
    },
    "UMI": {
        "full_name": "Ursa Minor",
        "area_sq_deg": 256,
        "description": "The Little Bear, contains Polaris the North Star",
    },
    "VEL": {
        "full_name": "Vela",
        "area_sq_deg": 500,
        "description": "The Sails, a southern constellation",
    },
    "VIR": {
        "full_name": "Virgo",
        "area_sq_deg": 1294,
        "description": "The Virgin, second largest constellation",
    },
    "VOL": {
        "full_name": "Volans",
        "area_sq_deg": 141,
        "description": "The Flying Fish, a southern constellation",
    },
    "VUL": {
        "full_name": "Vulpecula",
        "area_sq_deg": 268,
        "description": "The Fox, home to the Dumbbell Nebula",
    },
}
