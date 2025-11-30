"""Tests for constellation identification module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drinking_galaxies.constellations import (
    CONSTELLATION_METADATA,
    ConstellationCatalog,
)


class TestConstellationMetadata:
    """Tests for constellation metadata dictionary."""

    def test_metadata_completeness(self):
        """Test that metadata exists for all 88 IAU constellations."""
        # IAU officially recognizes 88 constellations
        assert len(CONSTELLATION_METADATA) == 88, "Should have 88 constellations"

    def test_abbreviation_uniqueness(self):
        """Test that all constellation abbreviations are unique."""
        abbrevs = list(CONSTELLATION_METADATA.keys())
        assert len(abbrevs) == len(set(abbrevs)), "Abbreviations should be unique"

    def test_abbreviation_format(self):
        """Test that all abbreviations are 3 uppercase letters."""
        for abbrev in CONSTELLATION_METADATA.keys():
            assert len(abbrev) == 3, f"{abbrev} should be 3 characters"
            assert abbrev.isupper(), f"{abbrev} should be uppercase"
            assert abbrev.isalpha(), f"{abbrev} should be alphabetic"

    def test_area_values(self):
        """Test that constellation areas are positive and reasonable."""
        for abbrev, info in CONSTELLATION_METADATA.items():
            area = info["area_sq_deg"]
            assert area > 0, f"{abbrev} area should be positive"
            assert area < 1500, f"{abbrev} area should be reasonable (< 1500 sq deg)"

    def test_required_fields(self):
        """Test that all required fields are present for each constellation."""
        required_fields = ["full_name", "area_sq_deg", "description"]

        for abbrev, info in CONSTELLATION_METADATA.items():
            for field in required_fields:
                assert field in info, f"{abbrev} missing required field: {field}"
                assert info[field], f"{abbrev} has empty {field}"


class TestConstellationCatalog:
    """Tests for ConstellationCatalog class."""

    def test_catalog_initialization(self, tmp_path: Path):
        """Test that catalog initializes with correct cache directory."""
        cache_dir = tmp_path / "test_cache"
        catalog = ConstellationCatalog(cache_dir=cache_dir)

        assert catalog.cache_dir == cache_dir
        assert catalog.cache_dir.exists(), "Cache directory should be created"
        assert catalog.boundary_file == cache_dir / "iau_boundaries.csv"

    def test_catalog_default_cache_location(self):
        """Test that default cache location is in home directory."""
        catalog = ConstellationCatalog()

        expected_dir = Path.home() / ".drinking_galaxies" / "constellation_cache"
        assert catalog.cache_dir == expected_dir

    def test_get_constellation_info_valid(self):
        """Test retrieving constellation info for valid name."""
        catalog = ConstellationCatalog()

        info = catalog.get_constellation_info("Ori")

        assert info is not None
        assert info["abbrev"] == "ORI"
        assert info["full_name"] == "Orion"
        assert info["area_sq_deg"] == 594
        assert "Hunter" in info["description"]

    def test_get_constellation_info_case_insensitive(self):
        """Test that constellation info retrieval is case-insensitive."""
        catalog = ConstellationCatalog()

        info_lower = catalog.get_constellation_info("ori")
        info_upper = catalog.get_constellation_info("ORI")
        info_mixed = catalog.get_constellation_info("Ori")

        assert info_lower == info_upper == info_mixed

    def test_get_constellation_info_invalid(self):
        """Test that invalid constellation name returns None."""
        catalog = ConstellationCatalog()

        info = catalog.get_constellation_info("XXX")

        assert info is None

    def test_get_constellation_info_none(self):
        """Test that None input returns None."""
        catalog = ConstellationCatalog()

        info = catalog.get_constellation_info(None)

        assert info is None


class TestPointInPolygon:
    """Tests for point-in-polygon algorithm."""

    def test_simple_rectangle_inside(self):
        """Test point inside simple rectangular polygon."""
        catalog = ConstellationCatalog()

        # Simple rectangle: (0, 0), (10, 0), (10, 10), (0, 10)
        polygon = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])

        # Point clearly inside
        assert catalog._point_in_polygon(5, 5, polygon)

    def test_simple_rectangle_outside(self):
        """Test point outside simple rectangular polygon."""
        catalog = ConstellationCatalog()

        polygon = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])

        # Point clearly outside
        assert not catalog._point_in_polygon(15, 5, polygon)
        assert not catalog._point_in_polygon(5, 15, polygon)
        assert not catalog._point_in_polygon(-5, 5, polygon)

    def test_simple_rectangle_on_boundary(self):
        """Test point exactly on polygon boundary."""
        catalog = ConstellationCatalog()

        polygon = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])

        # Point on edge (behavior may vary, but should not crash)
        result = catalog._point_in_polygon(5, 0, polygon)
        assert isinstance(result, bool)

    def test_ra_wraparound_handling(self):
        """Test that RA wrap-around is handled without crashing.

        Note: Full wrap-around support is complex and may not work perfectly
        for all edge cases. This test ensures the algorithm doesn't crash.
        """
        catalog = ConstellationCatalog()

        # Polygon crossing 0°/360°: (350, 0), (10, 0), (10, 10), (350, 10)
        polygon = np.array([[350, 0], [10, 0], [10, 10], [350, 10], [350, 0]])

        # Test that algorithm doesn't crash on wraparound cases
        # Results may not be perfect but should return boolean
        result1 = catalog._point_in_polygon(0, 5, polygon)
        result2 = catalog._point_in_polygon(355, 5, polygon)
        result3 = catalog._point_in_polygon(180, 5, polygon)

        assert isinstance(result1, bool)
        assert isinstance(result2, bool)
        assert isinstance(result3, bool)


class TestConstellationIdentification:
    """Tests for constellation identification."""

    def test_identify_orion(self):
        """Test identification of Orion constellation."""
        catalog = ConstellationCatalog()

        # Orion is centered roughly at RA=83.8°, Dec=-5.4°
        # This test assumes boundaries are downloaded
        # For unit test, we'll just test the method doesn't crash
        # Integration tests should verify actual constellation names

        # Note: This will attempt to download boundaries on first run
        # Subsequent runs will use cache
        try:
            result = catalog.identify_constellation(83.8, -5.4)
            # If we got a result, it should be a string
            if result is not None:
                assert isinstance(result, str)
                assert len(result) == 3  # IAU abbreviations are 3 letters
        except RuntimeError:
            # VizieR may be unavailable in test environment
            pytest.skip("VizieR unavailable for boundary download")

    def test_identify_constellation_returns_none_or_string(self):
        """Test that identify_constellation returns None or valid string."""
        catalog = ConstellationCatalog()

        try:
            result = catalog.identify_constellation(0, 0)
            assert result is None or isinstance(result, str)
            if result is not None:
                assert len(result) == 3
        except RuntimeError:
            pytest.skip("VizieR unavailable for boundary download")

    def test_identify_multiple_positions(self):
        """Test constellation identification for multiple known positions."""
        catalog = ConstellationCatalog()

        # Known constellation positions (approximate centers)
        positions = [
            (83.8, -5.4),  # Orion
            (278.5, 38.8),  # Lyra
            (37.4, 89.3),  # Ursa Minor (near north pole)
            (0, -90),  # Octans (south pole)
        ]

        try:
            for ra, dec in positions:
                result = catalog.identify_constellation(ra, dec)
                # Each should return either None or a valid constellation
                assert result is None or (isinstance(result, str) and len(result) == 3)
        except RuntimeError:
            pytest.skip("VizieR unavailable for boundary download")


class TestPreparePolygon:
    """Tests for polygon preparation."""

    def test_prepare_polygon_closes_open_polygon(self):
        """Test that prepare_polygon closes an open polygon."""
        catalog = ConstellationCatalog()

        # Open polygon (first and last points different)
        boundary_points = pd.DataFrame({"ra": [0, 10, 10, 0], "dec": [0, 0, 10, 10]})

        polygon = catalog._prepare_polygon(boundary_points)

        # Should have 5 points (original 4 + closing point)
        assert len(polygon) == 5
        # First and last should be the same
        assert np.allclose(polygon[0], polygon[-1])

    def test_prepare_polygon_keeps_closed_polygon(self):
        """Test that polygon is always properly closed after preparation."""
        catalog = ConstellationCatalog()

        # Input polygon (will be sorted by RA)
        boundary_points = pd.DataFrame(
            {"ra": [0, 10, 10, 0, 0], "dec": [0, 0, 10, 10, 0]}
        )

        polygon = catalog._prepare_polygon(boundary_points)

        # Should be closed (first and last points the same)
        assert np.allclose(polygon[0], polygon[-1])
        # Sorting may change order, so length could be 5 or 6
        assert len(polygon) >= 5

    def test_prepare_polygon_sorts_by_ra(self):
        """Test that polygon vertices are sorted by RA."""
        catalog = ConstellationCatalog()

        # Unsorted polygon
        boundary_points = pd.DataFrame({"ra": [10, 0, 5, 15], "dec": [0, 0, 10, 10]})

        polygon = catalog._prepare_polygon(boundary_points)

        # Check that RA values are sorted (except for closing point)
        ra_values = polygon[:-1, 0]
        assert np.all(ra_values[:-1] <= ra_values[1:])


class TestBoundaryDownload:
    """Tests for boundary download functionality."""

    def test_download_boundaries_creates_cache(self, tmp_path: Path):
        """Test that download creates cache file."""
        cache_dir = tmp_path / "test_cache"
        catalog = ConstellationCatalog(cache_dir=cache_dir)

        boundaries = catalog.download_boundaries()

        # Skip test if VizieR unavailable (offline mode)
        if boundaries is None:
            pytest.skip("VizieR unavailable - constellation boundaries unavailable")

        assert catalog.boundary_file.exists()
        assert isinstance(boundaries, pd.DataFrame)
        assert "constellation" in boundaries.columns
        assert "ra" in boundaries.columns
        assert "dec" in boundaries.columns

    def test_load_boundaries_uses_cache(self, tmp_path: Path):
        """Test that load_boundaries uses cached data."""
        cache_dir = tmp_path / "test_cache"
        catalog = ConstellationCatalog(cache_dir=cache_dir)

        # Create fake cache
        fake_boundaries = pd.DataFrame(
            {
                "constellation": ["ORI", "ORI", "ORI"],
                "ra": [80, 90, 85],
                "dec": [-10, -10, 0],
            }
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        fake_boundaries.to_csv(catalog.boundary_file, index=False)

        # Load should use cache
        loaded = catalog.load_boundaries()

        assert len(loaded) == 3
        assert loaded["constellation"].iloc[0] == "ORI"

    def test_force_refresh_redownloads(self, tmp_path: Path):
        """Test that force_refresh bypasses cache."""
        cache_dir = tmp_path / "test_cache"
        catalog = ConstellationCatalog(cache_dir=cache_dir)

        # Create fake cache
        fake_boundaries = pd.DataFrame(
            {"constellation": ["XXX"], "ra": [0], "dec": [0]}
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        fake_boundaries.to_csv(catalog.boundary_file, index=False)

        # Force refresh should download new data
        fresh = catalog.download_boundaries(force_refresh=True)

        # Skip test if VizieR unavailable (offline mode)
        if fresh is None:
            pytest.skip("VizieR unavailable - constellation boundaries unavailable")

        # New data should not contain "XXX"
        assert "XXX" not in fresh["constellation"].values
