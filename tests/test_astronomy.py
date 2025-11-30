"""Tests for star catalog and astronomy module."""

import pandas as pd
import pytest

from drinking_galaxies.astronomy import StarCatalog


@pytest.fixture
def star_catalog():
    """Create star catalog instance for testing."""
    return StarCatalog()


def test_star_catalog_initialization(star_catalog):
    """Test that StarCatalog initializes correctly."""
    assert star_catalog.cache_dir.exists(), "Cache directory should be created"
    assert star_catalog._catalog is None, "Catalog should not be loaded initially"


def test_load_catalog(star_catalog):
    """Test loading star catalog."""
    catalog = star_catalog.load_catalog()

    assert isinstance(catalog, pd.DataFrame), "Should return DataFrame"
    assert len(catalog) > 0, "Catalog should contain stars"
    assert "ra" in catalog.columns, "Should have RA column"
    assert "dec" in catalog.columns, "Should have DEC column"
    assert "magnitude" in catalog.columns, "Should have magnitude column"


def test_get_all_bright_stars(star_catalog):
    """Test getting bright stars."""
    bright_stars = star_catalog.get_all_bright_stars(max_magnitude=3.0)

    assert isinstance(bright_stars, pd.DataFrame), "Should return DataFrame"
    assert len(bright_stars) > 0, "Should find some bright stars"
    assert (
        bright_stars["magnitude"] <= 3.0
    ).all(), "All stars should be brighter than magnitude 3.0"


def test_get_stars_in_region(star_catalog):
    """Test getting stars in a sky region."""
    # Orion region (approximately)
    stars = star_catalog.get_stars_in_region(
        ra_center=83.8, dec_center=-5.4, radius_deg=15.0, max_magnitude=6.0
    )

    assert isinstance(stars, pd.DataFrame), "Should return DataFrame"
    assert len(stars) > 0, "Should find stars in Orion region"


def test_convert_to_stereographic(star_catalog):
    """Test stereographic projection conversion."""
    # Get some test stars
    catalog = star_catalog.load_catalog()
    test_stars = catalog.head(10)

    # Convert to stereographic projection
    positions = star_catalog.convert_to_stereographic(
        test_stars, ra_center=0.0, dec_center=0.0
    )

    assert positions.shape == (10, 2), "Should return N x 2 array"
    assert positions.dtype == float, "Should be float array"


def test_search_sky_regions(star_catalog):
    """Test searching random sky regions."""
    regions = star_catalog.search_sky_regions(num_samples=5, max_magnitude=6.0)

    assert isinstance(regions, list), "Should return list"
    assert len(regions) > 0, "Should find some regions"

    for region in regions:
        assert "ra" in region, "Region should have RA"
        assert "dec" in region, "Region should have DEC"
        assert "stars" in region, "Region should have stars DataFrame"
        assert isinstance(region["stars"], pd.DataFrame), "Stars should be DataFrame"
