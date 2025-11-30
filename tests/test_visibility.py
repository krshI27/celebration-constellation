"""Tests for visibility calculator module."""

import pytest

from drinking_galaxies.visibility import (
    calculate_best_viewing_months,
    calculate_visibility_range,
    get_example_cities,
    get_viewing_regions,
)


class TestVisibilityRange:
    """Tests for visibility range calculations."""

    def test_equatorial_constellation_globally_visible(self):
        """Test that equatorial constellations are visible globally."""
        # Orion: RA=83.8°, Dec=22.4°
        visibility = calculate_visibility_range(ra=83.8, dec=22.4)

        # With min_altitude=10°, should be visible from most latitudes
        # min_lat = 22.4 - (90 - 10) = -57.6
        # max_lat = 22.4 + (90 - 10) = 90 (clamped)
        assert visibility["min_latitude"] < -55.0
        assert visibility["max_latitude"] == 90.0
        assert visibility["optimal_latitude"] == pytest.approx(22.4, abs=0.1)

    def test_northern_constellation_visibility(self):
        """Test visibility of northern constellation."""
        # Ursa Major: Dec ≈ 55°
        visibility = calculate_visibility_range(ra=165.0, dec=55.0)

        # Should be visible from northern latitudes
        assert visibility["min_latitude"] < 0  # Some southern visibility
        assert visibility["max_latitude"] == 90.0  # Visible to North Pole
        assert visibility["optimal_latitude"] == 55.0

    def test_southern_constellation_visibility(self):
        """Test visibility of southern constellation."""
        # Crux (Southern Cross): Dec ≈ -60°
        visibility = calculate_visibility_range(ra=187.0, dec=-60.0)

        # Should be visible from southern latitudes
        assert visibility["min_latitude"] == -90.0  # Visible to South Pole
        assert visibility["max_latitude"] > 0  # Some northern visibility
        assert visibility["optimal_latitude"] == -60.0

    def test_circumpolar_detection_north(self):
        """Test circumpolar detection for northern constellation."""
        # Polaris: Dec ≈ 89°
        # Circumpolar above: 90° - 89° = 1°N
        visibility = calculate_visibility_range(ra=37.95, dec=89.0)

        # Should be circumpolar above ~1°N
        assert visibility["circumpolar_above"] == pytest.approx(1.0, abs=0.1)
        assert 0 < visibility["circumpolar_above"] < 90.0

    def test_never_visible_regions(self):
        """Test detection of never-visible regions."""
        # Southern constellation at dec=-50°
        # never_rises_below = -50 - (90 - 10) = -50 - 80 = -130 → clamped to -90
        # For a northern constellation to test this better:
        # Dec=60°: never_rises_below = 60 - 80 = -20°
        visibility = calculate_visibility_range(ra=180.0, dec=60.0)

        # Should have latitude below which never visible
        assert -90.0 <= visibility["never_rises_below"] < visibility["optimal_latitude"]

    def test_min_altitude_parameter(self):
        """Test that min_altitude parameter affects visibility range."""
        # Same constellation with different min_altitude
        vis_10 = calculate_visibility_range(ra=90.0, dec=30.0, min_altitude=10.0)
        vis_30 = calculate_visibility_range(ra=90.0, dec=30.0, min_altitude=30.0)

        # Higher min_altitude should reduce visible range
        lat_range_10 = vis_10["max_latitude"] - vis_10["min_latitude"]
        lat_range_30 = vis_30["max_latitude"] - vis_30["min_latitude"]

        assert lat_range_30 < lat_range_10

    def test_declination_bounds(self):
        """Test that extreme declinations are handled correctly."""
        # North Pole
        vis_north = calculate_visibility_range(ra=0.0, dec=90.0)
        assert vis_north["optimal_latitude"] == 90.0

        # South Pole
        vis_south = calculate_visibility_range(ra=0.0, dec=-90.0)
        assert vis_south["optimal_latitude"] == -90.0

        # Equator
        vis_equator = calculate_visibility_range(ra=0.0, dec=0.0)
        assert vis_equator["optimal_latitude"] == 0.0


class TestBestViewingMonths:
    """Tests for best viewing months calculation."""

    def test_returns_four_months(self):
        """Test that function returns 4 months."""
        months = calculate_best_viewing_months(ra=90.0)

        assert len(months) == 4
        assert all(isinstance(m, str) for m in months)

    def test_orion_viewing_months(self):
        """Test viewing months for Orion (winter constellation)."""
        # Orion: RA ≈ 84° (best viewed in winter)
        months = calculate_best_viewing_months(ra=84.0)

        # Should include winter months (Northern Hemisphere)
        assert any(month in months for month in ["November", "December", "January"])

    def test_different_ra_different_months(self):
        """Test that different RAs give different viewing months."""
        months_spring = calculate_best_viewing_months(ra=0.0)
        months_fall = calculate_best_viewing_months(ra=180.0)

        # Should have some different months
        assert set(months_spring) != set(months_fall)

    def test_all_month_names_valid(self):
        """Test that returned month names are valid."""
        valid_months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        # Test for several RAs
        for ra in [0, 90, 180, 270]:
            months = calculate_best_viewing_months(ra=ra)
            assert all(month in valid_months for month in months)


class TestViewingRegions:
    """Tests for geographic region mapping."""

    def test_arctic_region(self):
        """Test detection of Arctic region."""
        regions = get_viewing_regions(min_lat=70.0, max_lat=90.0)

        assert "Arctic" in regions or any("Arctic" in r for r in regions)

    def test_temperate_regions(self):
        """Test detection of temperate zones."""
        # Northern temperate
        regions_north = get_viewing_regions(min_lat=30.0, max_lat=60.0)
        assert any("Temperate" in r for r in regions_north)

        # Southern temperate
        regions_south = get_viewing_regions(min_lat=-60.0, max_lat=-30.0)
        assert any("Temperate" in r for r in regions_south)

    def test_tropics_region(self):
        """Test detection of tropics."""
        regions = get_viewing_regions(min_lat=-10.0, max_lat=10.0)

        assert any("Tropic" in r for r in regions)

    def test_global_visibility(self):
        """Test handling of global visibility."""
        regions = get_viewing_regions(min_lat=-90.0, max_lat=90.0)

        # Should include all major zones or indicate global
        assert len(regions) >= 3 or "Global" in " ".join(regions)

    def test_partial_overlap(self):
        """Test regions with partial zone overlap."""
        regions = get_viewing_regions(min_lat=10.0, max_lat=40.0)

        # Should detect overlap between tropics and temperate
        assert len(regions) >= 1


class TestExampleCities:
    """Tests for example city selection."""

    def test_returns_cities_in_range(self):
        """Test that returned cities are within latitude range."""
        cities = get_example_cities(min_lat=30.0, max_lat=60.0)

        for city in cities:
            assert 30.0 <= city["lat"] <= 60.0

    def test_northern_hemisphere_cities(self):
        """Test selection of northern hemisphere cities."""
        cities = get_example_cities(min_lat=40.0, max_lat=60.0)

        assert len(cities) > 0
        # Should include some northern cities
        city_names = [c["name"] for c in cities]
        assert any(
            name in " ".join(city_names)
            for name in ["London", "Paris", "Moscow", "Stockholm"]
        )

    def test_southern_hemisphere_cities(self):
        """Test selection of southern hemisphere cities."""
        cities = get_example_cities(min_lat=-50.0, max_lat=-20.0)

        assert len(cities) > 0
        # Should include some southern cities
        city_names = [c["name"] for c in cities]
        assert any(
            name in " ".join(city_names)
            for name in ["Sydney", "Buenos Aires", "Johannesburg"]
        )

    def test_tropical_cities(self):
        """Test selection of tropical cities."""
        cities = get_example_cities(min_lat=-10.0, max_lat=10.0)

        assert len(cities) > 0
        # Should include tropical cities
        city_names = [c["name"] for c in cities]
        assert any(
            name in " ".join(city_names) for name in ["Singapore", "Nairobi", "Jakarta"]
        )

    def test_returns_up_to_five_cities(self):
        """Test that function returns at most 5 cities."""
        # Wide range should have many candidates
        cities = get_example_cities(min_lat=-60.0, max_lat=60.0)

        assert 1 <= len(cities) <= 5

    def test_city_has_required_fields(self):
        """Test that city dictionaries have required fields."""
        cities = get_example_cities(min_lat=0.0, max_lat=50.0)

        assert len(cities) > 0
        for city in cities:
            assert "name" in city
            assert "lat" in city
            assert "lon" in city
            assert isinstance(city["name"], str)
            assert isinstance(city["lat"], (int, float))
            assert isinstance(city["lon"], (int, float))

    def test_no_cities_in_range(self):
        """Test handling when no cities match range."""
        # Very narrow range unlikely to have cities
        cities = get_example_cities(min_lat=89.0, max_lat=90.0)

        # Should return empty list, not crash
        assert isinstance(cities, list)

    def test_cities_distributed_across_range(self):
        """Test that cities are distributed across latitude range."""
        cities = get_example_cities(min_lat=-40.0, max_lat=40.0)

        if len(cities) >= 3:
            # Get latitudes
            lats = sorted([c["lat"] for c in cities])

            # Check that cities span the range reasonably
            lat_span = lats[-1] - lats[0]
            total_span = 40.0 - (-40.0)

            # Cities should cover at least 30% of the range
            assert lat_span >= 0.3 * total_span


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_declination(self):
        """Test visibility for object at celestial equator."""
        visibility = calculate_visibility_range(ra=0.0, dec=0.0)

        # Should be visible from both hemispheres
        assert visibility["min_latitude"] < 0
        assert visibility["max_latitude"] > 0
        assert visibility["optimal_latitude"] == 0.0

    def test_ra_wraparound(self):
        """Test that RA values near 0°/360° are handled correctly."""
        vis_0 = calculate_visibility_range(ra=0.0, dec=45.0)
        vis_360 = calculate_visibility_range(ra=360.0, dec=45.0)

        # Should give similar results (same declination)
        assert vis_0["min_latitude"] == pytest.approx(vis_360["min_latitude"], abs=0.1)
        assert vis_0["max_latitude"] == pytest.approx(vis_360["max_latitude"], abs=0.1)

    def test_invalid_declination_clamped(self):
        """Test that invalid declinations are clamped to valid range."""
        # Dec > 90 should be clamped to 90
        vis_high = calculate_visibility_range(ra=0.0, dec=100.0)
        assert -90.0 <= vis_high["optimal_latitude"] <= 90.0

        # Dec < -90 should be clamped to -90
        vis_low = calculate_visibility_range(ra=0.0, dec=-100.0)
        assert -90.0 <= vis_low["optimal_latitude"] <= 90.0

    def test_extreme_min_altitude(self):
        """Test extreme min_altitude values."""
        # Very low altitude (min_altitude=0)
        # For dec=30: range = 30 + 90 - (30 - 90) = 120 - (-60) = 180
        # But clamped to [-90, 90], so range = 90 - (-60) = 150
        vis_low = calculate_visibility_range(ra=90.0, dec=30.0, min_altitude=0.0)
        assert vis_low["max_latitude"] - vis_low["min_latitude"] >= 140

        # Very high altitude (near zenith only)
        # For dec=30, min_alt=80: range = 30 + 10 - (30 - 10) = 40 - 20 = 20
        vis_high = calculate_visibility_range(ra=90.0, dec=30.0, min_altitude=80.0)
        assert vis_high["max_latitude"] - vis_high["min_latitude"] < 30
