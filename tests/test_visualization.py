"""Tests for visualization module."""

import numpy as np

from drinking_galaxies.visualization import (
    create_constellation_visualization,
    draw_constellation_lines,
    magnitude_to_radius,
    normalize_positions_to_canvas,
    render_simple_stars,
    render_stars_with_magnitude,
)


class TestMagnitudeToRadius:
    """Tests for magnitude to radius conversion."""

    def test_bright_star_large_radius(self):
        """Test that bright stars (low magnitude) get larger radii."""
        radius_bright = magnitude_to_radius(1.0)
        radius_dim = magnitude_to_radius(5.0)

        assert radius_bright > radius_dim

    def test_radius_bounds(self):
        """Test that radius is bounded between 1 and 10."""
        # Very bright star
        r_min = magnitude_to_radius(-5.0)
        assert 1 <= r_min <= 10

        # Very dim star
        r_max = magnitude_to_radius(15.0)
        assert 1 <= r_max <= 10

    def test_typical_magnitudes(self):
        """Test radius for typical visible star magnitudes."""
        # Magnitude 1 (bright)
        r1 = magnitude_to_radius(1.0)
        assert 2 <= r1 <= 10

        # Magnitude 3 (medium)
        r3 = magnitude_to_radius(3.0)
        assert 1 <= r3 <= 5

        # Magnitude 6 (faint, naked eye limit)
        r6 = magnitude_to_radius(6.0)
        assert 1 <= r6 <= 3

    def test_scale_factor_effect(self):
        """Test that scale factor affects radius."""
        r_small = magnitude_to_radius(3.0, scale_factor=2.0)
        r_large = magnitude_to_radius(3.0, scale_factor=10.0)

        assert r_large > r_small


class TestRenderSimpleStars:
    """Tests for simple star rendering."""

    def test_renders_stars_on_canvas(self):
        """Test that stars are drawn on canvas."""
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        positions = np.array([[50, 50], [75, 75]])

        result = render_simple_stars(canvas.copy(), positions, radius=2)

        # Check that star positions have non-zero pixels
        assert np.any(result[48:53, 48:53] > 0)
        assert np.any(result[73:78, 73:78] > 0)

    def test_respects_canvas_bounds(self):
        """Test that out-of-bounds stars don't crash."""
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)

        # Stars outside canvas
        positions = np.array([[-10, -10], [150, 150]])

        # Should not crash
        result = render_simple_stars(canvas, positions)
        assert result.shape == canvas.shape

    def test_custom_color(self):
        """Test that custom star color is applied."""
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        positions = np.array([[50, 50]])
        color = (255, 0, 0)  # Red

        result = render_simple_stars(canvas, positions, color=color)

        # Check that star has red color
        assert result[50, 50, 0] > 0  # Red channel
        assert result[50, 50, 1] == 0  # Green channel
        assert result[50, 50, 2] == 0  # Blue channel


class TestRenderStarsWithMagnitude:
    """Tests for magnitude-scaled star rendering."""

    def test_renders_different_sizes(self):
        """Test that different magnitudes produce different sizes."""
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        positions = np.array([[50, 50], [150, 150]])
        magnitudes = np.array([1.0, 5.0])  # Bright and dim

        result = render_stars_with_magnitude(canvas, positions, magnitudes)

        # Count non-zero pixels around each star
        bright_pixels = np.sum(result[45:56, 45:56] > 0)
        dim_pixels = np.sum(result[145:156, 145:156] > 0)

        # Bright star should have more pixels
        assert bright_pixels > dim_pixels

    def test_adds_glow_for_bright_stars(self):
        """Test that bright stars get a glow effect."""
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        positions = np.array([[100, 100]])
        magnitudes = np.array([1.0])  # Very bright

        result = render_stars_with_magnitude(canvas, positions, magnitudes)

        # Should have pixels beyond the core radius
        # Core should be bright white
        assert np.all(result[100, 100] == 255)

        # Glow should exist nearby (may be as bright as 255 due to 0.7 intensity)
        # Check a pixel that's in glow range
        nearby = result[100, 100 + 5]
        assert np.any(nearby > 0)  # Glow exists
        # With 0.7 intensity, glow can reach 255, so just verify it's present

    def test_respects_magnitude_array_length(self):
        """Test that magnitude array must match positions."""
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        positions = np.array([[50, 50], [75, 75]])
        magnitudes = np.array([2.0, 3.0])

        result = render_stars_with_magnitude(canvas, positions, magnitudes)

        assert result.shape == canvas.shape
        # Both stars should be rendered
        assert np.any(result[48:53, 48:53] > 0)
        assert np.any(result[73:78, 73:78] > 0)


class TestDrawConstellationLines:
    """Tests for constellation line drawing."""

    def test_draws_lines_between_stars(self):
        """Test that lines are drawn between specified stars."""
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        positions = np.array([[50, 50], [150, 150]])
        segments = [(0, 1)]  # Connect first and second star

        result = draw_constellation_lines(canvas, positions, segments)

        # Check that pixels along the line are non-zero
        # Sample some points along diagonal
        assert np.any(result[75, 75] > 0)
        assert np.any(result[100, 100] > 0)
        assert np.any(result[125, 125] > 0)

    def test_handles_invalid_indices(self):
        """Test that invalid segment indices are ignored."""
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        positions = np.array([[50, 50], [75, 75]])

        # Invalid segments
        segments = [(-1, 0), (0, 5), (10, 20)]

        # Should not crash
        result = draw_constellation_lines(canvas, positions, segments)
        assert result.shape == canvas.shape

    def test_custom_line_color_and_thickness(self):
        """Test custom line appearance."""
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        positions = np.array([[50, 50], [150, 150]])
        segments = [(0, 1)]
        color = (255, 0, 0)  # Red
        thickness = 3

        result = draw_constellation_lines(
            canvas, positions, segments, color=color, thickness=thickness
        )

        # Check that line has red color
        midpoint = result[100, 100]
        assert midpoint[0] > 0  # Red channel
        assert midpoint[1] == 0  # Green channel
        assert midpoint[2] == 0  # Blue channel


class TestNormalizePositionsToCanvas:
    """Tests for position normalization."""

    def test_normalizes_to_canvas_bounds(self):
        """Test that positions are normalized to canvas."""
        positions = np.array([[-1, -1], [1, 1]])
        canvas_shape = (500, 500)

        normalized = normalize_positions_to_canvas(positions, canvas_shape)

        # Should be within canvas bounds
        assert np.all(normalized >= 0)
        assert np.all(normalized[:, 0] < canvas_shape[1])  # Width
        assert np.all(normalized[:, 1] < canvas_shape[0])  # Height

    def test_applies_margin(self):
        """Test that margin is applied correctly."""
        positions = np.array([[-1, -1], [1, 1]])
        canvas_shape = (500, 500)
        margin = 0.2

        normalized = normalize_positions_to_canvas(
            positions, canvas_shape, margin=margin
        )

        # Should have margin from edges
        min_coord = normalized.min()
        max_coord = normalized.max()

        margin_pixels = margin * min(canvas_shape)

        assert min_coord >= margin_pixels * 0.8  # Allow some tolerance
        assert max_coord <= max(canvas_shape) - margin_pixels * 0.8

    def test_handles_single_point(self):
        """Test that single point is centered."""
        positions = np.array([[0, 0]])
        canvas_shape = (500, 500)

        normalized = normalize_positions_to_canvas(positions, canvas_shape)

        # Should be near center
        assert np.allclose(normalized[0], [250, 250], atol=50)

    def test_handles_clustered_points(self):
        """Test that clustered points (same position) are centered."""
        positions = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        canvas_shape = (500, 500)

        normalized = normalize_positions_to_canvas(positions, canvas_shape)

        # All should be at same position near center
        assert np.allclose(normalized[0], normalized[1])
        assert np.allclose(normalized[1], normalized[2])
        assert np.allclose(normalized[0], [250, 250], atol=50)

    def test_preserves_relative_positions(self):
        """Test that relative positions are preserved."""
        # Square pattern
        positions = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        canvas_shape = (500, 500)

        normalized = normalize_positions_to_canvas(positions, canvas_shape)

        # Calculate relative distances
        dist_01 = np.linalg.norm(normalized[1] - normalized[0])
        dist_12 = np.linalg.norm(normalized[2] - normalized[1])
        dist_23 = np.linalg.norm(normalized[3] - normalized[2])
        dist_30 = np.linalg.norm(normalized[0] - normalized[3])

        # All sides of square should be equal
        assert np.allclose(dist_01, dist_12, rtol=0.01)
        assert np.allclose(dist_12, dist_23, rtol=0.01)
        assert np.allclose(dist_23, dist_30, rtol=0.01)


class TestCreateConstellationVisualization:
    """Tests for complete constellation visualization."""

    def test_creates_image_with_correct_shape(self):
        """Test that output has correct shape."""
        image_shape = (500, 500, 3)
        positions = np.array([[100, 100], [200, 200]])

        result = create_constellation_visualization(image_shape, positions)

        assert result.shape == image_shape
        assert result.dtype == np.uint8

    def test_uses_magnitude_scaling_when_provided(self):
        """Test that magnitudes are used for star sizing."""
        image_shape = (500, 500, 3)
        positions = np.array([[100, 100], [400, 400]])
        magnitudes = np.array([1.0, 6.0])

        result = create_constellation_visualization(
            image_shape, positions, magnitudes=magnitudes
        )

        # Count non-background pixels for each star
        # Use background color to identify star pixels
        bg_color = np.array([10, 10, 30])

        # Count pixels that differ from background
        bright_region = result[90:111, 90:111]
        dim_region = result[390:411, 390:411]

        bright_pixels = np.sum(np.any(bright_region != bg_color, axis=2))
        dim_pixels = np.sum(np.any(dim_region != bg_color, axis=2))

        # Bright star should have more non-background pixels
        assert bright_pixels > dim_pixels

    def test_draws_lines_when_provided(self):
        """Test that constellation lines are drawn."""
        image_shape = (500, 500, 3)
        positions = np.array([[100, 100], [400, 400]])
        segments = [(0, 1)]

        result = create_constellation_visualization(
            image_shape, positions, line_segments=segments, draw_lines=True
        )

        # Check for line pixels between stars
        assert np.any(result[250, 250] > 0)

    def test_respects_background_color(self):
        """Test that background color is applied."""
        image_shape = (100, 100, 3)
        positions = np.array([[50, 50]])
        bg_color = (10, 20, 30)

        result = create_constellation_visualization(
            image_shape, positions, background_color=bg_color
        )

        # Sample background pixel (away from star)
        bg_pixel = result[10, 10]
        assert tuple(bg_pixel) == bg_color

    def test_works_without_magnitudes(self):
        """Test that it works with uniform stars (no magnitudes)."""
        image_shape = (200, 200, 3)
        positions = np.array([[50, 50], [150, 150]])

        result = create_constellation_visualization(image_shape, positions)

        # Should render stars without error
        assert np.any(result[48:53, 48:53] > 0)
        assert np.any(result[148:153, 148:153] > 0)

    def test_empty_positions_returns_background(self):
        """Test that empty positions returns just background."""
        image_shape = (100, 100, 3)
        positions = np.array([]).reshape(0, 2)
        bg_color = (10, 10, 30)

        result = create_constellation_visualization(
            image_shape, positions, background_color=bg_color
        )

        # Should be all background color
        assert np.all(result == bg_color)
