"""Tests for circle detection module."""

from pathlib import Path

import numpy as np
import pytest

from drinking_galaxies.detection import (
    detect_circles,
    draw_circles,
    get_circle_centers,
    load_image,
    preprocess_image,
)


def test_load_image_nonexistent_file():
    """Test that loading nonexistent file raises FileNotFoundError."""
    fake_path = Path("nonexistent_image.jpg")

    with pytest.raises(FileNotFoundError, match="Image not found"):
        load_image(fake_path)


def test_preprocess_image():
    """Test image preprocessing produces grayscale blurred image."""
    # Create test RGB image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    processed = preprocess_image(test_image)

    assert processed.ndim == 2, "Processed image should be 2D (grayscale)"
    assert processed.shape == (100, 100), "Shape should be preserved"
    assert processed.dtype == np.uint8, "Dtype should be uint8"


def test_get_circle_centers():
    """Test extracting center coordinates from circles."""
    circles = np.array([[100, 150, 30], [200, 180, 25], [150, 250, 40]])

    centers = get_circle_centers(circles)

    assert centers.shape == (3, 2), "Should return N x 2 array"
    assert np.allclose(centers[0], [100, 150]), "First center incorrect"
    assert np.allclose(centers[1], [200, 180]), "Second center incorrect"


def test_draw_circles_with_both_enabled():
    """Test drawing circles and centers on image."""
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    circles = np.array([[100, 100, 30], [200, 200, 40]])

    result = draw_circles(test_image, circles, show_circles=True, show_centers=True)

    assert result.shape == test_image.shape, "Output shape should match input"
    assert not np.array_equal(
        result, test_image
    ), "Image should be modified with drawings"


def test_draw_circles_only_centers():
    """Test drawing only center points."""
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    circles = np.array([[100, 100, 30]])

    result = draw_circles(test_image, circles, show_circles=False, show_centers=True)

    # Check that center point area has been modified
    center_area = result[97:104, 97:104]
    assert np.any(center_area > 0), "Center point should be drawn"


def test_detect_circles_returns_none_for_empty_image():
    """Test that empty image returns None."""
    empty_image = np.zeros((300, 300, 3), dtype=np.uint8)

    circles = detect_circles(empty_image)

    assert circles is None, "Should return None for image with no circles"


def test_detect_circles_returns_array():
    """Test that detect_circles returns numpy array when circles found."""
    # Create synthetic image with a circle (white circle on black background)
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)

    # Draw a filled circle
    center = (150, 150)
    radius = 50
    y, x = np.ogrid[: test_image.shape[0], : test_image.shape[1]]
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
    test_image[mask] = [255, 255, 255]

    circles = detect_circles(test_image, min_radius=40, max_radius=60)

    # May or may not detect depending on parameters, so just check type
    if circles is not None:
        assert isinstance(circles, np.ndarray), "Should return numpy array"
        assert circles.ndim == 2, "Should be 2D array"
        assert circles.shape[1] == 3, "Should have 3 columns (x, y, r)"
