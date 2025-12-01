"""Enhanced visualization for star constellations.

This module provides advanced rendering for constellation matches,
including magnitude-scaled stars and constellation stick figures.
"""

from typing import Optional

import cv2
import numpy as np


def magnitude_to_radius(magnitude: float, scale_factor: float = 8.0) -> int:
    """Convert star magnitude to circle radius for rendering.

    Brighter stars (lower magnitude) get larger radii.
    Uses inverse logarithmic scaling to match human perception.

    Args:
        magnitude: Visual magnitude (lower = brighter)
        scale_factor: Scaling factor for radius calculation (default: 8.0 for better visibility)

    Returns:
        Circle radius in pixels (minimum 2, maximum 15)

    Example:
        >>> magnitude_to_radius(1.0)  # Very bright star
        7
        >>> magnitude_to_radius(6.0)  # Faint star
        2
    """
    # Invert magnitude: brighter stars = lower magnitude = larger radius
    # Clamp magnitude to reasonable range (0-6 for visible stars)
    mag_clamped = max(0.0, min(6.0, magnitude))

    # Exponential scaling: radius = scale * e^(-mag/3)
    # This gives natural brightness perception
    radius = scale_factor * np.exp(-mag_clamped / 3.0)

    # Clamp to pixel range (2-15 for better visibility)
    return max(2, min(15, int(radius)))


def render_stars_with_magnitude(
    canvas: np.ndarray,
    star_positions: np.ndarray,
    magnitudes: np.ndarray,
    color: tuple[int, int, int] = (255, 255, 255),
    scale_factor: float = 12.0,
) -> np.ndarray:
    """Render stars with magnitude-scaled sizes.

    Args:
        canvas: Image to draw on (H, W, 3)
        star_positions: Star positions in image coordinates (N, 2)
        magnitudes: Visual magnitudes for each star (N,)
        color: RGB color for stars
        scale_factor: Radius scaling factor

    Returns:
        Canvas with stars drawn

    Example:
        >>> canvas = np.zeros((500, 500, 3), dtype=np.uint8)
        >>> positions = np.array([[100, 100], [200, 200]])
        >>> mags = np.array([1.0, 5.0])
        >>> result = render_stars_with_magnitude(canvas, positions, mags)
    """
    for (x, y), mag in zip(star_positions, magnitudes):
        radius = magnitude_to_radius(mag, scale_factor)

        # Convert to integer coordinates
        x_int, y_int = int(x), int(y)

        # Check bounds
        if 0 <= x_int < canvas.shape[1] and 0 <= y_int < canvas.shape[0]:
            # Draw filled circle
            cv2.circle(canvas, (x_int, y_int), radius, color, -1)

            # Add glow effect to all stars for better visibility
            glow_radius = radius + 3
            glow_color = tuple(int(c * 0.7) for c in color)
            cv2.circle(canvas, (x_int, y_int), glow_radius, glow_color, 2)

            # Extra bright glow for very bright stars (mag < 3)
            if mag < 3.0:
                outer_glow_radius = radius + 5
                outer_glow_color = tuple(int(c * 0.3) for c in color)
                cv2.circle(
                    canvas, (x_int, y_int), outer_glow_radius, outer_glow_color, 1
                )

    return canvas


def render_simple_stars(
    canvas: np.ndarray,
    star_positions: np.ndarray,
    radius: int = 2,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Render stars with uniform size (fallback for no magnitude data).

    Args:
        canvas: Image to draw on (H, W, 3)
        star_positions: Star positions in image coordinates (N, 2)
        radius: Uniform radius for all stars
        color: RGB color for stars

    Returns:
        Canvas with stars drawn
    """
    for x, y in star_positions:
        x_int, y_int = int(x), int(y)

        if 0 <= x_int < canvas.shape[1] and 0 <= y_int < canvas.shape[0]:
            cv2.circle(canvas, (x_int, y_int), radius, color, -1)

    return canvas


def draw_constellation_lines(
    canvas: np.ndarray,
    star_positions: np.ndarray,
    line_segments: list[tuple[int, int]],
    color: tuple[int, int, int] = (100, 150, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Draw constellation stick figure lines.

    Args:
        canvas: Image to draw on (H, W, 3)
        star_positions: Star positions in image coordinates (N, 2)
        line_segments: List of (start_idx, end_idx) pairs for lines
        color: RGB color for lines
        thickness: Line thickness in pixels

    Returns:
        Canvas with constellation lines drawn

    Example:
        >>> canvas = np.zeros((500, 500, 3), dtype=np.uint8)
        >>> positions = np.array([[100, 100], [200, 200], [150, 300]])
        >>> lines = [(0, 1), (1, 2)]  # Connect stars 0-1 and 1-2
        >>> result = draw_constellation_lines(canvas, positions, lines)
    """
    for start_idx, end_idx in line_segments:
        # Validate indices
        if start_idx < 0 or start_idx >= len(star_positions):
            continue
        if end_idx < 0 or end_idx >= len(star_positions):
            continue

        # Get positions
        start = star_positions[start_idx]
        end = star_positions[end_idx]

        # Convert to integer coordinates
        pt1 = (int(start[0]), int(start[1]))
        pt2 = (int(end[0]), int(end[1]))

        # Draw line
        cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

    return canvas


def create_constellation_visualization(
    image_shape: tuple[int, int, int],
    star_positions: np.ndarray,
    magnitudes: Optional[np.ndarray] = None,
    line_segments: Optional[list[tuple[int, int]]] = None,
    background_color: tuple[int, int, int] = (10, 10, 30),
    star_color: tuple[int, int, int] = (255, 255, 255),
    line_color: tuple[int, int, int] = (100, 150, 255),
    draw_lines: bool = True,
    star_scale_factor: float = 12.0,
) -> np.ndarray:
    """Create complete constellation visualization.

    Args:
        image_shape: Target image shape (H, W, 3)
        star_positions: Star positions in image coordinates (N, 2)
        magnitudes: Optional star magnitudes for size scaling (N,)
        line_segments: Optional constellation line segments
        background_color: RGB background color
        star_color: RGB color for stars
        line_color: RGB color for constellation lines
        draw_lines: Whether to draw constellation lines

    Returns:
        Rendered constellation image

    Example:
        >>> positions = np.array([[100, 100], [200, 200]])
        >>> mags = np.array([1.0, 3.0])
        >>> lines = [(0, 1)]
        >>> img = create_constellation_visualization(
        ...     (500, 500, 3), positions, mags, lines
        ... )
    """
    # Create canvas with background
    canvas = np.full(image_shape, background_color, dtype=np.uint8)

    # Draw constellation lines first (behind stars)
    if draw_lines and line_segments:
        canvas = draw_constellation_lines(
            canvas, star_positions, line_segments, line_color
        )

    # Draw stars
    if magnitudes is not None and len(magnitudes) == len(star_positions):
        canvas = render_stars_with_magnitude(
            canvas,
            star_positions,
            magnitudes,
            star_color,
            scale_factor=star_scale_factor,
        )
    else:
        canvas = render_simple_stars(canvas, star_positions, color=star_color)

    return canvas


def create_composite_overlay(
    original_image: np.ndarray,
    circle_positions: np.ndarray,
    star_positions: np.ndarray,
    magnitudes: Optional[np.ndarray] = None,
    circle_color: tuple[int, int, int] = (0, 255, 0),
    star_color: tuple[int, int, int] = (255, 255, 0),
    alpha: float = 0.9,
    star_scale_factor: float = 12.0,
) -> np.ndarray:
    """Create composite overlay showing both circles and matched stars.

    Args:
        original_image: Original RGB image with detected circles
        circle_positions: Circle center coordinates (N, 2)
        star_positions: Transformed star positions in image coordinates (M, 2)
        magnitudes: Optional star magnitudes for size scaling (M,)
        circle_color: RGB color for circle centers (default: green)
        star_color: RGB color for stars (default: bright yellow)
        alpha: Transparency for star overlay (0.0-1.0, default: 0.9 for high visibility)

    Returns:
        Composite image showing circles and stars together

    Example:
        >>> img = np.zeros((500, 500, 3), dtype=np.uint8)
        >>> circles = np.array([[100, 100], [200, 200]])
        >>> stars = np.array([[105, 105], [195, 195]])
        >>> mags = np.array([2.0, 4.0])
        >>> composite = create_composite_overlay(img, circles, stars, mags)
    """
    # Create a copy of the original image
    overlay = original_image.copy()

    # Create star layer
    star_layer = np.zeros_like(original_image)

    # Draw stars on separate layer with enhanced visibility
    if magnitudes is not None and len(magnitudes) == len(star_positions):
        star_layer = render_stars_with_magnitude(
            star_layer,
            star_positions,
            magnitudes,
            star_color,
            scale_factor=star_scale_factor,
        )
    else:
        star_layer = render_simple_stars(
            star_layer, star_positions, radius=6, color=star_color
        )

    # Blend star layer with original image (high alpha for visibility)
    overlay = cv2.addWeighted(overlay, 1.0, star_layer, alpha, 0)

    # Draw circle centers on top (small markers)
    for x, y in circle_positions:
        x_int, y_int = int(x), int(y)
        if 0 <= x_int < overlay.shape[1] and 0 <= y_int < overlay.shape[0]:
            # Draw crosshair for circle centers
            cv2.drawMarker(
                overlay,
                (x_int, y_int),
                circle_color,
                cv2.MARKER_CROSS,
                10,
                2,
                cv2.LINE_AA,
            )

    return overlay


def create_circles_on_stars(
    image_shape: tuple[int, int, int],
    star_positions: np.ndarray,
    circles: np.ndarray,
    magnitudes: Optional[np.ndarray] = None,
    background_color: tuple[int, int, int] = (20, 20, 50),
    star_color: tuple[int, int, int] = (255, 255, 200),
    circle_color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw detected circles on top of the star pattern visualization.

    This is the inverse of create_composite_overlay - it shows the clean star
    pattern with the actual detected circles overlaid to show the alignment.

    Args:
        image_shape: Shape of output image (H, W, 3)
        star_positions: Star positions in canvas coordinates (M, 2)
        circles: Detected circles as (x, y, radius) tuples (N, 3)
        magnitudes: Optional star magnitudes for size scaling (M,)
        background_color: RGB background color (default: dark blue)
        star_color: RGB color for stars (default: warm yellow)
        circle_color: RGB color for circle outlines (default: green)

    Returns:
        Image with star pattern and actual circle outlines

    Example:
        >>> shape = (500, 500, 3)
        >>> stars = np.array([[100, 100], [200, 200]])
        >>> circles = np.array([[105, 105, 20], [195, 195, 30]])  # (x, y, radius)
        >>> mags = np.array([2.0, 4.0])
        >>> result = create_circles_on_stars(shape, stars, circles, mags)
    """
    # Create star visualization
    canvas = create_constellation_visualization(
        image_shape,
        star_positions,
        magnitudes=magnitudes,
        line_segments=None,
        background_color=background_color,
        star_color=star_color,
        draw_lines=False,
    )

    # Draw actual detected circles on top (same as in original photo)
    for x, y, r in circles:
        x_int, y_int, r_int = int(x), int(y), int(r)
        if 0 <= x_int < canvas.shape[1] and 0 <= y_int < canvas.shape[0]:
            # Draw circle outline
            cv2.circle(canvas, (x_int, y_int), r_int, circle_color, 2)
            # Draw center point
            cv2.circle(canvas, (x_int, y_int), 3, circle_color, -1)

    return canvas


def normalize_positions_to_canvas(
    positions: np.ndarray,
    canvas_shape: tuple[int, int],
    margin: float = 0.1,
) -> np.ndarray:
    """Normalize positions to fit within canvas with margin.

    Args:
        positions: Positions in arbitrary coordinates (N, 2)
        canvas_shape: Target canvas (height, width)
        margin: Margin as fraction of canvas size (0.0-0.5)

    Returns:
        Normalized positions in pixel coordinates

    Example:
        >>> positions = np.array([[-1, -1], [1, 1]])
        >>> normalized = normalize_positions_to_canvas(positions, (500, 500))
        >>> # Result will be centered in canvas with margin
    """
    if len(positions) == 0:
        return positions

    # Find bounding box
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    range_pos = max_pos - min_pos

    # Handle edge case where all points are the same
    if np.any(range_pos < 1e-10):
        # Center single point or cluster
        canvas_center = np.array([canvas_shape[1] / 2, canvas_shape[0] / 2])
        return np.tile(canvas_center, (len(positions), 1))

    # Normalize to [0, 1]
    normalized = (positions - min_pos) / range_pos

    # Apply margin and scale to canvas
    margin_size = margin
    scale_factor = 1.0 - 2 * margin_size

    normalized = normalized * scale_factor + margin_size

    # Convert to pixel coordinates
    pixel_positions = normalized * np.array([canvas_shape[1], canvas_shape[0]])

    return pixel_positions
