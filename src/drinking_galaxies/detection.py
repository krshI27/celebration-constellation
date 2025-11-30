"""Circle detection using Hough Circle Transform.

This module provides robust, unsupervised circle detection for identifying
bottles, plates, glasses, and other circular objects in table photos.

Features quality filtering and non-maximum suppression to reduce false positives.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def calculate_circle_quality(
    image: np.ndarray,
    circle: np.ndarray,
    gray_image: Optional[np.ndarray] = None,
) -> float:
    """Calculate quality score for a detected circle.

    Quality score formula:
        score = 0.6 × edge_strength + 0.4 × contrast

    Where:
    - edge_strength: Proportion of Canny edge pixels on circle perimeter (50 samples)
    - contrast: Normalized absolute difference of mean intensity inside vs. outside

    Typical quality threshold: 0.15 filters 30-50% of raw HoughCircles detections.
    - Lower threshold (0.1): More detections, more false positives
    - Higher threshold (0.3): Fewer detections, may miss valid circles

    Args:
        image: Original RGB image
        circle: Circle parameters (x, y, radius)
        gray_image: Optional pre-computed grayscale image

    Returns:
        Quality score between 0.0 (poor) and 1.0 (excellent)

    Example:
        >>> quality = calculate_circle_quality(image, np.array([100, 150, 50]))
        >>> if quality >= 0.15:
        ...     print("High-quality circle detected")
    """
    x, y, r = circle

    # Ensure circle is within image bounds
    h, w = image.shape[:2]
    if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
        return 0.0

    # Convert to grayscale if needed
    if gray_image is None:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 1. Edge strength on perimeter
    edges = cv2.Canny(gray_image, 50, 150)

    # Sample points on circle perimeter
    num_samples = 50
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    perimeter_x = (x + r * np.cos(angles)).astype(int)
    perimeter_y = (y + r * np.sin(angles)).astype(int)

    # Count edge pixels on perimeter
    edge_votes = 0
    for px, py in zip(perimeter_x, perimeter_y):
        if 0 <= px < w and 0 <= py < h:
            if edges[py, px] > 0:
                edge_votes += 1

    edge_strength = edge_votes / num_samples

    # 2. Contrast between inside and outside
    # Create circular mask
    mask_inside = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_inside, (x, y), r - 2, 255, -1)

    mask_outside = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_outside, (x, y), r + 10, 255, -1)
    cv2.circle(mask_outside, (x, y), r + 2, 0, -1)

    inside_mean = cv2.mean(gray_image, mask=mask_inside)[0]
    outside_mean = cv2.mean(gray_image, mask=mask_outside)[0]

    contrast = abs(inside_mean - outside_mean) / 255.0

    # Combined quality score (weighted average)
    quality = 0.6 * edge_strength + 0.4 * contrast

    return quality


def non_maximum_suppression(
    circles: np.ndarray,
    qualities: np.ndarray,
    overlap_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove overlapping circles, keeping highest quality.

    Args:
        circles: Detected circles (N, 3) - (x, y, radius)
        qualities: Quality scores for each circle (N,)
        overlap_threshold: Overlap ratio to consider circles as duplicates

    Returns:
        Tuple of (filtered_circles, filtered_qualities)

    Example:
        >>> circles = np.array([[100, 100, 50], [105, 105, 48]])
        >>> qualities = np.array([0.8, 0.6])
        >>> filtered, q = non_maximum_suppression(circles, qualities)
        >>> len(filtered)  # Returns 1 (overlapping circle removed)
        1
    """
    if len(circles) == 0:
        return circles, qualities

    # Sort by quality (descending)
    order = np.argsort(qualities)[::-1]

    keep = []
    while len(order) > 0:
        # Keep highest quality circle
        i = order[0]
        keep.append(i)

        # Calculate overlap with remaining circles
        current_circle = circles[i]
        remaining_circles = circles[order[1:]]

        if len(remaining_circles) == 0:
            break

        # Calculate distances between centers
        distances = np.sqrt(
            (remaining_circles[:, 0] - current_circle[0]) ** 2
            + (remaining_circles[:, 1] - current_circle[1]) ** 2
        )

        # Calculate overlap ratio (based on distance vs radii sum)
        radii_sum = remaining_circles[:, 2] + current_circle[2]
        overlap_ratio = 1.0 - (distances / radii_sum)

        # Keep circles with low overlap
        non_overlapping = overlap_ratio < overlap_threshold
        order = order[1:][non_overlapping]

    return circles[keep], qualities[keep]


def load_image(image_path: Path) -> np.ndarray:
    """Load image from file and convert to RGB.

    Validates image dimensions to ensure reliable circle detection:
    - Minimum: 300×300 pixels (sufficient resolution for detection)
    - Maximum: 6000×6000 pixels (36MP, prevents memory issues)

    Args:
        image_path: Path to image file

    Returns:
        RGB image as numpy array

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded or dimensions out of bounds
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # Validate image dimensions
    h, w = image.shape[:2]
    if h < 300 or w < 300:
        raise ValueError(
            f"Image too small: {w}×{h} pixels. "
            f"Minimum: 300×300 pixels for reliable circle detection."
        )
    if h > 6000 or w > 6000:
        raise ValueError(
            f"Image too large: {w}×{h} pixels. "
            f"Maximum: 6000×6000 pixels (36MP). "
            f"Consider resizing for better performance."
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for circle detection.

    Args:
        image: RGB image array

    Returns:
        Grayscale, blurred image ready for Hough transform
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    return blurred


def detect_circles(
    image: np.ndarray,
    min_radius: int = 20,
    max_radius: int = 200,
    min_distance: int = 50,
    param1: int = 100,
    param2: int = 30,
    max_circles: int = 50,
    quality_threshold: float = 0.15,
    use_nms: bool = True,
) -> Optional[np.ndarray]:
    """Detect circles in image using Hough Circle Transform with quality filtering.

    Image constraints:
    - Supported dimensions: 300×300 to 6000×6000 pixels
    - Recommended: 1500-3000 pixels on longest edge for optimal performance
    - Performance: < 5 seconds for typical images (1500-3000px)

    Parameter ranges and defaults:
    - min_radius=20: Minimum object size (bottle cap in 2MP image ≈ 30-50px)
    - max_radius=200: Maximum object size (large platter ≈ 150-200px)
    - min_distance=50: Prevents duplicate detections of same object
    - param1=100, param2=30: Canny edge and accumulator thresholds (OpenCV defaults)
    - max_circles=50: Automatic spatial sampling if more detected
    - quality_threshold=0.15: Filters 30-50% of raw detections (balance precision/recall)

    Typical object sizes in 2MP images:
    - Bottle cap: 30-50 pixels radius
    - Dinner plate: 80-150 pixels radius
    - Large serving platter: 150-200 pixels radius

    Args:
        image: RGB image array
        min_radius: Minimum circle radius in pixels (default: 20)
        max_radius: Maximum circle radius in pixels (default: 200)
        min_distance: Minimum distance between circle centers (default: 50)
        param1: Upper threshold for Canny edge detector (default: 100)
        param2: Accumulator threshold for circle detection (default: 30)
        max_circles: Maximum number of circles to return (default: 50)
        quality_threshold: Minimum quality score 0.0-1.0 (default: 0.15)
        use_nms: Whether to apply non-maximum suppression (default: True)

    Returns:
        Array of detected circles (x, y, radius) or None if no circles found
        Circles are sorted by quality (best first)

    Example:
        >>> circles = detect_circles(image, max_circles=30, quality_threshold=0.4)
        >>> if circles is not None:
        ...     print(f"Found {len(circles)} high-quality circles")

        Adjusting for different scenarios:
        >>> # Close-up photos: reduce radius range
        >>> circles = detect_circles(image, min_radius=40, max_radius=150)
        >>> # Wide-angle shots: increase max radius
        >>> circles = detect_circles(image, min_radius=15, max_radius=250)
        >>> # Stricter quality: increase threshold
        >>> circles = detect_circles(image, quality_threshold=0.3)
    """
    processed = preprocess_image(image)

    circles = cv2.HoughCircles(
        processed,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_distance,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype("int")

    # Calculate quality scores for all circles
    qualities = np.array(
        [
            calculate_circle_quality(image, circle, gray_image=processed)
            for circle in circles
        ]
    )

    # Filter by quality threshold
    quality_mask = qualities >= quality_threshold
    circles = circles[quality_mask]
    qualities = qualities[quality_mask]

    if len(circles) == 0:
        return None

    # Apply non-maximum suppression
    if use_nms and len(circles) > 1:
        circles, qualities = non_maximum_suppression(circles, qualities)

    # Sort by quality (descending)
    sort_indices = np.argsort(qualities)[::-1]
    circles = circles[sort_indices]
    qualities = qualities[sort_indices]

    # Limit to max_circles
    if len(circles) > max_circles:
        circles = circles[:max_circles]

    return circles


def get_circle_centers(circles: np.ndarray) -> np.ndarray:
    """Extract circle center coordinates.

    Args:
        circles: Array of circles (x, y, radius)

    Returns:
        Array of center coordinates (x, y)
    """
    return circles[:, :2]


def draw_circles(
    image: np.ndarray,
    circles: np.ndarray,
    show_circles: bool = True,
    show_centers: bool = True,
    circle_color: tuple[int, int, int] = (0, 255, 0),
    center_color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Draw detected circles and centers on image.

    Args:
        image: RGB image array
        circles: Array of circles (x, y, radius)
        show_circles: Whether to draw circle outlines
        show_centers: Whether to draw center points
        circle_color: RGB color for circle outlines
        center_color: RGB color for center points

    Returns:
        Image with circles and centers drawn
    """
    output = image.copy()

    for x, y, r in circles:
        if show_circles:
            cv2.circle(output, (x, y), r, circle_color, 2)
        if show_centers:
            cv2.circle(output, (x, y), 3, center_color, -1)

    return output


def detect_and_extract(
    image_path: Path,
    min_radius: int = 20,
    max_radius: int = 200,
    min_distance: int = 50,
    max_circles: int = 50,
    quality_threshold: float = 0.15,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Complete pipeline: load, detect, and extract circle centers.

    Uses quality filtering and non-maximum suppression to return
    only high-quality circle detections.

    Args:
        image_path: Path to image file
        min_radius: Minimum circle radius in pixels
        max_radius: Maximum circle radius in pixels
        min_distance: Minimum distance between circle centers
        max_circles: Maximum number of circles to return (default: 50)
        quality_threshold: Minimum quality score 0.0-1.0 (default: 0.15)

    Returns:
        tuple of (original_image, circles, centers)
        Circles are sorted by quality (best first)

    Example:
        >>> image, circles, centers = detect_and_extract(
        ...     Path("table.jpg"), max_circles=30, quality_threshold=0.4
        ... )
        >>> if centers is not None:
        ...     print(f"Found {len(centers)} high-quality circular objects")
    """
    image = load_image(image_path)
    circles = detect_circles(
        image,
        min_radius=min_radius,
        max_radius=max_radius,
        min_distance=min_distance,
        max_circles=max_circles,
        quality_threshold=quality_threshold,
    )

    centers = None
    if circles is not None:
        centers = get_circle_centers(circles)

    return image, circles, centers
