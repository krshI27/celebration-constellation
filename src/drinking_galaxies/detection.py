"""Circle detection using Hough Circle Transform.

This module provides robust, unsupervised circle detection for identifying
bottles, plates, glasses, and other circular objects in table photos.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def load_image(image_path: Path) -> np.ndarray:
    """Load image from file and convert to RGB.

    Args:
        image_path: Path to image file

    Returns:
        RGB image as numpy array

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

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
) -> Optional[np.ndarray]:
    """Detect circles in image using Hough Circle Transform.

    Args:
        image: RGB image array
        min_radius: Minimum circle radius in pixels
        max_radius: Maximum circle radius in pixels
        min_distance: Minimum distance between circle centers
        param1: Upper threshold for Canny edge detector
        param2: Accumulator threshold for circle detection

    Returns:
        Array of detected circles (x, y, radius) or None if no circles found
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

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles

    return None


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
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Complete pipeline: load, detect, and extract circle centers.

    Args:
        image_path: Path to image file
        min_radius: Minimum circle radius in pixels
        max_radius: Maximum circle radius in pixels
        min_distance: Minimum distance between circle centers

    Returns:
        tuple of (original_image, circles, centers)

    Example:
        >>> image, circles, centers = detect_and_extract(Path("table.jpg"))
        >>> if centers is not None:
        ...     print(f"Found {len(centers)} circular objects")
    """
    image = load_image(image_path)
    circles = detect_circles(
        image,
        min_radius=min_radius,
        max_radius=max_radius,
        min_distance=min_distance,
    )

    centers = None
    if circles is not None:
        centers = get_circle_centers(circles)

    return image, circles, centers
