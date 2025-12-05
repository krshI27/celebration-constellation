"""Streamlit web application for Celebration Constellation.

This app provides an interactive interface for:
- Uploading table photos
- Detecting circular objects
- Matching to star constellations
- Swipeable image comparison between photo and sky
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src directory to Python path for Streamlit Cloud deployment
# This ensures the celebration_constellation package can be imported
_src_path = Path(__file__).parent / "src"
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Configure OpenCV for headless environments (Streamlit Cloud)
# Prevents "libGL.so.1: cannot open shared object file" errors
os.environ["LIBGL_ALWAYS_INDIRECT"] = "1"

import astropy.units as u
import cv2
import numpy as np
import streamlit as st
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from PIL import Image
from streamlit_geolocation import streamlit_geolocation as geolocation

from celebration_constellation.astronomy import StarCatalog
from celebration_constellation.detection import detect_and_extract, draw_circles
from celebration_constellation.lines import (
    build_line_segments_for_region,
    load_constellation_lines,
)
from celebration_constellation.matching import match_to_sky_regions
from celebration_constellation.visualization import (
    create_composite_overlay,
    create_constellation_visualization,
    draw_constellation_lines,
    normalize_positions_to_canvas,
)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color (e.g., #00D9FF) to RGB tuple.

    Falls back to cyan if parsing fails.
    """
    try:
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except Exception:
        return (0, 217, 255)


def get_theme_primary_rgb() -> tuple[int, int, int]:
    """Get Streamlit theme primaryColor as RGB tuple, defaulting to cyan."""
    primary_hex = st.get_option("theme.primaryColor") or "#00D9FF"
    return _hex_to_rgb(primary_hex)


def to_gray_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale, then back to 3-channel RGB."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def apply_photometric_stretch(image: np.ndarray) -> np.ndarray:
    """Apply photometric stretching for better contrast.

    Uses percentile-based normalization to enhance visibility.
    """
    try:
        from astropy.visualization import ImageNormalize, PercentileInterval

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Apply percentile normalization
        interval = PercentileInterval(99.5)
        norm = ImageNormalize(gray, interval=interval)
        stretched = norm(gray)

        # Convert back to uint8 and RGB
        stretched_uint8 = (stretched * 255).astype(np.uint8)
        return cv2.cvtColor(stretched_uint8, cv2.COLOR_GRAY2RGB)
    except ImportError:
        # Fall back to simple grayscale if astropy not available
        return to_gray_rgb(image)


def get_per_star_colors(
    b_v_values: np.ndarray, theme_rgb: tuple[int, int, int] = (0, 217, 255)
) -> np.ndarray:
    """Map B-V color index to RGB colors (blue/hot to red/cool).

    Args:
        b_v_values: Array of B-V color indices
        theme_rgb: Default color for missing data

    Returns:
        Array of RGB colors corresponding to B-V values
    """
    colors = []
    for bv in b_v_values:
        if np.isnan(bv):
            colors.append(theme_rgb)
        elif bv <= 0.0:
            colors.append((180, 200, 255))  # Blue (very hot stars)
        elif bv <= 0.3:
            colors.append((220, 230, 255))  # Light blue
        elif bv <= 0.6:
            colors.append((255, 255, 255))  # White
        elif bv <= 1.0:
            colors.append((255, 235, 180))  # Yellow
        else:
            colors.append((255, 210, 150))  # Orange/red (cool stars)
    return np.array(colors, dtype=np.uint8)


# Palette for synthwave-inspired overlay
NEON_PINK = (255, 80, 180)


def create_synthwave_gradient(shape: tuple[int, int, int]) -> np.ndarray:
    """Create a subtle synthwave-style gradient background.

    Uses a vertical purple→blue blend with gentle sine modulation to avoid flat fills.
    """
    if len(shape) < 2:
        raise ValueError("Expected image shape with height and width")

    height, width = shape[:2]
    channels = shape[2] if len(shape) > 2 else 3

    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    top = np.array([60, 30, 110], dtype=np.float32)  # muted violet
    bottom = np.array([15, 45, 95], dtype=np.float32)  # deep blue

    vertical_blend = (top * (1.0 - y) + bottom * y)[:, None, :]
    base = np.broadcast_to(vertical_blend, (height, width, 3))

    # Gentle diagonal waves for depth; keep subtle to avoid noise
    ripple = 6.0 * np.sin((x * 1.2 + y * 1.6) * np.pi)
    gradient = base + ripple[:, :, None]
    gradient = np.clip(gradient, 0, 255).astype(np.uint8)

    if channels == 4:
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        gradient = np.concatenate([gradient, alpha], axis=2)

    return gradient


def build_background(
    image: np.ndarray, mode: str = "Original", darken_factor: float = 0.35
) -> np.ndarray:
    """Construct background based on selected mode."""

    if mode == "Original":
        return image.copy()

    if mode == "Greyscale":
        return to_gray_rgb(image)

    if mode == "Dark Greyscale":
        gray = to_gray_rgb(image)
        return (gray.astype(np.float32) * darken_factor).astype(np.uint8)

    if mode == "Black":
        return np.zeros_like(image)

    if mode == "Synthwave Gradient":
        return create_synthwave_gradient(image.shape)

    if mode == "Synthwave Colorized":
        synth_base = create_synthwave_gradient(image.shape)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_norm = cv2.normalize(
            gray.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX
        )
        return (
            (synth_base.astype(np.float32) * gray_norm[:, :, None])
            .clip(0, 255)
            .astype(np.uint8)
        )

    # Fallback
    return image.copy()


def resolve_star_color(
    star_color_mode: str,
    theme_rgb: tuple[int, int, int],
    per_star_colors: np.ndarray | None,
) -> tuple[np.ndarray | None, tuple[int, int, int]]:
    """Return per-star palette and fallback color based on mode."""

    mode = (star_color_mode or "").lower()

    if mode.startswith("color by"):
        return per_star_colors, theme_rgb

    if mode.startswith("white"):
        return None, (245, 245, 245)

    if "neon" in mode:
        return None, NEON_PINK

    # Default to theme
    return None, theme_rgb


def estimate_radius_bounds(image: np.ndarray) -> tuple[int, int]:
    """Estimate min/max circle radii from image statistics.

    Uses edge detection and blob analysis on downscaled image.
    """
    # Downscale for speed
    h, w = image.shape[:2]
    scale = 1024 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(image, None, fx=scale, fy=scale)
    else:
        small = image
        scale = 1.0

    # Edge detection
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours as potential circles
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # No contours found, use defaults
        return 20, 200

    # Estimate radii from contour bounding circles
    radii = []
    for cnt in contours:
        if len(cnt) >= 5:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            radii.append(radius)

    if len(radii) == 0:
        return 20, 200

    # Compute percentiles and scale back
    radii = np.array(radii)
    min_r = max(10, int(np.percentile(radii, 20) / scale))
    max_r = min(300, int(np.percentile(radii, 85) / scale))

    # Ensure reasonable bounds
    if min_r >= max_r:
        min_r = 20
        max_r = 200

    return min_r, max_r


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

    if "uploaded_path" not in st.session_state:
        st.session_state.uploaded_path = None

    if "last_uploaded_name" not in st.session_state:
        st.session_state.last_uploaded_name = None

    if "circles" not in st.session_state:
        st.session_state.circles = None

    if "centers" not in st.session_state:
        st.session_state.centers = None

    if "matches" not in st.session_state:
        st.session_state.matches = []

    if "current_match_index" not in st.session_state:
        st.session_state.current_match_index = 0

    if "bg_ready_dark" not in st.session_state:
        st.session_state.bg_ready_dark = False

    if "bg_ready_synth" not in st.session_state:
        st.session_state.bg_ready_synth = False

    if "circles_ready" not in st.session_state:
        st.session_state.circles_ready = False

    if "matches_ready" not in st.session_state:
        st.session_state.matches_ready = False

    if "active_backgrounds" not in st.session_state:
        st.session_state.active_backgrounds = ["Black"]

    if "background_cache" not in st.session_state:
        st.session_state.background_cache = {}

    if "show_circles" not in st.session_state:
        st.session_state.show_circles = True

    if "show_centers" not in st.session_state:
        st.session_state.show_centers = True

    if "background_mode" not in st.session_state:
        st.session_state.background_mode = "Black"

    if "star_color_mode" not in st.session_state:
        st.session_state.star_color_mode = "Color by B–V"

    if "show_stars" not in st.session_state:
        st.session_state.show_stars = True

    if "show_constellation_lines" not in st.session_state:
        st.session_state.show_constellation_lines = False

    if "pending_params" not in st.session_state:
        st.session_state.pending_params = None

    if "star_catalog" not in st.session_state:
        st.session_state.star_catalog = StarCatalog()


def reset_pipeline_state(clear_image: bool = False):
    """Reset staged outputs and readiness flags.

    Args:
        clear_image: When True, also drop the uploaded image reference.
    """

    if clear_image:
        st.session_state.uploaded_image = None
        st.session_state.uploaded_path = None

    st.session_state.background_cache = {}
    st.session_state.active_backgrounds = ["Black"]
    st.session_state.bg_ready_dark = False
    st.session_state.bg_ready_synth = False
    st.session_state.circles_ready = False
    st.session_state.matches_ready = False
    st.session_state.circles = None
    st.session_state.centers = None
    st.session_state.matches = []
    st.session_state.current_match_index = 0


def process_uploaded_image(
    uploaded_file,
    min_radius: int,
    max_radius: int,
    max_circles: int = 50,
    quality_threshold: float = 0.15,
):
    """Process uploaded image and detect circles with quality filtering."""
    # Save uploaded file temporarily
    temp_path = Path("/tmp/uploaded_image.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Detect circles with quality filtering
    image, circles, centers = detect_and_extract(
        temp_path,
        min_radius=min_radius,
        max_radius=max_radius,
        max_circles=max_circles,
        quality_threshold=quality_threshold,
    )

    st.session_state.uploaded_image = image
    st.session_state.circles = circles
    st.session_state.centers = centers

    return image, circles, centers


def _compute_staged_backgrounds(image: np.ndarray) -> dict[str, np.ndarray]:
    """Compute baseline and derived backgrounds for staged availability."""

    st.session_state.background_cache = {}
    st.session_state.active_backgrounds = ["Black"]

    # Baseline black background
    st.session_state.background_cache["Black"] = np.zeros_like(image)

    # Baseline synthwave gradient (immediately available)
    synth_base = create_synthwave_gradient(image.shape)
    st.session_state.background_cache["Synthwave Gradient"] = synth_base
    st.session_state.active_backgrounds.append("Synthwave Gradient")

    # Derived dark greyscale
    dark_gray = build_background(image, mode="Dark Greyscale")
    st.session_state.background_cache["Dark Greyscale"] = dark_gray
    st.session_state.bg_ready_dark = True
    st.session_state.active_backgrounds.append("Dark Greyscale")

    # Synthwave colorized greyscale (image-weighted gradient)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_norm = cv2.normalize(gray.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    synth_colorized = (
        (synth_base.astype(np.float32) * gray_norm[:, :, None])
        .clip(0, 255)
        .astype(np.uint8)
    )
    st.session_state.background_cache["Synthwave Colorized"] = synth_colorized
    st.session_state.bg_ready_synth = True
    st.session_state.active_backgrounds.append("Synthwave Colorized")

    # Original (optional) once available
    st.session_state.background_cache["Original"] = image
    st.session_state.active_backgrounds.append("Original")

    return st.session_state.background_cache


def run_pipeline(
    uploaded_file,
    *,
    detection_scale: str,
    quality_threshold: float,
    num_regions: int,
    radius_deg: float,
    star_brightness: float,
    max_circles: int = 50,
    live_placeholders: dict[str, Any] | None = None,
    detection_progress_placeholder: Any | None = None,
):
    """Run staged processing pipeline and update session state."""

    live_slots = live_placeholders or {}
    progress_bar = None

    def update_live(name: str, image: np.ndarray | None, caption: str):
        slot = live_slots.get(name)
        if slot is not None and image is not None:
            slot.image(image, caption=caption, use_container_width=True)

    def update_progress(value: float, text: str):
        nonlocal progress_bar
        if detection_progress_placeholder is None:
            return
        if progress_bar is None:
            progress_bar = detection_progress_placeholder.progress(value, text=text)
        else:
            progress_bar.progress(value, text=text)

    # If new upload provided, store and reset pipeline outputs
    if uploaded_file is not None:
        reset_pipeline_state(clear_image=False)
        temp_path = Path("/tmp/uploaded_image.jpg")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.uploaded_path = temp_path
        st.session_state.last_uploaded_name = getattr(uploaded_file, "name", None)

        # Read image into memory for backgrounds
        uploaded_file.seek(0)
        pil_image = Image.open(uploaded_file)
        image = np.array(pil_image.convert("RGB"))
        st.session_state.uploaded_image = image
        bg_cache = _compute_staged_backgrounds(image)
        update_live("uploaded", image, "Uploaded image")
        dark_greyscale = bg_cache.get("Dark Greyscale")
        update_live(
            "background",
            dark_greyscale if dark_greyscale is not None else to_gray_rgb(image),
            "Greyscale preview",
        )
    else:
        image = st.session_state.uploaded_image
        if image is not None:
            update_live("uploaded", image, "Uploaded image")
            gray_preview = (
                st.session_state.background_cache.get("Dark Greyscale")
                if st.session_state.background_cache
                else to_gray_rgb(image)
            )
            update_live("background", gray_preview, "Greyscale preview")

    if image is None or st.session_state.uploaded_path is None:
        return

    # Estimate radii if auto
    if detection_scale == "Auto":
        min_radius, max_radius = estimate_radius_bounds(image)
    elif detection_scale == "Close-up":
        min_radius, max_radius = 40, 150
    elif detection_scale == "Wide":
        min_radius, max_radius = 15, 250
    else:
        min_radius, max_radius = 20, 200

    # Circle detection
    update_progress(0.05, "Preparing image for detection...")
    with st.spinner("Detecting circular objects..."):
        update_progress(0.35, "Running circle detection...")
        image, circles, centers = detect_and_extract(
            Path(st.session_state.uploaded_path),
            min_radius=min_radius,
            max_radius=max_radius,
            max_circles=max_circles,
            quality_threshold=quality_threshold,
        )
    update_progress(0.9, "Scoring detections...")
    st.session_state.uploaded_image = image
    st.session_state.circles = circles
    st.session_state.centers = centers
    st.session_state.circles_ready = circles is not None and len(circles) > 0

    if circles is not None and len(circles):
        detection_preview = draw_circles(
            image,
            circles,
            show_circles=True,
            show_centers=False,
            circle_color=get_theme_primary_rgb(),
            fill_alpha=0.2,
        )
        update_live("detection", detection_preview, "Detected circles")
    else:
        slot = live_slots.get("detection")
        if slot is not None:
            slot.info("No circles detected.")
    update_progress(1.0, "Circle detection complete")

    # Constellation matching (only if circles/centers found)
    if centers is not None and len(centers):
        matches = find_constellation_matches(centers, num_regions, float(radius_deg))
        st.session_state.matches = matches or []
        st.session_state.matches_ready = bool(matches)
        st.session_state.current_match_index = 0
    else:
        st.session_state.matches = []
        st.session_state.matches_ready = False

    # Persist rendering params
    st.session_state.star_brightness = star_brightness
    st.session_state.pending_params = {
        "detection_scale": detection_scale,
        "quality_threshold": quality_threshold,
        "num_regions": num_regions,
        "radius_deg": radius_deg,
        "star_brightness": star_brightness,
        "min_radius": min_radius,
        "max_radius": max_radius,
    }


def find_constellation_matches(
    centers: np.ndarray,
    num_regions: int = 100,
    radius_deg: float = 20.0,
):
    """Find matching star constellations."""
    catalog = st.session_state.star_catalog

    with st.spinner("Searching the night sky for matching constellations..."):
        # Sample sky regions
        regions = catalog.search_sky_regions(
            num_samples=num_regions, radius_deg=radius_deg
        )

        # Add stereographic projections
        for region in regions:
            positions = catalog.convert_to_stereographic(
                region["stars"], region["ra"], region["dec"]
            )
            region["positions"] = positions

        # Match circle centers to sky regions
        # Progress bar for matching
        progress = st.progress(0.0)

        def _progress_cb(done: int, total: int):
            frac = max(0.0, min(1.0, done / max(total, 1)))
            progress.progress(frac)

        matches = match_to_sky_regions(centers, regions, progress_callback=_progress_cb)

    st.session_state.matches = matches
    st.session_state.current_match_index = 0

    return matches


def render_sky_visualization(match: dict, brightness: float = 1.0) -> np.ndarray:
    """Render star constellation with magnitude-scaled stars.

    Args:
        match: Match result dict with star positions, magnitudes, and transform
        brightness: Multiplier for star size (1.0 = normal)

    Returns:
        RGB image with enhanced star visualization
    """
    # Get image shape
    image_shape = st.session_state.uploaded_image.shape

    # Get star positions in stereographic projection space and transform to image coords
    target_positions = match.get("target_positions")
    if target_positions is None:
        star_positions = np.empty((0, 2))
    else:
        # Apply inverse transform to map star positions to image coordinates
        star_positions = apply_inverse_transform(
            target_positions,
            match["scale"],
            match["rotation"],
            match["translation"],
        )

    # Get star magnitudes if available
    magnitudes = None
    if "stars" in match and "magnitude" in match["stars"].columns:
        magnitudes = match["stars"]["magnitude"].values[: len(star_positions)]

    # Get per-star colors from B-V temperature index
    per_star_colors = None
    if "stars" in match and "b_v" in match["stars"].columns:
        bvs = match["stars"]["b_v"].values[: len(star_positions)]
        per_star_colors = get_per_star_colors(bvs, (255, 255, 220))

    # Create dark sky background
    canvas = np.full(image_shape, (5, 5, 15), dtype=np.uint8)

    # Render stars with magnitude scaling and B-V colors
    for idx, (x, y) in enumerate(star_positions):
        x_int, y_int = int(x), int(y)
        if not (0 <= x_int < canvas.shape[1] and 0 <= y_int < canvas.shape[0]):
            continue

        # Get magnitude for this star
        mag = magnitudes[idx] if magnitudes is not None else 3.0
        mag_clamped = max(0.0, min(6.0, mag))

        # Radius based on magnitude (brighter = larger), scaled by brightness multiplier
        base_radius = 35.0 * np.exp(-mag_clamped / 3.0)
        radius = max(4, min(60, int(base_radius * brightness)))

        # Get color for this star
        if per_star_colors is not None:
            color = tuple(int(c) for c in per_star_colors[idx])
        else:
            color = (255, 255, 220)

        # Draw filled circle
        cv2.circle(canvas, (x_int, y_int), radius, color, -1)

    return canvas


def apply_inverse_transform(
    points: np.ndarray,
    scale: float,
    rotation: float,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply inverse 2D similarity transform to map points to image space.

    Given the transform that maps circles -> stars:
        transformed = scale * (circles @ R.T) + translation

    The inverse maps stars -> image space:
        image_coords = (stars - translation) @ R / scale

    Args:
        points: Points to transform (N, 2) in star/target space
        scale: Scale factor from original transform
        rotation: Rotation angle in radians from original transform
        translation: Translation vector (2,) from original transform

    Returns:
        Transformed points in image coordinate space
    """
    # Build rotation matrix (same as forward transform)
    R = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]
    )

    # Inverse: subtract translation, apply inverse rotation (R.T = R^-1), divide by scale
    centered = points - translation
    return (centered @ R) / scale


def render_composite_overlay(match: dict) -> np.ndarray:
    """Render composite overlay showing both circles and stars.

    Uses photometric stretching and per-star colors based on B-V temperature.

    Args:
        match: Match result dict with star positions, magnitudes, and transform

    Returns:
        RGB image with circles and stars overlaid
    """
    # Get star positions in stereographic projection space
    target_positions = match.get("target_positions")
    if target_positions is None:
        # Fallback to empty if not available
        star_positions = np.empty((0, 2))
    else:
        # Apply inverse transform to map star positions to image coordinates
        star_positions = apply_inverse_transform(
            target_positions,
            match["scale"],
            match["rotation"],
            match["translation"],
        )

    # Get star magnitudes if available
    magnitudes = None
    if "stars" in match and "magnitude" in match["stars"].columns:
        magnitudes = match["stars"]["magnitude"].values[: len(star_positions)]

    # Theme highlight color
    theme_rgb = get_theme_primary_rgb()

    # Always apply photometric stretching for better contrast
    gray_base = apply_photometric_stretch(st.session_state.uploaded_image)

    # Do not show circle centers in overlay
    empty_centers = np.empty((0, 2), dtype=np.int32)

    # Create composite overlay: grayscale photo + theme-colored stars
    # If B-V color index present, map to per-star RGB colors (respect global toggle if present)
    per_star_colors = None
    use_bv_setting = st.session_state.get("color_by_bv", True)
    if use_bv_setting and "stars" in match and "b_v" in match["stars"].columns:
        bvs = match["stars"]["b_v"].values[: len(star_positions)]
        per_star_colors = get_per_star_colors(bvs, theme_rgb)

    # Build overlay with per-star colors if available
    if per_star_colors is not None:
        # Ensure gray_base is 3-channel RGB before operations
        if len(gray_base.shape) == 2:
            gray_base = cv2.cvtColor(gray_base, cv2.COLOR_GRAY2RGB)
        elif gray_base.shape[2] == 1:
            gray_base = cv2.cvtColor(gray_base[:, :, 0], cv2.COLOR_GRAY2RGB)

        overlay = gray_base.copy()
        star_layer = np.zeros_like(gray_base)

        # Render stars with theme color and magnitude scaling
        for (x, y), mag in zip(
            star_positions,
            magnitudes if magnitudes is not None else [3.0] * len(star_positions),
        ):
            # Use larger radius for better visibility (not clamped by magnitude_to_radius)
            # Brighter stars (lower magnitude) get larger radii
            mag_clamped = max(0.0, min(6.0, mag))
            radius = max(6, min(50, int(40.0 * np.exp(-mag_clamped / 3.0))))
            x_int, y_int = int(x), int(y)

            if 0 <= x_int < star_layer.shape[1] and 0 <= y_int < star_layer.shape[0]:
                # Draw filled circle with theme color (same as circles)
                cv2.circle(star_layer, (x_int, y_int), radius, theme_rgb, -1)

                # Add very prominent glow effect
                glow_radius = radius + 8
                glow_color = tuple(int(c * 0.9) for c in theme_rgb)
                cv2.circle(star_layer, (x_int, y_int), glow_radius, glow_color, 4)

        # Apply light Gaussian blur for soft glow
        star_layer = cv2.GaussianBlur(star_layer, (5, 5), 1.0)

        # Darken the grayscale base significantly for better star visibility
        darkened_base = (gray_base * 0.35).astype(np.uint8)

        # Blend with very high star prominence
        overlay = cv2.addWeighted(darkened_base, 1.0, star_layer, 3.0, 0)
    else:
        # Fall back to single-color rendering with darkened background
        # Ensure gray_base is 3-channel RGB before operations
        if len(gray_base.shape) == 2:
            gray_base = cv2.cvtColor(gray_base, cv2.COLOR_GRAY2RGB)
        elif gray_base.shape[2] == 1:
            gray_base = cv2.cvtColor(gray_base[:, :, 0], cv2.COLOR_GRAY2RGB)

        darkened_base = (gray_base * 0.35).astype(np.uint8)
        overlay = create_composite_overlay(
            darkened_base,
            empty_centers,
            star_positions,
            magnitudes=magnitudes,
            circle_color=theme_rgb,
            star_color=theme_rgb,
            alpha=3.0,
            star_scale_factor=40.0,
        )

    return overlay


def render_unified_view(
    match: dict,
    brightness: float = 1.0,
    star_color_mode: str = "Color by B–V",
    background_mode: str = "Original",
    show_stars: bool = True,
    show_circles: bool = True,
    show_lines: bool = False,
) -> np.ndarray:
    """Render a unified view based on toggles.

    - Background: photo (darkened) or black
    - Stars: B-V color or theme color; magnitude-scaled sizes with brightness multiplier
    - Circles: optional overlays
    - Constellation lines: optional, if data available
    """
    image_shape = st.session_state.uploaded_image.shape
    theme_rgb = get_theme_primary_rgb()

    # Star positions in image coordinates
    target_positions = match.get("target_positions")
    if target_positions is None:
        star_positions = np.empty((0, 2))
    else:
        star_positions = apply_inverse_transform(
            target_positions, match["scale"], match["rotation"], match["translation"]
        )

    # Magnitudes
    magnitudes = None
    if "stars" in match and "magnitude" in match["stars"].columns:
        magnitudes = match["stars"]["magnitude"].values[: len(star_positions)]

    per_star_colors = None
    if "stars" in match and "b_v" in match["stars"].columns:
        bvs = match["stars"]["b_v"].values[: len(star_positions)]
        per_star_colors = get_per_star_colors(bvs, theme_rgb)

    # Base background
    photo_base = apply_photometric_stretch(st.session_state.uploaded_image)
    if len(photo_base.shape) == 2:
        photo_base = cv2.cvtColor(photo_base, cv2.COLOR_GRAY2RGB)
    elif photo_base.shape[2] == 1:
        photo_base = cv2.cvtColor(photo_base[:, :, 0], cv2.COLOR_GRAY2RGB)

    base = build_background(photo_base, mode=background_mode)
    canvas = base.copy()

    # Star colors
    if (
        background_mode == "Synthwave Gradient"
        and not star_color_mode.lower().startswith("color by")
    ):
        star_color_mode = "Neon pink"

    per_star_colors, fallback_color = resolve_star_color(
        star_color_mode, theme_rgb, per_star_colors
    )

    # Draw stars
    if show_stars:
        for idx, (x, y) in enumerate(star_positions):
            x_int, y_int = int(x), int(y)
            if not (0 <= x_int < canvas.shape[1] and 0 <= y_int < canvas.shape[0]):
                continue

            mag = magnitudes[idx] if magnitudes is not None else 3.0
            mag_clamped = max(0.0, min(6.0, mag))
            base_radius = 35.0 * np.exp(-mag_clamped / 3.0)
            radius = max(4, min(60, int(base_radius * float(brightness))))

            if per_star_colors is not None:
                color = tuple(int(c) for c in per_star_colors[idx])
            else:
                color = fallback_color

            cv2.circle(canvas, (x_int, y_int), radius, color, -1)

    # Constellation lines (optional)
    if show_lines and match.get("constellation") and len(star_positions):
        # Load line map lazily and cache in session
        if "_lines_map" not in st.session_state:
            st.session_state._lines_map = load_constellation_lines()
        lines_map = st.session_state._lines_map
        segments = build_line_segments_for_region(
            match.get("stars"), match.get("constellation"), lines_map
        )
        if segments:
            # Draw on top using a contrasting color (theme)
            canvas = draw_constellation_lines(
                canvas, star_positions, segments, color=(100, 150, 255), thickness=2
            )

    # Circles (optional)
    if show_circles:
        circles = st.session_state.circles
        if circles is not None and len(circles):
            for x, y, r in circles:
                x_int, y_int, r_int = int(x), int(y), int(r)
                if 0 <= x_int < canvas.shape[1] and 0 <= y_int < canvas.shape[0]:
                    fill_layer = np.zeros_like(canvas)
                    cv2.circle(fill_layer, (x_int, y_int), r_int, theme_rgb, -1)
                    canvas = cv2.addWeighted(canvas, 1.0, fill_layer, 0.25, 0)
                    cv2.circle(canvas, (x_int, y_int), r_int, theme_rgb, 3)

    return canvas


def render_circles_on_stars(match: dict, brightness: float = 1.0) -> np.ndarray:
    """Render circle positions overlaid on star constellation pattern.

    Uses per-star colors based on B-V temperature index.

    Args:
        match: Match result dict with star positions, magnitudes, and transform
        brightness: Multiplier for star size (1.0 = normal)

    Returns:
        RGB image with circles drawn on star pattern
    """
    # Get image shape
    image_shape = st.session_state.uploaded_image.shape

    # Get star positions in stereographic projection and transform to image coords
    target_positions = match.get("target_positions")
    if target_positions is None:
        star_positions = np.empty((0, 2))
    else:
        # Apply inverse transform to map star positions to image coordinates
        star_positions = apply_inverse_transform(
            target_positions,
            match["scale"],
            match["rotation"],
            match["translation"],
        )

    # Get detected circles (x, y, radius) - already in image coordinates
    circles = st.session_state.circles

    # Get star magnitudes if available
    magnitudes = None
    if "stars" in match and "magnitude" in match["stars"].columns:
        magnitudes = match["stars"]["magnitude"].values[: len(star_positions)]

    # Theme color for circle outlines
    theme_rgb = get_theme_primary_rgb()

    # Get per-star colors from B-V temperature index
    per_star_colors = None
    if "stars" in match and "b_v" in match["stars"].columns:
        bvs = match["stars"]["b_v"].values[: len(star_positions)]
        per_star_colors = get_per_star_colors(bvs, (255, 255, 220))

    # Create dark sky background
    canvas = np.full(image_shape, (5, 5, 15), dtype=np.uint8)

    # Render stars with magnitude scaling and B-V colors
    for idx, (x, y) in enumerate(star_positions):
        x_int, y_int = int(x), int(y)
        if not (0 <= x_int < canvas.shape[1] and 0 <= y_int < canvas.shape[0]):
            continue

        # Get magnitude for this star
        mag = magnitudes[idx] if magnitudes is not None else 3.0
        mag_clamped = max(0.0, min(6.0, mag))

        # Radius based on magnitude (brighter = larger), scaled by brightness multiplier
        base_radius = 35.0 * np.exp(-mag_clamped / 3.0)
        radius = max(4, min(60, int(base_radius * brightness)))

        # Get color for this star
        if per_star_colors is not None:
            color = tuple(int(c) for c in per_star_colors[idx])
        else:
            color = (255, 255, 220)

        # Draw filled circle
        cv2.circle(canvas, (x_int, y_int), radius, color, -1)

    # Draw circle fills and outlines in theme color (no centers)
    if circles is not None and len(circles):
        for x, y, r in circles:
            x_int, y_int, r_int = int(x), int(y), int(r)
            if 0 <= x_int < canvas.shape[1] and 0 <= y_int < canvas.shape[0]:
                # Draw semi-transparent fill
                fill_layer = np.zeros_like(canvas)
                cv2.circle(fill_layer, (x_int, y_int), r_int, theme_rgb, -1)
                canvas = cv2.addWeighted(canvas, 1.0, fill_layer, 0.35, 0)
                # Draw outline on top
                cv2.circle(canvas, (x_int, y_int), r_int, theme_rgb, 4)

    return canvas


def render_local_sky_map(
    match: dict,
    latitude: float,
    longitude: float,
    star_color_mode: str = "Color by B–V",
) -> np.ndarray | None:
    """Render a location-aware sky map highlighting the matched constellation region."""

    if "stars" not in match:
        return None

    stars_df = match["stars"]
    if not {"ra", "dec"}.issubset(stars_df.columns):
        return None

    ra_vals = stars_df["ra"].to_numpy(dtype=float)
    dec_vals = stars_df["dec"].to_numpy(dtype=float)

    try:
        coords = SkyCoord(ra=ra_vals * u.deg, dec=dec_vals * u.deg, frame="icrs")
        location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg)
        obstime = Time(datetime.utcnow())
        altaz = coords.transform_to(AltAz(obstime=obstime, location=location))
    except Exception:
        return None

    alt = altaz.alt.deg
    az = altaz.az.deg

    valid = np.isfinite(alt) & np.isfinite(az)
    if not np.any(valid):
        return None

    # Prepare canvas (extra padding keeps labels visible)
    size = 520
    padding = 40
    center = (size // 2, size // 2)
    horizon_r = center[0] - padding
    canvas = np.full((size, size, 3), (6, 6, 14), dtype=np.uint8)

    # Horizon circle
    cv2.circle(canvas, center, horizon_r, (40, 40, 70), 2, cv2.LINE_AA)

    # Cardinal labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        "N",
        (center[0] - 10, center[1] - horizon_r + 18),
        font,
        0.8,
        (120, 140, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "S",
        (center[0] - 10, center[1] + horizon_r - 8),
        font,
        0.8,
        (120, 140, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "E",
        (center[0] + horizon_r - 6, center[1] + 6),
        font,
        0.8,
        (120, 140, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "W",
        (center[0] - horizon_r - 16, center[1] + 6),
        font,
        0.8,
        (120, 140, 255),
        2,
        cv2.LINE_AA,
    )

    # Only consider stars above/barely below horizon for plotting
    mask = valid & (alt > -5)
    if not np.any(mask):
        mask = valid

    alt = alt[mask]
    az = az[mask]

    magnitudes = None
    if "magnitude" in stars_df.columns:
        magnitudes = stars_df.loc[mask, "magnitude"].to_numpy(dtype=float)

    theme_rgb = get_theme_primary_rgb()
    per_star_colors = None
    if "b_v" in stars_df.columns:
        bvs = stars_df.loc[mask, "b_v"].to_numpy(dtype=float)
        per_star_colors = get_per_star_colors(bvs, theme_rgb)

    # Keep the mini-map uncluttered: plot only the brightest stars when many are visible
    max_plot_stars = 120
    if len(alt) > max_plot_stars:
        if magnitudes is not None and len(magnitudes) == len(alt):
            keep_idx = np.argsort(magnitudes)[:max_plot_stars]
        else:
            keep_idx = np.linspace(0, len(alt) - 1, max_plot_stars, dtype=int)

        alt = alt[keep_idx]
        az = az[keep_idx]
        if magnitudes is not None:
            magnitudes = magnitudes[keep_idx]
        if per_star_colors is not None:
            per_star_colors = per_star_colors[keep_idx]

    # Color resolution
    per_star_colors, fallback_color = resolve_star_color(
        star_color_mode, theme_rgb, per_star_colors
    )

    def to_xy(alt_deg: float, az_deg: float) -> tuple[int, int]:
        r = (90.0 - np.clip(alt_deg, 0.0, 90.0)) / 90.0 * horizon_r
        theta = np.deg2rad(az_deg - 90.0)
        x = int(center[0] + r * np.cos(theta))
        y = int(center[1] + r * np.sin(theta))
        return x, y

    # Draw stars
    for idx, (a, z) in enumerate(zip(alt, az)):
        x, y = to_xy(a, z)
        if not (0 <= x < size and 0 <= y < size):
            continue

        mag = (
            magnitudes[idx] if magnitudes is not None and idx < len(magnitudes) else 3.0
        )
        mag_clamped = max(0.0, min(6.0, mag))
        radius = max(2, min(10, int(12.0 * np.exp(-mag_clamped / 3.0))))

        if per_star_colors is not None:
            color = tuple(int(c) for c in per_star_colors[idx])
        else:
            color = fallback_color

        cv2.circle(canvas, (x, y), radius, color, -1)
        cv2.circle(canvas, (x, y), radius + 3, tuple(int(c * 0.5) for c in color), 1)

    # Highlight region bounding the matched constellation
    if len(alt) >= 2:
        min_alt, max_alt = float(np.min(alt)), float(np.max(alt))
        min_az, max_az = float(np.min(az)), float(np.max(az))

        corners = [
            to_xy(min_alt, min_az),
            to_xy(min_alt, max_az),
            to_xy(max_alt, max_az),
            to_xy(max_alt, min_az),
        ]
        overlay = np.zeros_like(canvas)
        cv2.fillPoly(overlay, [np.array(corners, dtype=np.int32)], (80, 120, 200))
        canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.15, 0)
        cv2.polylines(
            canvas,
            [np.array(corners, dtype=np.int32)],
            True,
            (160, 200, 255),
            2,
            cv2.LINE_AA,
        )

        # Mark approximate center
        center_alt = (min_alt + max_alt) / 2.0
        center_az = (min_az + max_az) / 2.0
        cx, cy = to_xy(center_alt, center_az)
        cv2.drawMarker(
            canvas, (cx, cy), (255, 220, 120), cv2.MARKER_STAR, 18, 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            "Image footprint",
            (cx - 70, cy - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 210, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Celebration Constellation",
        page_icon="★",
        layout="wide",
        initial_sidebar_state="auto",
    )

    # Load Font Awesome and compact styling
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
        .stButton > button {
            min-height: 44px;
            font-weight: 900;
            border-radius: 10px;
            font-family: "Font Awesome 6 Free", "Font Awesome 6 Brands", "Font Awesome 5 Free", "Segoe UI", system-ui, -apple-system, sans-serif;
        }
        .stButton > button[kind="primary"] {
            min-height: 52px;
        }
        .status-pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            background: #0f1116;
            border: 1px solid #223;
        }
        .status-pill.ready { color: #6fffe9; border-color: #3de0c6; }
        .status-pill.wait { color: #9ba3b4; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    initialize_session_state()

    st.title("Celebration Constellation")
    st.caption(
        "Upload an image to auto-run detection and matching. Adjust settings in the sidebar, then hit Run to reprocess."
    )

    # Sidebar upload and processing form
    with st.sidebar:
        st.header("Image")
        uploaded_file = st.file_uploader(
            "Upload a photo",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="uploader",
        )

        st.divider()
        st.header("Processing")
        with st.form("processing_form"):
            detection_scale = st.selectbox(
                "Object Size",
                options=["Auto", "Close-up", "Wide"],
                index=0,
                help="Expected size of circular objects in your photo. Auto analyzes the image; Close-up for large glasses/plates; Wide for distant or small objects",
            )

            quality_threshold = st.slider(
                "Detection Sensitivity",
                min_value=0.05,
                max_value=0.35,
                value=0.10,
                step=0.01,
                help="Lower finds more circles; higher is stricter.",
            )

            num_regions = st.slider(
                "Sky Regions",
                min_value=50,
                max_value=500,
                value=100,
                step=10,
                help="Higher = better coverage, slower.",
            )

            radius_deg = st.slider(
                "Sky Window (°)",
                min_value=10,
                max_value=40,
                value=20,
                step=1,
                help="Angular radius per sampled sky window.",
            )

            star_brightness = st.slider(
                "Star Size",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="Size multiplier for stars in overlays.",
            )

            run_clicked = st.form_submit_button(
                "Run", type="primary", use_container_width=True
            )

        st.caption("Changes apply when Run is clicked. Auto-runs once on upload.")

    # Live feed so users see stages as soon as they are ready
    live_placeholders: dict[str, Any] | None = None
    detection_progress_placeholder: Any | None = None
    if uploaded_file is not None or st.session_state.uploaded_image is not None:
        live_container = st.container()
        live_container.subheader("Live processing feed")
        live_container.caption(
            "Stages update as soon as they finish: upload → greyscale → detection."
        )
        live_cols = live_container.columns(3)
        live_placeholders = {
            "uploaded": live_cols[0].empty(),
            "background": live_cols[1].empty(),
            "detection": live_cols[2].empty(),
        }
        detection_progress_placeholder = live_container.empty()

    # Trigger pipeline: auto on first upload, manual on Run
    new_file_selected = uploaded_file is not None and (
        st.session_state.last_uploaded_name != getattr(uploaded_file, "name", None)
    )
    should_auto_run = new_file_selected
    if run_clicked and (uploaded_file is not None or st.session_state.uploaded_path):
        run_pipeline(
            uploaded_file if uploaded_file is not None else None,
            detection_scale=detection_scale,
            quality_threshold=quality_threshold,
            num_regions=num_regions,
            radius_deg=float(radius_deg),
            star_brightness=float(star_brightness),
            live_placeholders=live_placeholders,
            detection_progress_placeholder=detection_progress_placeholder,
        )
    elif should_auto_run:
        run_pipeline(
            uploaded_file,
            detection_scale=detection_scale,
            quality_threshold=quality_threshold,
            num_regions=num_regions,
            radius_deg=float(radius_deg),
            star_brightness=float(star_brightness),
            live_placeholders=live_placeholders,
            detection_progress_placeholder=detection_progress_placeholder,
        )

    # Status row
    if st.session_state.uploaded_image is not None:
        status_cols = st.columns(4)
        status_cols[0].markdown(
            f"<span class='status-pill {'ready' if st.session_state.bg_ready_dark or st.session_state.bg_ready_synth else 'wait'}'>BG</span>",
            unsafe_allow_html=True,
        )
        status_cols[1].markdown(
            f"<span class='status-pill {'ready' if st.session_state.circles_ready else 'wait'}'>Circles</span>",
            unsafe_allow_html=True,
        )
        status_cols[2].markdown(
            f"<span class='status-pill {'ready' if st.session_state.matches_ready else 'wait'}'>Matches</span>",
            unsafe_allow_html=True,
        )
        status_cols[3].markdown(
            f"<span class='status-pill ready'>Black bg</span>",
            unsafe_allow_html=True,
        )

    # Viewer and layer controls
    if st.session_state.uploaded_image is not None:
        st.subheader("Viewer")

        # Layer controls in main area
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1, 1, 1])
        available_backgrounds = st.session_state.active_backgrounds or ["Black"]
        background_mode = ctrl1.selectbox(
            "Background",
            options=available_backgrounds,
            index=(
                available_backgrounds.index(st.session_state.background_mode)
                if st.session_state.background_mode in available_backgrounds
                else 0
            ),
        )
        st.session_state.background_mode = background_mode

        show_circles_toggle = ctrl2.checkbox(
            "Circles",
            value=st.session_state.show_circles,
            disabled=not st.session_state.circles_ready,
        )
        st.session_state.show_circles = show_circles_toggle

        show_stars_toggle = ctrl3.checkbox(
            "Stars",
            value=st.session_state.show_stars,
            disabled=not st.session_state.matches_ready,
        )
        st.session_state.show_stars = show_stars_toggle

        show_constellation_lines = ctrl4.checkbox(
            "Lines",
            value=st.session_state.show_constellation_lines,
            disabled=not st.session_state.matches_ready,
        )
        st.session_state.show_constellation_lines = show_constellation_lines

        ctrl5, ctrl6 = st.columns([1, 1])
        star_color_mode = ctrl5.selectbox(
            "Star color",
            ["Color by B–V", "Theme blue", "White", "Neon pink"],
            index=["Color by B–V", "Theme blue", "White", "Neon pink"].index(
                st.session_state.star_color_mode
                if st.session_state.star_color_mode
                in [
                    "Color by B–V",
                    "Theme blue",
                    "White",
                    "Neon pink",
                ]
                else "Color by B–V"
            ),
            disabled=not st.session_state.matches_ready,
        )
        st.session_state.star_color_mode = star_color_mode

        star_brightness = ctrl6.slider(
            "Star size",
            min_value=0.5,
            max_value=3.0,
            value=float(st.session_state.get("star_brightness", 1.0)),
            step=0.1,
            disabled=not st.session_state.matches_ready,
        )
        st.session_state.star_brightness = star_brightness

        # Pick current match if ready
        active_match = None
        if st.session_state.matches_ready and st.session_state.matches:
            active_match = st.session_state.matches[
                st.session_state.current_match_index
            ]

        # Render main image
        if active_match is not None:
            unified = render_unified_view(
                active_match,
                brightness=float(star_brightness),
                star_color_mode=str(star_color_mode),
                background_mode=str(background_mode),
                show_stars=bool(show_stars_toggle),
                show_circles=bool(show_circles_toggle),
                show_lines=bool(show_constellation_lines),
            )
            st.image(unified, use_container_width=True)
        else:
            # Circles-only or background-only view
            base_bg = st.session_state.background_cache.get(background_mode)
            if base_bg is None and st.session_state.uploaded_image is not None:
                base_bg = build_background(
                    st.session_state.uploaded_image, mode=background_mode
                )
            canvas = (
                base_bg.copy()
                if base_bg is not None
                else np.zeros_like(st.session_state.uploaded_image)
            )
            if show_circles_toggle and st.session_state.circles_ready:
                canvas = draw_circles(
                    canvas,
                    st.session_state.circles,
                    show_circles=True,
                    show_centers=False,
                    circle_color=get_theme_primary_rgb(),
                    thickness=3,
                    fill_alpha=0.25,
                )
            st.image(canvas, use_container_width=True)

        # Navigation for matches
        if st.session_state.matches_ready and st.session_state.matches:
            st.divider()
            nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
            prev_disabled = st.session_state.current_match_index == 0
            next_disabled = (
                st.session_state.current_match_index
                >= len(st.session_state.matches) - 1
            )

            if nav_col1.button(
                "\uf060", disabled=prev_disabled, use_container_width=True
            ):
                st.session_state.current_match_index -= 1
                st.rerun()

            nav_col2.markdown(
                f"<div style='text-align: center; padding: 12px; font-weight: 700;'>{st.session_state.current_match_index + 1} / {len(st.session_state.matches)}</div>",
                unsafe_allow_html=True,
            )

            if nav_col3.button(
                "\uf061", disabled=next_disabled, use_container_width=True
            ):
                st.session_state.current_match_index += 1
                st.rerun()

            st.caption("Navigation buttons use Font Awesome icons only.")

            # Mini starmap with footprint
            st.subheader("Mini starmap")
            loc1, loc2 = st.columns(2)
            lat_val = loc1.number_input(
                "Latitude (°)",
                min_value=-90.0,
                max_value=90.0,
                value=0.0,
                step=0.1,
                format="%.2f",
            )
            lon_val = loc2.number_input(
                "Longitude (°)",
                min_value=-180.0,
                max_value=180.0,
                value=0.0,
                step=0.1,
                format="%.2f",
            )

            sky_map = render_local_sky_map(
                active_match,
                latitude=float(lat_val),
                longitude=float(lon_val),
                star_color_mode=str(st.session_state.star_color_mode),
            )
            if sky_map is not None:
                st.image(sky_map, use_container_width=True)

        # Reset button
        if st.button("Reset", use_container_width=True):
            reset_pipeline_state(clear_image=True)
            st.rerun()

    else:
        st.info("Upload an image to get started. The pipeline will run automatically.")


if __name__ == "__main__":
    main()
