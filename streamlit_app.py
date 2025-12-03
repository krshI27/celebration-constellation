"""Streamlit web application for Celebration Constellation.

This app provides an interactive interface for:
- Uploading table photos
- Detecting circular objects
- Matching to star constellations
- Swipeable image comparison between photo and sky
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path for Streamlit Cloud deployment
# This ensures the celebration_constellation package can be imported
_src_path = Path(__file__).parent / "src"
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Configure OpenCV for headless environments (Streamlit Cloud)
# Prevents "libGL.so.1: cannot open shared object file" errors
os.environ["LIBGL_ALWAYS_INDIRECT"] = "1"

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from celebration_constellation.astronomy import StarCatalog
from celebration_constellation.detection import detect_and_extract, draw_circles
from celebration_constellation.matching import match_to_sky_regions
from celebration_constellation.visualization import (
    create_composite_overlay,
    create_constellation_visualization,
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

    if "circles" not in st.session_state:
        st.session_state.circles = None

    if "centers" not in st.session_state:
        st.session_state.centers = None

    if "matches" not in st.session_state:
        st.session_state.matches = []

    if "current_match_index" not in st.session_state:
        st.session_state.current_match_index = 0

    if "show_circles" not in st.session_state:
        st.session_state.show_circles = True

    if "show_centers" not in st.session_state:
        st.session_state.show_centers = True

    if "star_catalog" not in st.session_state:
        st.session_state.star_catalog = StarCatalog()


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


def find_constellation_matches(centers: np.ndarray, num_regions: int = 100):
    """Find matching star constellations."""
    catalog = st.session_state.star_catalog

    with st.spinner("Searching the night sky for matching constellations..."):
        # Sample sky regions
        regions = catalog.search_sky_regions(num_samples=num_regions)

        # Add stereographic projections
        for region in regions:
            positions = catalog.convert_to_stereographic(
                region["stars"], region["ra"], region["dec"]
            )
            region["positions"] = positions

        # Match circle centers to sky regions
        matches = match_to_sky_regions(centers, regions)

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
    # If B-V color index present, map to per-star RGB colors
    per_star_colors = None
    if "stars" in match and "b_v" in match["stars"].columns:
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


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Celebration Constellation",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="auto",
    )

    # Add PWA meta tags and custom CSS for mobile-first design
    st.markdown(
        """
        <!-- PWA Meta Tags -->
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
        <meta name="theme-color" content="#00D9FF">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <meta name="apple-mobile-web-app-title" content="Galaxies">
        <link rel="manifest" href="/.streamlit/manifest.json">
        
        <style>
        /* Button styling for larger tap targets */
        .stButton > button {
            min-height: 48px;
            font-weight: 600;
            border-radius: 8px;
        }
        
        /* Primary CTA button emphasis */
        .stButton > button[kind="primary"] {
            min-height: 56px;
            font-size: 1.1rem;
        }
        
        /* Bottom navigation bar styling */
        .nav-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1rem;
            padding: 1rem 0;
            margin-top: 1rem;
        }
        
        /* Lighter dividers */
        hr {
            margin: 1.5rem 0;
            opacity: 0.3;
        }
        
        /* Compact metric styling */
        [data-testid="stMetricValue"] {
            font-size: 1.2rem;
        }
        
        /* Image container constraints for mobile */
        @media (max-width: 768px) {
            .stImage {
                max-height: 60vh;
            }
        }
        
        /* Tab label simplification */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        </style>
        
        <script>
        // Register service worker for PWA functionality
        if ('serviceWorker' in navigator) {
          window.addEventListener('load', () => {
            navigator.serviceWorker.register('/.streamlit/service-worker.js')
              .then((registration) => {
                console.log('ServiceWorker registered:', registration);
              })
              .catch((error) => {
                console.log('ServiceWorker registration failed:', error);
              });
          });
        }
        </script>
        """,
        unsafe_allow_html=True,
    )

    initialize_session_state()

    st.title("🌌 Celebration Constellation")
    st.markdown(
        "Upload a photo of your table to discover which star constellation matches the pattern!"
    )

    # Simplified sidebar - essential controls only
    with st.sidebar:
        st.header("⚙️ Settings")

        detection_scale = st.selectbox(
            "Detection Scale",
            options=["Auto", "Close-up", "Wide"],
            index=0,
            help="Auto-tunes circle sizes; choose Close-up for large objects or Wide for small/far objects",
        )

        search_depth = st.selectbox(
            "Search Depth",
            options=["Fast", "Balanced", "Thorough"],
            index=1,
            help="Fast=50 regions, Balanced=100, Thorough=200 (slower)",
        )

        quality_threshold = st.slider(
            "Detection Quality",
            min_value=0.05,
            max_value=0.35,
            value=0.10,
            step=0.01,
            help="Lower = more detections (catches more, may have false positives). Higher = stricter (fewer false positives, may miss some)",
        )

        star_brightness = st.slider(
            "Star Brightness",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Adjust the size of stars in the visualization (1.0 = normal)",
        )

        num_regions = {"Fast": 50, "Balanced": 100, "Thorough": 200}[search_depth]
        if num_regions > 100:
            st.caption(
                "⏱️ Estimated time: 30–90 seconds"
            )  # Minimal derived detection parameters (auto-tuned defaults)
    max_circles = 50

    # Coarse min/max radii based on scale; Auto estimates from image
    if detection_scale == "Close-up":
        min_radius, max_radius = 40, 150
    elif detection_scale == "Wide":
        min_radius, max_radius = 15, 250
    else:
        min_radius, max_radius = 20, 200

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a photo of your table",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        # Process image
        if st.session_state.uploaded_image is None:
            with st.spinner("Detecting circular objects..."):
                # Load image first for auto-tuning
                pil_image = Image.open(uploaded_file)
                temp_image = np.array(pil_image.convert("RGB"))

                # Auto-tune radii if needed
                if detection_scale == "Auto":
                    with st.spinner("Auto-tuning detection parameters..."):
                        min_radius, max_radius = estimate_radius_bounds(temp_image)
                        st.caption(
                            f"🔍 Auto-detected range: {min_radius}–{max_radius}px"
                        )

                uploaded_file.seek(0)  # Reset file pointer
                image, circles, centers = process_uploaded_image(
                    uploaded_file,
                    min_radius,
                    max_radius,
                    max_circles,
                    quality_threshold,
                )

            if circles is None:
                st.error(
                    "❌ No circular objects detected. Try adjusting radius settings in the sidebar."
                )
                return

            st.success(f"✅ Detected {len(circles)} circles")

        # Show uploaded image with detection overlay
        st.subheader("📸 Your Table")

        # Visualization toggle (centers hidden by design)
        show_circles = st.checkbox("Show Circles", value=True)
        st.session_state.show_circles = show_circles
        st.session_state.show_centers = False

        if st.session_state.circles is not None:
            # Apply photometric stretching for better contrast
            base = apply_photometric_stretch(st.session_state.uploaded_image)

            if st.session_state.show_circles:
                annotated_image = draw_circles(
                    base,
                    st.session_state.circles,
                    show_circles=True,
                    show_centers=False,
                    circle_color=get_theme_primary_rgb(),
                    thickness=4,
                    fill_alpha=0.35,
                )
            else:
                annotated_image = base
            st.image(annotated_image, use_container_width=True)

        st.divider()

        # Primary CTA for matching (prominent placement)
        if st.session_state.centers is not None and not st.session_state.matches:
            st.markdown("### 🔍 Ready to find your constellation?")
            if st.button(
                "Find Matching Constellations",
                type="primary",
                use_container_width=True,
            ):
                matches = find_constellation_matches(
                    st.session_state.centers, num_regions
                )

                if not matches:
                    st.warning(
                        "No matches found. Try increasing Sky Regions in the sidebar."
                    )
                else:
                    st.success(f"✨ Found {len(matches)} matches!")

        # Display matches
        if st.session_state.matches:
            match = st.session_state.matches[st.session_state.current_match_index]

            # Constellation name
            if match.get("constellation_info"):
                info = match["constellation_info"]
                st.markdown(f"## ⭐ {info['full_name']} ({info['abbrev']})")
                st.caption(f"📐 {info['area_sq_deg']} sq. degrees")
                st.info(info["description"])
            elif match.get("constellation"):
                st.markdown(f"## ⭐ {match['constellation']}")

            st.divider()

            # Compact metrics in one row
            met_col1, met_col2, met_col3 = st.columns(3)
            with met_col1:
                st.metric("Score", f"{match['score']:.2f}")
            with met_col2:
                st.metric("Stars", match["num_inliers"])
            with met_col3:
                st.metric("Position", f"{match['ra']:.0f}°, {match['dec']:.0f}°")

            st.divider()

            # Verification section - collapsed by default
            residual_summary = ""
            mean_err = None
            max_err = None
            if "target_positions" in match:
                from scipy.spatial import distance_matrix

                tp = match["target_positions"]
                transformed = match["transformed_points"]
                if len(transformed) and len(tp):
                    dists = distance_matrix(transformed, tp).min(axis=1)
                    mean_err = float(np.mean(dists))
                    max_err = float(np.max(dists))
                    residual_summary = f"Mean residual: {mean_err:.3f}"

            with st.expander(
                f"✅ Verification {f'({residual_summary})' if residual_summary else ''}"
            ):
                # Basic star catalog info
                if "stars" in match:
                    num_stars = len(match["stars"])
                    st.caption(
                        f"**{num_stars} catalog stars** from Yale Bright Star Catalog"
                    )
                    if "magnitude" in match["stars"].columns:
                        mags = match["stars"]["magnitude"].dropna()
                        if len(mags):
                            st.caption(
                                f"Magnitude: {mags.min():.1f} – {mags.max():.1f}"
                            )

                # Residual error details
                if mean_err is not None and max_err is not None:
                    st.caption(
                        f"Residual error: mean {mean_err:.3f}, max {max_err:.3f} (lower = better fit)"
                    )

                # External verification links
                ra_deg = match["ra"] % 360.0
                dec_deg = match["dec"]
                in_the_sky_url = (
                    "https://in-the-sky.org/skymap.php?"
                    f"ra={ra_deg:.3f}&dec={dec_deg:.3f}&zoom=2"
                )
                wikisky_url = (
                    "https://server1.wikisky.org/v2?"
                    f"ra={ra_deg:.3f}&de={dec_deg:.3f}&zoom=4&show_grid=1&show_constellation_lines=1"
                )
                st.markdown(
                    f"🔭 [In-The-Sky.org]({in_the_sky_url}) · [Wikisky]({wikisky_url})"
                )

                # Copy-friendly coordinates
                coord_str = f"RA {ra_deg:.3f}°  Dec {dec_deg:.3f}°"
                st.text_input(
                    "Coordinates",
                    value=coord_str,
                    disabled=True,
                    label_visibility="collapsed",
                )

            # Visibility section - one-line summary, details in expander
            visibility_summary = ""
            if "visibility" in match:
                vis = match["visibility"]
                if vis["globally_visible"]:
                    visibility_summary = "Visible globally"
                else:
                    visibility_summary = f"Visible {vis['min_latitude']:.0f}° to {vis['max_latitude']:.0f}°"

            with st.expander(
                f"📍 Where to See This {f'({visibility_summary})' if visibility_summary else ''}"
            ):
                if "visibility" in match:
                    vis = match["visibility"]

                    # Latitude range
                    if vis["globally_visible"]:
                        st.success("✨ Visible from anywhere on Earth!")
                    else:
                        lat_range = (
                            f"{vis['min_latitude']:.1f}° to "
                            f"{vis['max_latitude']:.1f}°"
                        )
                        st.info(f"**Visible:** {lat_range} latitude")

                        # Optimal viewing
                        optimal = vis["optimal_latitude"]
                        hemisphere = "N" if optimal >= 0 else "S"
                        st.caption(f"**Best:** {abs(optimal):.1f}°{hemisphere}")

                    # Geographic regions
                    if "viewing_regions" in match and match["viewing_regions"]:
                        regions_str = ", ".join(match["viewing_regions"])
                        st.markdown(f"**Regions:** {regions_str}")

                    # Example cities
                    if "example_cities" in match and match["example_cities"]:
                        st.markdown("**Example Cities:**")
                        for city in match["example_cities"][:5]:
                            lat_str = f"{abs(city['lat']):.1f}°"
                            lat_str += "N" if city["lat"] >= 0 else "S"
                            lon_str = f"{abs(city['lon']):.1f}°"
                            lon_str += "E" if city["lon"] >= 0 else "W"
                            st.markdown(f"• {city['name']} ({lat_str}, {lon_str})")

                    # Best viewing months
                    if "best_viewing_months" in match and match["best_viewing_months"]:
                        months_str = ", ".join(match["best_viewing_months"])
                        st.markdown(f"🗓️ **Best months:** {months_str}")

            # Advanced star data
            with st.expander("🔬 Star Data"):
                if "stars" in match:
                    # Show star table with key columns
                    star_display = match["stars"].copy()

                    # Select relevant columns
                    display_cols = []
                    if "star_id" in star_display.columns:
                        display_cols.append("star_id")
                    if "ra" in star_display.columns:
                        display_cols.append("ra")
                    if "dec" in star_display.columns:
                        display_cols.append("dec")
                    if "magnitude" in star_display.columns:
                        display_cols.append("magnitude")

                    if display_cols:
                        star_display = star_display[display_cols].head(20)
                        st.dataframe(
                            star_display,
                            use_container_width=True,
                            hide_index=True,
                        )

                        if len(match["stars"]) > 20:
                            st.caption(f"Showing 20 of {len(match['stars'])} stars")

            st.divider()

            # Render visualizations with tabs (reordered, default to Overlay)
            tab1, tab2, tab3 = st.tabs(["📸 Overlay", "⭐ Pattern", "🎯 Circles"])

            with tab1:
                st.caption("Stars overlaid on your photo (colored by temperature)")
                composite_image = render_composite_overlay(match)
                st.image(composite_image, use_container_width=True)

            with tab2:
                st.caption("Matched star constellation pattern")
                sky_image = render_sky_visualization(match, star_brightness)
                st.image(sky_image, use_container_width=True)

            with tab3:
                st.caption("Detected circles on star pattern (colored by temperature)")
                circles_on_stars_image = render_circles_on_stars(match, star_brightness)
                st.image(circles_on_stars_image, use_container_width=True)

            # Bottom-centered navigation
            st.divider()

            nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])

            with nav_col1:
                prev_disabled = st.session_state.current_match_index == 0
                if st.button(
                    "⬅️ Prev",
                    disabled=prev_disabled,
                    use_container_width=True,
                ):
                    st.session_state.current_match_index -= 1
                    st.rerun()

            with nav_col2:
                st.markdown(
                    f"<div style='text-align: center; padding: 12px;'>"
                    f"<strong>{st.session_state.current_match_index + 1} of {len(st.session_state.matches)}</strong>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with nav_col3:
                next_disabled = (
                    st.session_state.current_match_index
                    >= len(st.session_state.matches) - 1
                )
                if st.button(
                    "Next ➡️",
                    disabled=next_disabled,
                    use_container_width=True,
                ):
                    st.session_state.current_match_index += 1
                    st.rerun()

        # Reset button
        if st.button("🔄 Start Over"):
            st.session_state.uploaded_image = None
            st.session_state.circles = None
            st.session_state.centers = None
            st.session_state.matches = []
            st.rerun()

    else:
        st.info("👆 Upload an image to get started!")


if __name__ == "__main__":
    main()
