"""Streamlit web application for Drinking Galaxies.

This app provides an interactive interface for:
- Uploading table photos
- Detecting circular objects
- Matching to star constellations
- Swipeable image comparison between photo and sky
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from drinking_galaxies.astronomy import StarCatalog
from drinking_galaxies.detection import detect_and_extract, draw_circles
from drinking_galaxies.matching import match_to_sky_regions
from drinking_galaxies.visualization import (
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


def render_sky_visualization(match: dict) -> np.ndarray:
    """Render star constellation with magnitude-scaled stars.

    Args:
        match: Match result dict with star positions, magnitudes, and transform

    Returns:
        RGB image with enhanced star visualization
    """
    # Get image shape
    image_shape = st.session_state.uploaded_image.shape

    # Get transformed circle positions
    transformed = match["transformed_points"]

    # Normalize to canvas coordinates
    star_positions = normalize_positions_to_canvas(
        transformed, (image_shape[0], image_shape[1]), margin=0.1
    )

    # Get star magnitudes if available
    magnitudes = None
    if "stars" in match and "magnitude" in match["stars"].columns:
        magnitudes = match["stars"]["magnitude"].values[: len(star_positions)]

    # Create visualization with magnitude-scaled stars (brighter background)
    canvas = create_constellation_visualization(
        image_shape,
        star_positions,
        magnitudes=magnitudes,
        line_segments=None,  # TODO: Add constellation lines in future
        background_color=(20, 20, 50),  # Slightly brighter dark blue background
        star_color=(255, 255, 200),  # Warm white/yellow for better visibility
        draw_lines=False,
        star_scale_factor=16.0,
    )

    return canvas


def render_composite_overlay(match: dict) -> np.ndarray:
    """Render composite overlay showing both circles and stars.

    Uses photometric stretching and per-star colors based on B-V temperature.

    Args:
        match: Match result dict with star positions, magnitudes, and transform

    Returns:
        RGB image with circles and stars overlaid
    """
    # Get transformed star positions (already in image coordinates)
    star_positions = match["transformed_points"]

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
        # Map B-V to RGB: blue (hot) to red (cool)
        per_star_colors = []
        for bv in bvs:
            if np.isnan(bv):
                per_star_colors.append(theme_rgb)
            elif bv <= 0.0:
                per_star_colors.append((180, 200, 255))
            elif bv <= 0.3:
                per_star_colors.append((220, 230, 255))
            elif bv <= 0.6:
                per_star_colors.append((255, 255, 255))
            elif bv <= 1.0:
                per_star_colors.append((255, 235, 180))
            else:
                per_star_colors.append((255, 210, 150))
        per_star_colors = np.array(per_star_colors, dtype=np.uint8)

    # Build overlay with per-star colors if available
    if per_star_colors is not None:
        overlay = gray_base.copy()
        star_layer = np.zeros_like(gray_base)

        # Render stars with individual colors and magnitude scaling
        for idx, ((x, y), mag, color) in enumerate(
            zip(
                star_positions,
                magnitudes if magnitudes is not None else [3.0] * len(star_positions),
                per_star_colors,
            )
        ):
            from drinking_galaxies.visualization import magnitude_to_radius

            radius = magnitude_to_radius(mag, 24.0)
            x_int, y_int = int(x), int(y)

            if 0 <= x_int < star_layer.shape[1] and 0 <= y_int < star_layer.shape[0]:
                # Draw filled circle with per-star color
                cv2.circle(
                    star_layer, (x_int, y_int), radius, tuple(int(c) for c in color), -1
                )

                # Add glow effect
                glow_radius = radius + 4
                glow_color = tuple(int(c * 0.6) for c in color)
                cv2.circle(star_layer, (x_int, y_int), glow_radius, glow_color, 2)

        # Apply Gaussian blur for natural PSF-like glow
        star_layer = cv2.GaussianBlur(star_layer, (7, 7), 2.0)

        # Blend with full opacity
        overlay = cv2.addWeighted(gray_base, 1.0, star_layer, 1.0, 0)
    else:
        # Fall back to single-color rendering
        overlay = create_composite_overlay(
            gray_base,
            empty_centers,
            star_positions,
            magnitudes=magnitudes,
            circle_color=theme_rgb,
            star_color=theme_rgb,
            alpha=1.0,
            star_scale_factor=24.0,
        )

    return overlay


def render_circles_on_stars(match: dict) -> np.ndarray:
    """Render circle positions overlaid on star constellation pattern.

    Uses per-star colors based on B-V temperature index.

    Args:
        match: Match result dict with star positions, magnitudes, and transform

    Returns:
        RGB image with circles drawn on star pattern
    """
    # Get image shape
    image_shape = st.session_state.uploaded_image.shape

    # Get transformed star positions
    transformed = match["transformed_points"]

    # Normalize star positions to canvas coordinates
    star_positions = normalize_positions_to_canvas(
        transformed, (image_shape[0], image_shape[1]), margin=0.1
    )

    # Get detected circles (x, y, radius) - already in image coordinates
    circles = st.session_state.circles

    # Get star magnitudes if available
    magnitudes = None
    if "stars" in match and "magnitude" in match["stars"].columns:
        magnitudes = match["stars"]["magnitude"].values[: len(star_positions)]

    # Theme color for circle outlines
    theme_rgb = get_theme_primary_rgb()

    # Build per-star colors from B-V if available
    per_star_colors = None
    if "stars" in match and "b_v" in match["stars"].columns:
        bvs = match["stars"]["b_v"].values[: len(star_positions)]
        per_star_colors = []
        for bv in bvs:
            if np.isnan(bv):
                per_star_colors.append((255, 255, 200))
            elif bv <= 0.0:
                per_star_colors.append((180, 200, 255))
            elif bv <= 0.3:
                per_star_colors.append((220, 230, 255))
            elif bv <= 0.6:
                per_star_colors.append((255, 255, 255))
            elif bv <= 1.0:
                per_star_colors.append((255, 235, 180))
            else:
                per_star_colors.append((255, 210, 150))
        per_star_colors = np.array(per_star_colors, dtype=np.uint8)

    # Create star-only visualization first (use per-star colors if available)
    canvas = create_constellation_visualization(
        image_shape,
        star_positions,
        magnitudes=magnitudes,
        line_segments=None,
        background_color=(20, 20, 50),
        star_color=(
            per_star_colors if per_star_colors is not None else (255, 255, 200)
        ),
        draw_lines=False,
        star_scale_factor=24.0,
    )

    # Draw only circle outlines in theme color (no centers)
    if circles is not None and len(circles):
        for x, y, r in circles:
            x_int, y_int, r_int = int(x), int(y), int(r)
            if 0 <= x_int < canvas.shape[1] and 0 <= y_int < canvas.shape[0]:
                cv2.circle(canvas, (x_int, y_int), r_int, theme_rgb, 4)

    return canvas


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Drinking Galaxies",
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

    st.title("🌌 Drinking Galaxies")
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

        num_regions = {"Fast": 50, "Balanced": 100, "Thorough": 200}[search_depth]
        if num_regions > 100:
            st.caption(
                "⏱️ Estimated time: 30–90 seconds"
            )  # Minimal derived detection parameters (auto-tuned defaults)
    quality_threshold = 0.15
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
                from PIL import Image

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
            # Apply enhanced rendering if enabled
            if enhanced_render:
                base = apply_photometric_stretch(st.session_state.uploaded_image)
            else:
                base = to_gray_rgb(st.session_state.uploaded_image)

            if st.session_state.show_circles:
                annotated_image = draw_circles(
                    base,
                    st.session_state.circles,
                    show_circles=True,
                    show_centers=False,
                    circle_color=get_theme_primary_rgb(),
                    thickness=4,
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
            if "target_positions" in match:
                from scipy.spatial import distance_matrix

                tp = match["target_positions"]
                transformed = match["transformed_points"]
                if len(transformed) and len(tp):
                    dists = distance_matrix(transformed, tp).min(axis=1)
                    mean_err = float(np.mean(dists))
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
                if "target_positions" in match:
                    tp = match["target_positions"]
                    transformed = match["transformed_points"]
                    if len(transformed) and len(tp):
                        from scipy.spatial import distance_matrix

                        dists = distance_matrix(transformed, tp).min(axis=1)
                        mean_err = float(np.mean(dists))
                        max_err = float(np.max(dists))
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
                sky_image = render_sky_visualization(match)
                st.image(sky_image, use_container_width=True)

            with tab3:
                st.caption("Detected circles on star pattern (colored by temperature)")
                circles_on_stars_image = render_circles_on_stars(match)
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
        if st.button("🔄 Upload New Image"):
            st.session_state.uploaded_image = None
            st.session_state.circles = None
            st.session_state.centers = None
            st.session_state.matches = []
            st.rerun()

    else:
        st.info("👆 Upload an image to get started!")


if __name__ == "__main__":
    main()
