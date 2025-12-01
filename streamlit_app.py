"""Streamlit web application for Drinking Galaxies.

This app provides an interactive interface for:
- Uploading table photos
- Detecting circular objects
- Matching to star constellations
- Swipeable image comparison between photo and sky
"""

from pathlib import Path

import numpy as np
import streamlit as st

from drinking_galaxies.astronomy import StarCatalog
from drinking_galaxies.detection import detect_and_extract, draw_circles
from drinking_galaxies.matching import match_to_sky_regions
from drinking_galaxies.visualization import (
    create_circles_on_stars,
    create_composite_overlay,
    create_constellation_visualization,
    normalize_positions_to_canvas,
)


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
    )

    return canvas


def render_composite_overlay(match: dict) -> np.ndarray:
    """Render composite overlay showing both circles and stars.

    Args:
        match: Match result dict with star positions, magnitudes, and transform

    Returns:
        RGB image with circles and stars overlaid
    """
    # Get circle centers
    circle_centers = st.session_state.centers

    # Get transformed star positions (already in image coordinates)
    star_positions = match["transformed_points"]

    # Get star magnitudes if available
    magnitudes = None
    if "stars" in match and "magnitude" in match["stars"].columns:
        magnitudes = match["stars"]["magnitude"].values[: len(star_positions)]

    # Create composite overlay
    overlay = create_composite_overlay(
        st.session_state.uploaded_image,
        circle_centers,
        star_positions,
        magnitudes=magnitudes,
        circle_color=(0, 255, 0),  # Green for circles
        star_color=(255, 255, 0),  # Yellow for stars
        alpha=0.7,  # Star transparency
    )

    return overlay


def render_circles_on_stars(match: dict) -> np.ndarray:
    """Render circle positions overlaid on star constellation pattern.

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

    # Create visualization with actual circle outlines
    canvas = create_circles_on_stars(
        image_shape,
        star_positions,
        circles,
        magnitudes=magnitudes,
        background_color=(20, 20, 50),
        star_color=(255, 255, 200),
        circle_color=(0, 255, 0),
    )

    return canvas


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Drinking Galaxies",
        page_icon="🌌",
        layout="wide",
    )

    initialize_session_state()

    st.title("🌌 Drinking Galaxies")
    st.markdown(
        "Find your table's hidden constellation! "
        "Upload a photo of bottles, plates, and glasses to discover "
        "which star pattern they match in the night sky."
    )

    # Sidebar for controls
    with st.sidebar:
        st.header("Detection Settings")

        min_radius = st.slider(
            "Min Circle Radius (px)",
            min_value=10,
            max_value=100,
            value=20,
            step=5,
        )

        max_radius = st.slider(
            "Max Circle Radius (px)",
            min_value=50,
            max_value=500,
            value=200,
            step=10,
        )

        max_circles = st.slider(
            "Max Circles",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
            help="Maximum number of circles to detect (higher quality circles preferred)",
        )

        quality_threshold = st.slider(
            "Quality Threshold",
            min_value=0.0,
            max_value=0.5,
            value=0.15,
            step=0.05,
            help="Minimum quality score (0.0-0.5). Higher = stricter filtering",
        )

        st.divider()

        st.header("Matching Settings")

        num_regions = st.slider(
            "Sky Regions to Search",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="More regions = better matches but slower",
        )

        st.divider()

        st.header("Visualization")

        show_circles = st.checkbox("Show Circles", value=True)
        show_centers = st.checkbox("Show Centers", value=True)

        st.session_state.show_circles = show_circles
        st.session_state.show_centers = show_centers

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a photo of your table",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        # Process image
        if st.session_state.uploaded_image is None:
            image, circles, centers = process_uploaded_image(
                uploaded_file,
                min_radius,
                max_radius,
                max_circles,
                quality_threshold,
            )

            if circles is None:
                st.error(
                    "No circular objects detected. Try adjusting the detection settings."
                )
                return

            st.success(f"✅ Detected {len(circles)} high-quality circular objects!")

        # Display detection results
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Your Table")
            if st.session_state.circles is not None:
                annotated_image = draw_circles(
                    st.session_state.uploaded_image,
                    st.session_state.circles,
                    show_circles=st.session_state.show_circles,
                    show_centers=st.session_state.show_centers,
                )
                st.image(annotated_image, use_container_width=True)

        with col2:
            st.subheader("Constellation Match")

            if st.session_state.centers is not None:
                if not st.session_state.matches:
                    if st.button("🔍 Find Matching Constellations"):
                        matches = find_constellation_matches(
                            st.session_state.centers, num_regions
                        )

                        if not matches:
                            st.warning(
                                "No matching constellations found. Try different detection settings."
                            )
                        else:
                            st.success(f"Found {len(matches)} matching constellations!")

                if st.session_state.matches:
                    # Display current match
                    match = st.session_state.matches[
                        st.session_state.current_match_index
                    ]

                    # Display constellation name if available
                    if match.get("constellation_info"):
                        info = match["constellation_info"]
                        st.markdown(f"### ⭐ {info['full_name']} ({info['abbrev']})")
                        st.caption(f"📐 Sky Area: {info['area_sq_deg']} sq. degrees")
                        st.info(info["description"])
                        st.divider()
                    elif match.get("constellation"):
                        st.markdown(f"### ⭐ Constellation: {match['constellation']}")
                        st.divider()

                    st.metric(
                        "Match Score",
                        f"{match['score']:.2f}",
                        help="Higher is better",
                    )

                    st.metric(
                        "Matching Stars",
                        match["num_inliers"],
                    )

                    st.metric(
                        "Sky Position",
                        f"RA: {match['ra']:.1f}°, Dec: {match['dec']:.1f}°",
                    )

                    # Verification section
                    st.divider()
                    st.markdown("### ✅ Verification")

                    # Basic star catalog info
                    if "stars" in match:
                        num_stars = len(match["stars"])
                        st.caption(
                            f"**{num_stars} catalog stars** (Yale Bright Star Catalog BSC5) in sampled region"
                        )
                        if "magnitude" in match["stars"].columns:
                            mags = match["stars"]["magnitude"].dropna()
                            if len(mags):
                                st.caption(
                                    f"Magnitude range: {mags.min():.1f} – {mags.max():.1f} (lower = brighter)"
                                )

                    # Residual error metrics (distance between transformed circles and nearest stars)
                    if "target_positions" in match:
                        tp = match["target_positions"]
                        transformed = match["transformed_points"]
                        if len(transformed) and len(tp):
                            # Compute distances
                            from scipy.spatial import distance_matrix

                            dists = distance_matrix(transformed, tp).min(axis=1)
                            mean_err = float(np.mean(dists))
                            max_err = float(np.max(dists))
                            st.caption(
                                f"Residual error (normalized units): mean {mean_err:.3f}, max {max_err:.3f}"
                            )
                            st.caption(
                                "Lower residuals indicate a tighter geometric fit to a real sky pattern."
                            )

                    # External verification links (use reliable sky map service)
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
                        f"🔭 **External sky map:** [In-The-Sky.org]({in_the_sky_url}) | [Wikisky]({wikisky_url})"
                    )
                    st.caption(
                        "Open one of the links to compare the star field at the matched RA/Dec with the pattern above."
                    )

                    # Copy-friendly coordinates
                    coord_str = f"RA {ra_deg:.3f}°  Dec {dec_deg:.3f}°"
                    st.text_input(
                        "Coordinates (copy for external tools)",
                        value=coord_str,
                        disabled=True,
                    )
                    st.caption(
                        "Paste RA/Dec into Stellarium desktop (Search → Position) for planetarium verification."
                    )

                    # Display viewing location information
                    st.divider()
                    st.markdown("### 📍 Where Can You See This?")

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
                            st.info(f"**Visible from:** {lat_range} latitude")

                            # Optimal viewing
                            optimal = vis["optimal_latitude"]
                            hemisphere = "N" if optimal >= 0 else "S"
                            st.caption(
                                f"**Best viewing:** {abs(optimal):.1f}°{hemisphere} "
                                "(constellation highest in sky)"
                            )

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
                        if (
                            "best_viewing_months" in match
                            and match["best_viewing_months"]
                        ):
                            months_str = ", ".join(match["best_viewing_months"])
                            st.markdown(f"🗓️ **Best months:** {months_str}")

                    st.divider()

                    # Advanced verification - show actual star data
                    with st.expander("🔬 Advanced: Show Star Data"):
                        st.caption(
                            "This table shows the actual stars from the Yale Bright Star Catalog "
                            "used in this match. Each star is a real celestial object with cataloged "
                            "coordinates and brightness."
                        )

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
                                    st.caption(
                                        f"Showing 20 of {len(match['stars'])} stars in this region"
                                    )

                            st.caption(
                                "**star_id**: Harvard Revised (HR) catalog number | "
                                "**ra/dec**: Sky coordinates in degrees | "
                                "**magnitude**: Brightness (lower = brighter)"
                            )

                    st.divider()

                    # Render visualizations with tabs
                    tab1, tab2, tab3 = st.tabs(
                        ["⭐ Star Pattern", "📸 Stars on Photo", "🎯 Circles on Stars"]
                    )

                    with tab1:
                        st.caption("Matched star constellation pattern")
                        sky_image = render_sky_visualization(match)
                        st.image(sky_image, use_container_width=True)

                    with tab2:
                        st.caption(
                            "Matched stars (bright yellow) overlaid on your photo with circle centers (green)"
                        )
                        composite_image = render_composite_overlay(match)
                        st.image(composite_image, use_container_width=True)

                    with tab3:
                        st.caption(
                            "Your detected circles (green) drawn on the star pattern"
                        )
                        circles_on_stars_image = render_circles_on_stars(match)
                        st.image(circles_on_stars_image, use_container_width=True)

                    # Navigation buttons
                    st.divider()
                    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

                    with nav_col1:
                        if (
                            st.button("⬅️ Previous")
                            and st.session_state.current_match_index > 0
                        ):
                            st.session_state.current_match_index -= 1
                            st.rerun()

                    with nav_col2:
                        st.write(
                            f"Match {st.session_state.current_match_index + 1} "
                            f"of {len(st.session_state.matches)}"
                        )

                    with nav_col3:
                        if (
                            st.button("Next ➡️")
                            and st.session_state.current_match_index
                            < len(st.session_state.matches) - 1
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
