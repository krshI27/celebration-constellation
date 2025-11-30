"""Streamlit web application for Drinking Galaxies.

This app provides an interactive interface for:
- Uploading table photos
- Detecting circular objects
- Matching to star constellations
- Swipeable image comparison between photo and sky
"""

from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

from drinking_galaxies.astronomy import StarCatalog
from drinking_galaxies.detection import detect_and_extract, draw_circles
from drinking_galaxies.matching import match_to_sky_regions


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


def process_uploaded_image(uploaded_file, min_radius: int, max_radius: int):
    """Process uploaded image and detect circles."""
    # Save uploaded file temporarily
    temp_path = Path("/tmp/uploaded_image.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Detect circles
    image, circles, centers = detect_and_extract(
        temp_path,
        min_radius=min_radius,
        max_radius=max_radius,
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
    """Render star constellation on black background.

    Args:
        match: Match result dict with star positions and transform

    Returns:
        RGB image with stars drawn
    """
    # Create black canvas matching uploaded image size
    image_shape = st.session_state.uploaded_image.shape
    canvas = np.zeros(image_shape, dtype=np.uint8)

    # Get transformed circle positions
    transformed = match["transformed_points"]

    # Scale to image dimensions
    # Normalize to [-1, 1] range first
    t_min = transformed.min(axis=0)
    t_max = transformed.max(axis=0)
    t_range = t_max - t_min

    if np.any(t_range < 1e-10):
        return canvas

    normalized = (transformed - t_min) / t_range

    # Scale to image dimensions with margin
    margin = 0.1
    scaled = normalized * (1 - 2 * margin) + margin
    scaled[:, 0] *= image_shape[1]  # Width
    scaled[:, 1] *= image_shape[0]  # Height

    # Draw stars at transformed positions
    star_color = (255, 255, 255)  # White stars
    for x, y in scaled.astype(int):
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            # Draw star as small circle
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx * dx + dy * dy <= 4:
                        px, py = x + dx, y + dy
                        if 0 <= px < image_shape[1] and 0 <= py < image_shape[0]:
                            canvas[py, px] = star_color

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
                uploaded_file, min_radius, max_radius
            )

            if circles is None:
                st.error("No circular objects detected. Try adjusting the detection settings.")
                return

            st.success(f"Detected {len(circles)} circular objects!")

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
                            st.warning("No matching constellations found. Try different detection settings.")
                        else:
                            st.success(f"Found {len(matches)} matching constellations!")

                if st.session_state.matches:
                    # Display current match
                    match = st.session_state.matches[st.session_state.current_match_index]

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

                    # Render sky visualization
                    sky_image = render_sky_visualization(match)
                    st.image(sky_image, use_container_width=True)

                    # Navigation buttons
                    st.divider()
                    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

                    with nav_col1:
                        if st.button("⬅️ Previous") and st.session_state.current_match_index > 0:
                            st.session_state.current_match_index -= 1
                            st.rerun()

                    with nav_col2:
                        st.write(
                            f"Match {st.session_state.current_match_index + 1} "
                            f"of {len(st.session_state.matches)}"
                        )

                    with nav_col3:
                        if st.button("Next ➡️") and st.session_state.current_match_index < len(st.session_state.matches) - 1:
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
