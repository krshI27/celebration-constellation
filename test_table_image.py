#!/usr/bin/env python3
"""Test script to process the table image and show viewing locations."""

from pathlib import Path

from drinking_galaxies.astronomy import StarCatalog
from drinking_galaxies.detection import detect_and_extract
from drinking_galaxies.matching import match_to_sky_regions


def main():
    """Process table image and display results."""
    image_path = Path("data/input/table.jpg")

    print("🌌 Drinking Galaxies - Table Image Test")
    print("=" * 60)
    print()

    # Step 1: Detect circles with quality filtering
    print("Step 1: Detecting circular objects (with quality filtering)...")
    image, circles, centers = detect_and_extract(
        image_path,
        min_radius=20,
        max_radius=200,
        max_circles=50,  # Limit to 50 best circles
        quality_threshold=0.15,  # Minimum quality score (lowered)
    )

    if circles is None or len(circles) == 0:
        print("❌ No circular objects detected!")
        return

    print(f"✅ Detected {len(circles)} circular objects")
    print()

    # Step 2: Load star catalog and search sky regions
    print("Step 2: Searching the night sky...")
    catalog = StarCatalog()
    regions = catalog.search_sky_regions(num_samples=100)

    # Add stereographic projections
    for region in regions:
        positions = catalog.convert_to_stereographic(
            region["stars"], region["ra"], region["dec"]
        )
        region["positions"] = positions

    print(f"✅ Loaded {len(regions)} sky regions")
    print()

    # Step 3: Match to constellations
    print("Step 3: Matching to sky regions...")
    # Skip constellation identification (requires network)
    matches = match_to_sky_regions(centers, regions, identify_constellations=False)

    if not matches:
        print("❌ No matching constellations found!")
        return

    print(f"✅ Found {len(matches)} matching constellations")
    print()

    # Step 4: Display top 3 matches with viewing locations
    print("=" * 60)
    print("Top Matches with Viewing Locations:")
    print("=" * 60)
    print()

    for i, match in enumerate(matches[:3], 1):
        print(f"Match #{i}")
        print("-" * 60)

        # Constellation info
        if match.get("constellation_info"):
            info = match["constellation_info"]
            print(f"⭐ Constellation: {info['full_name']} ({info['abbrev']})")
            print(f"   Area: {info['area_sq_deg']} sq. degrees")
            print(f"   {info['description']}")
        else:
            print(f"⭐ Sky Region: RA={match['ra']:.1f}°, Dec={match['dec']:.1f}°")

        print()

        # Match quality
        print(f"📊 Match Score: {match['score']:.2f}")
        print(f"   Matching Stars: {match['num_inliers']}")
        print()

        # Viewing location information
        if "visibility" in match:
            vis = match["visibility"]

            print("📍 WHERE CAN YOU SEE THIS?")

            if vis["globally_visible"]:
                print("   ✨ Visible from anywhere on Earth!")
            else:
                print(f"   Visible from: {vis['min_latitude']:.1f}° to "
                      f"{vis['max_latitude']:.1f}° latitude")

                optimal = vis["optimal_latitude"]
                hemisphere = "N" if optimal >= 0 else "S"
                print(f"   Best viewing: {abs(optimal):.1f}°{hemisphere} "
                      "(highest in sky)")

            print()

            # Geographic regions
            if match.get("viewing_regions"):
                regions_str = ", ".join(match["viewing_regions"])
                print(f"   Regions: {regions_str}")

            print()

            # Example cities
            if match.get("example_cities"):
                print("   Example Cities:")
                for city in match["example_cities"][:5]:
                    lat_str = f"{abs(city['lat']):.1f}°"
                    lat_str += "N" if city["lat"] >= 0 else "S"
                    lon_str = f"{abs(city['lon']):.1f}°"
                    lon_str += "E" if city["lon"] >= 0 else "W"
                    print(f"   • {city['name']} ({lat_str}, {lon_str})")

            print()

            # Best viewing months
            if match.get("best_viewing_months"):
                months_str = ", ".join(match["best_viewing_months"])
                print(f"   🗓️  Best months: {months_str}")

        print()
        print()

    print("=" * 60)
    print("Test complete! ✅")
    print()


if __name__ == "__main__":
    main()
