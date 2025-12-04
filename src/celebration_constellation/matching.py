"""RANSAC-based point cloud matching for constellation identification.

This module implements robust point pattern matching using RANSAC to find
the best-fitting star constellation for detected circle centers, with
correction for star density to avoid bias toward dense regions.

Includes smart point sampling to handle large datasets efficiently.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import distance_matrix

from celebration_constellation.constellations import ConstellationCatalog
from celebration_constellation.visibility import (
    calculate_best_viewing_months,
    calculate_visibility_range,
    get_example_cities,
    get_viewing_regions,
)


def sample_points_spatially(
    points: np.ndarray,
    max_points: int = 50,
) -> np.ndarray:
    """Sample points to preserve spatial distribution.

    Uses grid-based sampling to ensure good coverage of the pattern.

    Args:
        points: Point coordinates (N, 2)
        max_points: Maximum points to return

    Returns:
        Sampled points with preserved spatial distribution
    """
    if len(points) <= max_points:
        return points

    # Create a grid over the point cloud
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    ranges = max_coords - min_coords

    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(max_points)))

    # Assign points to grid cells
    cell_indices = ((points - min_coords) / ranges * (grid_size - 1)).astype(int)

    # Sample one point from each occupied cell
    sampled = []
    cell_dict = {}

    for i, cell in enumerate(cell_indices):
        cell_key = tuple(cell)
        if cell_key not in cell_dict:
            cell_dict[cell_key] = i
            sampled.append(i)

    # If we still have too many, randomly sample
    if len(sampled) > max_points:
        sampled = np.random.choice(sampled, max_points, replace=False)

    return points[sampled]


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Normalize point cloud to zero mean and unit scale.

    Args:
        points: Array of shape (N, 2) with point coordinates

    Returns:
        Tuple of (normalized_points, scale, center)
    """
    center = np.mean(points, axis=0)
    centered = points - center
    scale = np.std(centered)

    if scale < 1e-10:
        scale = 1.0

    normalized = centered / scale

    return normalized, scale, center


def estimate_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Estimate 2D similarity transform (scale, rotation, translation).

    Uses least squares to find best-fit transformation from source to target.

    Args:
        source: Source points (N, 2)
        target: Target points (N, 2)

    Returns:
        Tuple of (scale, rotation_angle_rad, translation)
    """
    # Normalize both point sets
    src_norm, src_scale, src_center = normalize_points(source)
    tgt_norm, tgt_scale, tgt_center = normalize_points(target)

    # Compute rotation using SVD
    H = src_norm.T @ tgt_norm
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Extract rotation angle
    rotation_angle = np.arctan2(R[1, 0], R[0, 0])

    # Compute scale
    scale = tgt_scale / src_scale

    # Compute translation
    translation = tgt_center - scale * (R @ src_center)

    return scale, rotation_angle, translation


def apply_transform(
    points: np.ndarray,
    scale: float,
    rotation: float,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply 2D similarity transform to points.

    Args:
        points: Points to transform (N, 2)
        scale: Scale factor
        rotation: Rotation angle in radians
        translation: Translation vector (2,)

    Returns:
        Transformed points
    """
    R = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]
    )

    return scale * (points @ R.T) + translation


def compute_match_score(
    source: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.1,
) -> tuple[float, int]:
    """Compute matching score between two point sets with proportion penalty.

    Score formula:
        score = num_inliers × min(1.0, len(source) / len(target))

    Proportion penalty rationale:
    Without penalty, dense star regions (Milky Way, Pleiades) would always win due to
    higher probability of random inlier matches. The penalty ensures score reflects
    pattern quality, not star density.

    Examples:
    - 10 circles, 10 stars: penalty = min(1.0, 10/10) = 1.0 (no penalty)
    - 10 circles, 100 stars: penalty = min(1.0, 10/100) = 0.1 (strong penalty)
    - 50 circles, 20 stars: penalty = min(1.0, 50/20) = 1.0 (no penalty, source > target)

    This prevents bias toward dense regions while preserving scores when point counts match.

    Args:
        source: Source points (N, 2)
        target: Target points (M, 2)
        threshold: Distance threshold for considering points matched (default: 0.1)

    Returns:
        Tuple of (score, num_matches)
        score: Proportion-corrected matching score
        num_matches: Raw count of inliers within threshold
    """
    if len(source) == 0 or len(target) == 0:
        return 0.0, 0

    # Compute pairwise distances
    distances = distance_matrix(source, target)

    # Find closest target point for each source point
    min_distances = np.min(distances, axis=1)

    # Count matches within threshold
    num_matches = np.sum(min_distances < threshold)

    # Compute proportion-corrected score
    # Penalize if number of target points >> number of source points
    proportion_penalty = min(1.0, len(source) / max(len(target), 1))

    score = num_matches * proportion_penalty

    return score, num_matches


def ransac_match(
    source: np.ndarray,
    target: np.ndarray,
    num_iterations: int = 1000,
    sample_size: int = 3,
    inlier_threshold: float = 0.1,
    min_inliers: int = 3,
) -> Optional[dict]:
    """RANSAC-based point pattern matching with 2D similarity transform.

    RANSAC parameters and rationale:
    - num_iterations=1000: Provides 99% confidence for 3-point sampling with 50% inlier ratio
      Confidence calculation: 1 - (1 - p^n)^k where p=0.5, n=3, k=1000
      (Based on Fischler & Bolles 1981 RANSAC paper)
    - sample_size=3: Minimum points required for 2D similarity transform
      (scale, rotation, translation = 4 DOF, 3 point pairs = 6 constraints)
    - inlier_threshold=0.1: Distance threshold in normalized coordinate space
      In stereographic projection: 0.1 ≈ 0.1° angular separation
      Lower threshold: Stricter matching, fewer false positives
      Higher threshold: More lenient, may match random clusters
    - min_inliers=3: Minimum pattern complexity for valid match

    Performance:
    - Per region: ~0.5 seconds (1000 iterations × transform estimation)
    - Total (100 regions): 30-60 seconds with progress feedback

    Args:
        source: Source points (circle centers) (N, 2)
        target: Target points (star positions) (M, 2)
        num_iterations: Number of RANSAC iterations (default: 1000)
        sample_size: Number of points to sample for hypothesis (default: 3)
        inlier_threshold: Distance threshold for inliers (default: 0.1 in normalized space)
        min_inliers: Minimum number of inliers required (default: 3)

    Returns:
        Dict with transform parameters and score, or None if no match found
        Dict keys: 'scale', 'rotation', 'translation', 'score', 'num_inliers', 'transformed_points'

    Example:
        >>> circles = np.array([[100, 150], [200, 180], [150, 250]])
        >>> stars = np.array([[0.5, 0.8], [1.2, 0.9], [0.8, 1.5]])
        >>> result = ransac_match(circles, stars, num_iterations=1000, inlier_threshold=0.05)
        >>> if result:
        ...     print(f"Found {result['num_inliers']} matching points")
        ...     print(f"Match score: {result['score']:.2f}")
        ...     print(f"Scale: {result['scale']:.2f}, Rotation: {np.degrees(result['rotation']):.1f}°")
    """
    if len(source) < sample_size or len(target) < sample_size:
        return None

    best_score = 0
    best_result = None

    for _ in range(num_iterations):
        # Randomly sample points from source
        sample_indices = np.random.choice(len(source), sample_size, replace=False)
        source_sample = source[sample_indices]

        # Randomly sample corresponding points from target
        target_indices = np.random.choice(len(target), sample_size, replace=False)
        target_sample = target[target_indices]

        try:
            # Estimate transform from samples
            scale, rotation, translation = estimate_similarity_transform(
                source_sample, target_sample
            )

            # Apply transform to all source points
            transformed = apply_transform(source, scale, rotation, translation)

            # Compute score
            score, num_matches = compute_match_score(
                transformed, target, threshold=inlier_threshold
            )

            # Update best match
            if score > best_score and num_matches >= min_inliers:
                best_score = score
                # Store additional target positions for verification (residual computation)
                best_result = {
                    "scale": scale,
                    "rotation": rotation,
                    "translation": translation,
                    "score": score,
                    "num_inliers": num_matches,
                    "transformed_points": transformed,
                    "target_positions": target,
                }

        except (np.linalg.LinAlgError, ValueError):
            # Skip if transform estimation fails
            continue

    return best_result


def match_to_sky_regions(
    circle_centers: np.ndarray,
    sky_regions: list[dict],
    num_iterations: int = 1000,
    inlier_threshold: float = 0.05,
    identify_constellations: bool = True,
    max_points: int = 50,
    progress_callback: Optional[callable] = None,
) -> list[dict]:
    """Match circle centers to multiple sky regions and rank by score.

    Automatically applies spatial sampling if too many circles detected.

    Sky region parameters:
    - num_regions: 100 (default in Streamlit UI, adjustable 20-200)
    - Sampling: Uniform random RA (0-360°) and Dec (-90 to +90°)
    - Total matching time: 30-60 seconds for 100 regions

    Spatial sampling trigger:
    - Activated if circle_centers > max_points (default: 50)
    - Preserves pattern geometry using grid-based sampling
    - Reduces RANSAC complexity: O(iterations × circles × stars)

    RANSAC parameters per region:
    - num_iterations=1000: 99% confidence (Fischler & Bolles 1981)
    - inlier_threshold=0.05: Normalized coordinate space (≈0.1° angular separation)
    - sample_size=3: Minimum for 2D similarity transform

    Args:
        circle_centers: Detected circle centers (N, 2)
        sky_regions: List of dicts with 'ra', 'dec', 'stars', 'positions'
        num_iterations: RANSAC iterations per region (default: 1000)
        inlier_threshold: Distance threshold for matches (default: 0.05 in normalized space)
        identify_constellations: Whether to identify constellation names (default: True)
        max_points: Maximum circle centers to use (default: 50)

    Returns:
        Sorted list of match results with sky region info
        Each result includes: ra, dec, score, num_inliers, constellation, visibility, viewing_regions

    Example:
        >>> from celebration_constellation.astronomy import StarCatalog
        >>> catalog = StarCatalog()
        >>> regions = catalog.search_sky_regions(num_samples=50)
        >>> # Add stereographic projections
        >>> for region in regions:
        ...     positions = catalog.convert_to_stereographic(
        ...         region['stars'], region['ra'], region['dec']
        ...     )
        ...     region['positions'] = positions
        >>> matches = match_to_sky_regions(circle_centers, regions)
        >>> m = matches[0]
        >>> print(f"Best match: RA={m['ra']:.1f}, Score={m['score']:.2f}")
        >>> if matches[0].get('constellation'):
        ...     info = matches[0]['constellation_info']
        ...     print(f"Constellation: {info['full_name']}")
    """
    results = []

    # Apply spatial sampling if too many points
    original_count = len(circle_centers)
    if original_count > max_points:
        circle_centers = sample_points_spatially(circle_centers, max_points)
        print(
            f"Sampled {len(circle_centers)} points from {original_count} "
            "for efficient matching"
        )

    # Initialize constellation catalog if needed
    constellation_catalog = None
    # Optional local constellation metadata cache
    const_meta_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "supplemental"
        / "constellations_meta.json"
    )
    const_meta_index = None
    if const_meta_path.exists():
        try:
            with open(const_meta_path, "r", encoding="utf-8") as f:
                meta_list = json.load(f)
            # Build abbrev → info index
            const_meta_index = {
                (item.get("abbrev") or "").upper(): {
                    "full_name": item.get("full_name"),
                    "abbrev": (item.get("abbrev") or "").upper(),
                    "area_sq_deg": item.get("area_sq_deg"),
                }
                for item in meta_list
                if isinstance(item, dict) and item.get("abbrev")
            }
        except Exception:
            const_meta_index = None
    if identify_constellations:
        constellation_catalog = ConstellationCatalog()

    total = len(sky_regions)
    for idx, region in enumerate(sky_regions):
        star_positions = region.get("positions")

        if star_positions is None or len(star_positions) == 0:
            continue

        match_result = ransac_match(
            circle_centers,
            star_positions,
            num_iterations=num_iterations,
            inlier_threshold=inlier_threshold,
        )

        if match_result is not None:
            result = {
                "ra": region["ra"],
                "dec": region["dec"],
                "stars": region["stars"],
                **match_result,
            }

            # Add constellation identification
            if constellation_catalog:
                constellation_name = constellation_catalog.identify_constellation(
                    region["ra"], region["dec"]
                )
                result["constellation"] = constellation_name
                # Prefer local cached metadata when available
                info = None
                if const_meta_index and isinstance(constellation_name, str):
                    info = const_meta_index.get(constellation_name.upper())
                if info is None:
                    info = constellation_catalog.get_constellation_info(
                        constellation_name
                    )
                result["constellation_info"] = info

            # Add viewing location information
            visibility = calculate_visibility_range(region["ra"], region["dec"])
            result["visibility"] = visibility

            # Add geographic regions
            regions_list = get_viewing_regions(
                visibility["min_latitude"], visibility["max_latitude"]
            )
            result["viewing_regions"] = regions_list

            # Add example cities
            cities = get_example_cities(
                visibility["min_latitude"], visibility["max_latitude"]
            )
            result["example_cities"] = cities

            # Add best viewing months
            months = calculate_best_viewing_months(region["ra"])
            result["best_viewing_months"] = months

            results.append(result)

        # Update progress if a callback is provided
        if progress_callback is not None:
            try:
                progress_callback(idx + 1, total)
            except Exception:
                pass

    # Sort by score (descending)
    results.sort(key=lambda x: x["score"], reverse=True)

    return results
