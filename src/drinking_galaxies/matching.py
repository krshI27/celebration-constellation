"""RANSAC-based point cloud matching for constellation identification.

This module implements robust point pattern matching using RANSAC to find
the best-fitting star constellation for detected circle centers, with
correction for star density to avoid bias toward dense regions.
"""

from typing import Optional

import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation


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
    """Compute matching score between two point sets.

    Score accounts for both number of matches and proportion of points matched.

    Args:
        source: Source points (N, 2)
        target: Target points (M, 2)
        threshold: Distance threshold for considering points matched

    Returns:
        Tuple of (score, num_matches)
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
    """RANSAC-based point pattern matching.

    Args:
        source: Source points (circle centers) (N, 2)
        target: Target points (star positions) (M, 2)
        num_iterations: Number of RANSAC iterations
        sample_size: Number of points to sample for hypothesis
        inlier_threshold: Distance threshold for inliers
        min_inliers: Minimum number of inliers required

    Returns:
        Dict with transform parameters and score, or None if no match found

    Example:
        >>> circles = np.array([[100, 150], [200, 180], [150, 250]])
        >>> stars = np.array([[0.5, 0.8], [1.2, 0.9], [0.8, 1.5]])
        >>> result = ransac_match(circles, stars)
        >>> if result:
        ...     print(f"Found {result['num_inliers']} matching points")
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
                best_result = {
                    "scale": scale,
                    "rotation": rotation,
                    "translation": translation,
                    "score": score,
                    "num_inliers": num_matches,
                    "transformed_points": transformed,
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
) -> list[dict]:
    """Match circle centers to multiple sky regions and rank by score.

    Args:
        circle_centers: Detected circle centers (N, 2)
        sky_regions: List of dicts with 'ra', 'dec', 'stars', 'positions'
        num_iterations: RANSAC iterations per region
        inlier_threshold: Distance threshold for matches

    Returns:
        Sorted list of match results with sky region info

    Example:
        >>> from drinking_galaxies.astronomy import StarCatalog
        >>> catalog = StarCatalog()
        >>> regions = catalog.search_sky_regions(num_samples=50)
        >>> # Add stereographic projections
        >>> for region in regions:
        ...     positions = catalog.convert_to_stereographic(
        ...         region['stars'], region['ra'], region['dec']
        ...     )
        ...     region['positions'] = positions
        >>> matches = match_to_sky_regions(circle_centers, regions)
        >>> print(f"Best match: RA={matches[0]['ra']:.1f}, Score={matches[0]['score']:.2f}")
    """
    results = []

    for region in sky_regions:
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
            results.append(result)

    # Sort by score (descending)
    results.sort(key=lambda x: x["score"], reverse=True)

    return results
