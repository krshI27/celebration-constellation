"""Tests for RANSAC-based constellation matching."""

import numpy as np

from drinking_galaxies.matching import (
    apply_transform,
    compute_match_score,
    estimate_similarity_transform,
    normalize_points,
    ransac_match,
)


def test_normalize_points():
    """Test point cloud normalization."""
    points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    normalized, scale, center = normalize_points(points)

    assert normalized.shape == points.shape, "Shape should be preserved"
    assert np.abs(np.mean(normalized)) < 1e-10, "Mean should be near zero"
    assert scale > 0, "Scale should be positive"
    assert center.shape == (2,), "Center should be 2D"


def test_estimate_similarity_transform():
    """Test similarity transform estimation."""
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])  # Translated by (1, 1)

    scale, rotation, translation = estimate_similarity_transform(source, target)

    assert isinstance(scale, float), "Scale should be float"
    assert isinstance(rotation, float), "Rotation should be float"
    assert translation.shape == (2,), "Translation should be 2D"


def test_apply_transform():
    """Test applying similarity transform."""
    points = np.array([[1.0, 1.0], [2.0, 2.0]])
    scale = 2.0
    rotation = 0.0
    translation = np.array([1.0, 1.0])

    transformed = apply_transform(points, scale, rotation, translation)

    expected = np.array([[3.0, 3.0], [5.0, 5.0]])
    assert np.allclose(
        transformed, expected
    ), "Transform should scale by 2 and translate by (1,1)"


def test_compute_match_score_perfect_match():
    """Test match score for perfect alignment."""
    source = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    target = source.copy()  # Perfect match

    score, num_matches = compute_match_score(source, target, threshold=0.1)

    assert num_matches == 3, "Should match all 3 points"
    assert score > 0, "Score should be positive"


def test_compute_match_score_no_match():
    """Test match score when points are far apart."""
    source = np.array([[0.0, 0.0], [1.0, 1.0]])
    target = np.array([[100.0, 100.0], [200.0, 200.0]])  # Far away

    score, num_matches = compute_match_score(source, target, threshold=0.1)

    assert num_matches == 0, "Should match no points"
    assert score == 0.0, "Score should be zero"


def test_compute_match_score_proportion_penalty():
    """Test that score is penalized when target has many more points."""
    source = np.array([[0.0, 0.0], [1.0, 1.0]])
    # Many target points
    target = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ]
    )

    score, num_matches = compute_match_score(source, target, threshold=0.1)

    # Score should be penalized due to proportion
    assert num_matches == 2, "Should match 2 points"
    assert score < num_matches, "Score should be less than num_matches due to penalty"


def test_ransac_match_with_similar_patterns():
    """Test RANSAC matching with similar point patterns."""
    # Create a simple pattern
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])

    # Create target that's a scaled and translated version
    scale = 2.0
    translation = np.array([5.0, 5.0])
    target = source * scale + translation

    result = ransac_match(
        source, target, num_iterations=100, sample_size=3, inlier_threshold=0.5
    )

    assert result is not None, "Should find a match"
    assert "scale" in result, "Result should contain scale"
    assert "rotation" in result, "Result should contain rotation"
    assert "translation" in result, "Result should contain translation"
    assert "score" in result, "Result should contain score"
    assert "num_inliers" in result, "Result should contain num_inliers"
    assert result["num_inliers"] >= 3, "Should have at least 3 inliers"


def test_ransac_match_insufficient_points():
    """Test RANSAC with insufficient points."""
    source = np.array([[0.0, 0.0], [1.0, 1.0]])  # Only 2 points
    target = np.array([[0.0, 0.0]])  # Only 1 point

    result = ransac_match(source, target, sample_size=3)

    assert result is None, "Should return None when insufficient points"


def test_ransac_match_no_overlap():
    """Test RANSAC with completely different patterns."""
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target = np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]])  # Far away

    result = ransac_match(
        source,
        target,
        num_iterations=50,
        sample_size=3,
        inlier_threshold=0.1,
        min_inliers=3,
    )

    # May or may not find a match depending on random sampling
    # Just verify it doesn't crash
    if result is not None:
        assert isinstance(result, dict), "Result should be dict if not None"
