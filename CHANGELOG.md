# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - UX/UI Improvements (Mobile-First PWA)

- Progressive Web App (PWA) support with manifest and service worker
- Dark astronomy theme with high-contrast colors (cyan accent, deep space background)
- Bottom-centered navigation with larger tap targets (≥48px, WCAG AA compliant)
- Progressive disclosure: verification and visibility details in collapsible expanders
- Time estimates for Sky Regions slider (30-60s for 100-200 regions, 60-90s for >200)
- Advanced Settings expander for max_circles and quality_threshold
- Custom CSS for mobile-responsive design (image height constraints, larger buttons)
- Compact metrics display (single row: Score | Stars | Position)
- PWA meta tags for iOS and Android (theme-color, apple-mobile-web-app-capable)
- Service worker for offline caching and improved performance
- Visualization toggles moved inline above images (Show Circles/Centers)
- Simplified tab labels ("Overlay", "Pattern", "Circles") with Overlay as default

### Changed - UX/UI Improvements

- Streamlined sidebar to 3 essential controls (Min/Max Radius, Sky Regions)
- Reorganized flow: upload → detection → prominent CTA → results
- Primary CTA button now full-width with type="primary" emphasis
- Metrics display condensed to single horizontal row (60% less vertical space)
- Tab order changed to show most relevant view (Overlay) first
- Navigation buttons now equal-width columns with centered match counter
- Expanders include one-line summaries (e.g., "Mean residual: 0.023")
- Dividers lightened (30% opacity) for reduced visual noise
- Success/error messages shortened and more scannable
- Star Data table shows 20 rows by default (was showing all)

### Documentation

- Added explicit parameter documentation to all detection and matching functions
- Created ALGORITHMS.md documenting all formulas and thresholds with justifications
- Created ARCHITECTURE.md explaining file organization and structure decisions
- Created UX_UI_IMPROVEMENTS.md with complete mobile-first design documentation
- Enhanced README.md with Configuration section showing tunable parameters
- Updated docstrings with parameter ranges, default values, and usage examples
- Documented RANSAC confidence calculation (1000 iterations = 99% confidence)
- Clarified match scoring formula with proportion penalty rationale

### Added - Validation

- Image dimension validation in `detection.py:load_image()` (300×300 to 6000×6000 pixels)
- Clear error messages for out-of-bounds image dimensions
- Four new tests for dimension validation edge cases
- `.flake8` configuration for Black-compatible linting (88 char line length)

### Changed - Code Quality

- Updated `calculate_circle_quality()` docstring with quality score formula and threshold guidance
- Updated `detect_circles()` docstring with image constraints and performance expectations
- Updated `ransac_match()` docstring with RANSAC parameter justifications
- Updated `compute_match_score()` docstring explaining proportion penalty
- Updated `match_to_sky_regions()` docstring with spatial sampling trigger and sky region parameters
- README.md Configuration section now includes detection, matching, and performance parameters

## [0.3.0] - 2025-11-XX

### Added

- Viewing location calculator - discover where on Earth you can see matched constellations
- Latitude range calculations for constellation visibility
- Example cities and geographic regions for optimal viewing
- Best viewing months based on RA/Dec coordinates
- Optimal viewing locations for maximum altitude
- Offline mode support with local star catalog
- Bundled Bright Star Catalogue (V/50) with 9,110 stars
- Automatic fallback from VizieR to local data

## [0.2.0] - 2025-11-XX

### Added

- Constellation name identification with full IAU constellation data
- IAU official boundary data for 88 constellations
- Constellation information lookup (full names, abbreviations, mythology)

## [0.1.0] - Initial Release

### Added

- Circle detection using OpenCV Hough Circle Transform
- Quality filtering with edge strength and contrast analysis
- Non-maximum suppression to reduce false positives
- RANSAC-based point pattern matching for constellation identification
- Star catalog management using Yale Bright Star Catalog via VizieR
- Stereographic projection for 2D celestial-to-image matching
- Match scoring with proportion penalty to avoid dense-region bias
- Streamlit web interface for image upload and visualization
- CLI entry point for batch processing
- Comprehensive test suite with synthetic test data
