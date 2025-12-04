# 🌌 Celebration Constellation: Star Pattern Matcher

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://celebration-constellation.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Find your table's hidden constellation! Upload a photo of bottles, plates, and glasses to discover which star pattern they match in the night sky.

## Overview

**Celebration Constellation** detects circular objects (bottles, plates, glasses) in table photos and matches their spatial arrangement to star constellations visible from Earth using computer vision and RANSAC-based point cloud registration. The app identifies constellation names using IAU official boundaries, provides visibility guidance (where/when to look), and works offline thanks to a bundled star catalog cache.

## 🚀 Live Demo

Try the app now: **[celebration-constellation.streamlit.app](https://celebration-constellation.streamlit.app)**

Deployment:

- **Platform**: Streamlit Community Cloud (free tier)
- **Source**: GitHub [@krshI27/celebration-constellation](https://github.com/krshI27/celebration-constellation)
- **Mirror**: GitLab [@krshi27/celebration-constellation](https://gitlab.com/krshi27/celebration-constellation)

Note: App may sleep on inactivity; first load takes 10-20 seconds to wake.

## Features

✨ **NEW in v0.3.0**: Viewing location calculator and offline mode.

- Shows latitude ranges where constellations are visible
- Provides example cities and geographic regions plus best viewing months
- Optimal viewing locations for maximum altitude
- Works offline using a bundled Bright Star Catalogue cache

✨ **v0.2.0**: Constellation name identification with full IAU constellation data

## Configuration

### Detection Parameters

- **Image Dimensions**: 300×300 to 6000×6000 pixels (recommended: 1500-3000px)
- **Circle Radius Range**: 20-200 pixels (adjustable for close-up or wide shots)
- **Quality Threshold**: 0.15 (lower = more detections, higher = fewer false positives)
- **Maximum Circles**: 50 (automatic spatial sampling if more detected)

### Matching Parameters

- **RANSAC Iterations**: 1000 per sky region (99% confidence)
- **Sky Regions**: 100 (adjustable 20-200 in UI sidebar)
- **Inlier Threshold**: 0.05 (normalized coordinate space)

### Performance Expectations

- Detection: < 5 seconds for typical images
- Matching: 30-60 seconds for 100 sky regions
- Star catalog: 10-second VizieR timeout with instant cache fallback

### Adjusting Parameters

Most parameters have sensible defaults and don't require tuning. However:

- **Increase sky regions** (150-200) if no good matches found
- **Decrease sky regions** (20-50) for faster results when testing
- **Adjust quality threshold** in code if too many/few circles detected

## Example Output

### Circle Detection

The app uses quality filtering and non-maximum suppression to detect high-quality circular objects:

![Detection Example](docs/images/detection_example.jpg)

**46 circular objects detected with quality filtering (edge strength + contrast analysis)**

### Constellation Matching

RANSAC-based point cloud matching finds the best-fitting star constellation:

![Match Example](docs/images/match_example.jpg)

**Best match: Score 23.25, 46 matching stars at RA 193.1°, Dec -12.9°**

### Overlay Comparison

See how detected circles spatially correspond to matched star positions:

![Composite Example](docs/images/composite_example.jpg)

Green circles show detected objects, cyan/yellow stars show the matched constellation pattern.

The app provides:

- Match quality score and number of inliers
- Sky position (RA/Dec coordinates)
- Constellation identification (when boundaries available)
- Viewing location information (latitude ranges, example cities, best months)

## Offline Mode Support

✨ **NEW**: Works without internet connection using local star catalog data

- Bundled Bright Star Catalogue (V/50) with 9,110 stars (cached in `data/supplemental/`)
- Automatic fallback from VizieR to local data
- Graceful degradation when constellation boundaries unavailable
- Full functionality for star matching without network access

## 🚀 Getting Started

### Online Demo

Try the app instantly without installation:
**[celebration-constellation.streamlit.app](https://celebration-constellation.streamlit.app)**

### Local Installation

```bash
# Clone the repository
git clone https://github.com/krshI27/celebration-constellation.git
cd celebration-constellation

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
   # Access at http://localhost:8501
```

## ☁️ Deploy on Streamlit Community Cloud

1) Fork this repo to your GitHub account.
2) Go to [share.streamlit.io](https://share.streamlit.io), create a new app, and point it to your fork:
    - Repository: `your-username/celebration-constellation`
    - Branch: `main`
    - Main file path: `streamlit_app.py`
3) Runtime is pinned via `runtime.txt` (Python 3.11) and `requirements.txt`; no extra config is needed.
4) (Optional) Preload the star catalog cache by running the app once locally and committing the generated `data/supplemental` cache, or let the cloud build fetch VizieR on first run.
5) Press **Deploy**. Cold starts may take ~20s while Streamlit wakes the app and downloads the catalog.

## 📂 Project Structure

```text
celebration-constellation/
├── streamlit_app.py          # Main Streamlit web application
├── setup.py                  # Package configuration
├── requirements.txt          # Production dependencies
├── runtime.txt               # Python version for deployment
├── environment.yml           # Optional conda environment
├── src/
│   └── celebration_constellation/  # Core package
│       ├── __init__.py           # Package metadata
│       ├── astronomy.py          # Star catalog (VizieR + cache)
│       ├── constellations.py     # IAU constellation boundaries + metadata
│       ├── detection.py          # Circle detection (OpenCV + quality filtering)
│       ├── matching.py           # RANSAC matcher with proportion penalty
│       ├── visibility.py         # Geographic visibility and timing
│       ├── visualization.py      # Plotting/overlay utilities
│       └── lines.py              # Constellation line segments
├── scripts/                  # Utility scripts (e.g., ingest_constellation_info.py)
├── data/                     # Data directory (git-ignored)
│   ├── supplemental/         # Cached star catalog and notebooks
│   ├── raw/                  # Original uploads
│   ├── input/                # Preprocessed inputs
│   └── output/               # Generated results
├── docs/                     # Technical documentation
│   ├── ARCHITECTURE.md       # System design overview
│   ├── ALGORITHMS.md         # Algorithm details
│   └── QUICKSTART.md         # Development guide
└── LICENSE                   # MIT license
```

## 🛠 Development

### Prerequisites

- Python 3.11+
- pip or conda/mamba

### Setup Development Environment

```bash
# Install package in editable mode
pip install -e .

# Run tests
pytest

# Format code
black src/ tests/ streamlit_app.py

# Check linting
flake8 src/ tests/ streamlit_app.py
```

### Running Tests

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_matching.py -v
```

## 📖 How It Works

The application uses a four-stage pipeline implemented in `src/celebration_constellation/`:

### 1) Circle Detection (`detection.py`)

- Load and validate images (300×300 to 6000×6000 px), convert to grayscale, apply Gaussian blur.
- OpenCV Hough Circle Transform finds circular edges; quality score mixes Canny edge strength and contrast.
- Non-maximum suppression removes overlapping circles; spatial sampling keeps ≤50 best circles to speed matching.

### 2) Star Catalog (`astronomy.py`)

- Fetches the Yale Bright Star Catalog (V/50) via Astroquery/VizieR with a cached local copy in `data/supplemental/` for offline use.
- Stars are filtered to sky regions and projected to a normalized 2D plane for fast matching.
- Provides constellation boundary data and metadata for downstream steps.

### 3) RANSAC Matching (`matching.py`)

- Circle centers are normalized (zero-mean, unit-scale) and optionally grid-sampled to preserve spatial coverage.
- Samples random sky regions, estimating a 2D similarity transform (scale, rotation, translation) with RANSAC.
- Scores use an inlier threshold plus a proportion penalty to avoid bias toward dense star fields; best-scoring hypotheses keep transformed points for visualization.

### 4) Constellation Identification & Visibility (`constellations.py`, `visibility.py`)

- The best match is intersected with IAU constellation polygons to name the constellation and list member stars/lines.
- Visibility calculators derive latitude ranges, example cities/regions, optimal months (based on RA vs. Sun), and meridian transit timing to suggest when/where to look.

For technical details, see:

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design
- [ALGORITHMS.md](docs/ALGORITHMS.md) - Algorithm deep-dive
- [QUICKSTART.md](docs/QUICKSTART.md) - Development guide

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Format code with Black: `black src/ tests/ streamlit_app.py`
4. Run tests: `pytest`
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for development setup.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yale Bright Star Catalog (V/50)**: Hoffleit & Jaschek via VizieR
- **IAU Constellation Boundaries**: International Astronomical Union
- **OpenCV**: Computer vision library for circle detection
- **Astropy/Astroquery**: Astronomical data access and coordinates

## 📬 Contact

Created by **Maximilian Sperlich**

- GitHub: [@krshI27](https://github.com/krshI27)
- GitLab: [@krshi27](https://gitlab.com/krshi27)

---

*Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) • Built with Python 3.11*
