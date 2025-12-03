# 🌌 Celebration Constellation: Star Pattern Matcher

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://celebration-constellation.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Find your table's hidden constellation! Upload a photo of bottles, plates, and glasses to discover which star pattern they match in the night sky.

## Overview

**Celebration Constellation** detects circular objects (bottles, plates, glasses) in table photos and matches their spatial arrangement to star constellations visible from Earth using computer vision and RANSAC-based point cloud registration. The app identifies constellation names using IAU official boundaries and provides educational information about each match.

## 🚀 Live Demo

Try the app now: **[celebration-constellation.streamlit.app](https://celebration-constellation.streamlit.app)**

Deployment:

- **Platform**: Streamlit Community Cloud (free tier)
- **Source**: GitHub [@krshI27/celebration-constellation](https://github.com/krshI27/celebration-constellation)
- **Mirror**: GitLab [@krshi27/celebration-constellation](https://gitlab.com/krshi27/celebration-constellation)

Note: App may sleep on inactivity; first load takes 10-20 seconds to wake.

## Features

✨ **NEW in v0.3.0**: Viewing location calculator - discover where on Earth you can see your matched constellation!

- Shows latitude ranges where constellations are visible
- Provides example cities and geographic regions
- Displays best viewing months based on RA/Dec coordinates
- Optimal viewing locations for maximum altitude

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

- Bundled Bright Star Catalogue (V/50) with 9,110 stars
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

## 📂 Project Structure

```text
celebration-constellation/
├── streamlit_app.py          # Main Streamlit web application
├── setup.py                  # Package configuration
├── requirements.txt          # Production dependencies
├── runtime.txt              # Python version for deployment
├── src/
│   └── celebration_constellation/  # Core package
│       ├── __init__.py           # Package metadata
│       ├── detection.py          # Circle detection (OpenCV)
│       ├── astronomy.py          # Star catalog (VizieR + cached)
│       ├── constellations.py     # IAU boundaries + info
│       ├── matching.py           # RANSAC matching algorithm
│       ├── visibility.py         # Geographic visibility calc
│       └── visualization.py      # Plotting utilities
├── tests/                    # Pytest test suite
├── docs/                     # Technical documentation
│   ├── ARCHITECTURE.md       # System design overview
│   ├── ALGORITHMS.md         # Algorithm details
│   └── QUICKSTART.md         # Development guide
├── data/                     # Data directory (git-ignored)
│   ├── supplemental/         # Cached star catalog
│   └── input/output/raw/     # User data directories
└── .streamlit/
    └── config.toml           # PWA-optimized theme
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

The application uses a four-stage pipeline:

### 1. Circle Detection
Uses OpenCV's Hough Circle Transform to detect circular objects in the uploaded image:
- Converts to grayscale
- Applies Gaussian blur
- Detects circles (bottles, plates, glasses)
- Extracts center coordinates

### 2. Star Catalog
Queries Yale Bright Star Catalog (V/50) from VizieR:
- 9,110 bright stars (magnitude < 6.0)
- Cached locally for offline access
- Filters by celestial region
- Projects to 2D plane using stereographic projection

### 3. RANSAC Matching
Matches circle positions to star patterns:
- Samples random sky regions
- Estimates 2D similarity transform (scale, rotation, translation)
- Scores matches using proportion penalty
- Ranks constellations by match quality

### 4. Constellation Identification
Uses IAU official constellation boundaries:
- Determines which constellation contains each match
- Provides constellation information (mythology, best viewing)
- Calculates geographic visibility
- Displays optimal viewing locations

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
