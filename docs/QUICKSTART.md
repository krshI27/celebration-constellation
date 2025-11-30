# Quick Start Guide: Drinking Galaxies

## Prerequisites

- **Conda/Mamba**: For environment management
- **Docker**: For containerized deployment (optional)
- **Git**: For version control

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url> drinking-galaxies
cd drinking-galaxies
```

### 2. Create Conda Environment

```bash
# Using conda
conda env create -f .config/environment.yml
conda activate drinking-galaxies

# Or using mamba (faster)
mamba env create -f .config/environment.yml
conda activate drinking-galaxies
```

### 3. Install Package in Development Mode

```bash
pip install -e .
```

## Running the Application

### Option 1: Streamlit Web App (Recommended)

```bash
# Activate environment
conda activate drinking-galaxies

# Run Streamlit app
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Option 2: Docker

```bash
# Build and start containers
cd .docker
docker compose up -d

# Access Streamlit app
open http://localhost:8501

# View logs
docker compose logs -f

# Stop containers
docker compose down
```

## Usage

1. **Upload Photo**: Click "Browse files" and select a table photo
2. **Adjust Detection**: Use sidebar sliders for circle detection parameters
3. **Find Constellations**: Click "Find Matching Constellations"
4. **Browse Matches**: Use Previous/Next buttons to explore different constellation matches
5. **Toggle Overlays**: Use checkboxes to show/hide circles and center points

## How It Works

1. **Circle Detection**: OpenCV Hough Circle Transform detects bottles, plates, glasses
2. **Star Catalog**: Yale Bright Star Catalog (~9,000 stars, < 1MB)
3. **Matching**: RANSAC point cloud registration with proportion correction
4. **Ranking**: Matches sorted by quality score

## Configuration

Edit `.env` file (copy from `.config/.env.example`):

```bash
cp .config/.env.example .env
```

Key settings:

- `MIN_CIRCLE_RADIUS`: Minimum circle size (default: 20)
- `MAX_CIRCLE_RADIUS`: Maximum circle size (default: 200)
- `RANSAC_ITERATIONS`: Matching iterations (default: 1000)
- `SKY_SEARCH_REGIONS`: Number of sky regions to search (default: 100)

## Troubleshooting

### No circles detected

- Adjust `Min Circle Radius` and `Max Circle Radius` sliders
- Ensure objects are reasonably circular in the photo
- Try different lighting conditions

### No constellation matches

- Increase "Sky Regions to Search"
- Check that at least 3-4 circular objects are detected
- Verify circle centers form a distinctive pattern

### Slow performance

- Reduce "Sky Regions to Search"
- Lower RANSAC iterations in `.env`
- Use smaller images (resize before upload)

## Development

### Run Tests

```bash
pytest
pytest --cov=src --cov-report=term-missing
```

### Format Code

```bash
black src/ tests/
flake8 src/ tests/
```

### Add New Features

Follow the spec-driven workflow:

```bash
# Install spec-kit CLI
pip install uv
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git

# Initialize for your AI assistant
specify init  # Select GitHub Copilot or Claude Code

# Create specification
specify specification

# Generate plan
specify plan

# Break into tasks
specify tasks

# Implement
specify implement
```

## Project Structure

```
drinking-galaxies/
├── src/drinking_galaxies/   # Core Python package
│   ├── detection.py         # Circle detection
│   ├── astronomy.py         # Star catalog
│   └── matching.py          # RANSAC matching
├── streamlit_app.py         # Web interface
├── tests/                   # Test files
├── docs/                    # Documentation
└── .specify/                # Specifications and plans
```

## Next Steps

- Add camera capture for live photo input
- Implement constellation identification (names)
- Add social sharing features
- Create augmented reality overlay
- Time-based matching (visible tonight from user location)

## Support

For issues, questions, or contributions, see the main README.md
