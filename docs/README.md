# Documentation

Documentation for the Drinking Galaxies project.

## Contents

- **architecture.md**: System architecture and design decisions
- **algorithms.md**: Detailed explanation of circle detection and star matching algorithms
- **deployment.md**: Deployment guide for web and mobile platforms
- **api.md**: API documentation for core modules
- **user-guide.md**: End-user instructions for the web application

## Project Overview

Drinking Galaxies detects circular objects (bottles, plates, glasses) in table photos and matches their spatial arrangement to star constellations visible from Earth.

### Key Technologies

- **Circle Detection**: OpenCV Hough Circle Transform
- **Star Catalog**: Yale Bright Star Catalog or Hipparcos (< 20GB)
- **Matching Algorithm**: RANSAC-based point cloud registration
- **Web Interface**: Streamlit with swipeable image overlays
- **Deployment**: Docker containers, no Android SDK required
