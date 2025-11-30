# Architecture and File Organization

This document explains the architectural decisions and file organization structure for Drinking Galaxies, with explicit justifications referencing the project constitution and industry best practices.

## Project Structure

```
drinking-galaxies/
├── src/                      # Source code (package layout)
│   └── drinking_galaxies/    # Main package
│       ├── __init__.py
│       ├── main.py           # CLI entry point
│       ├── astronomy.py      # Star catalog management
│       ├── constellations.py # Constellation identification
│       ├── detection.py      # Circle detection (OpenCV)
│       ├── matching.py       # RANSAC-based matching
│       ├── visibility.py     # Viewing location calculations
│       └── visualization.py  # Image overlays and rendering
├── streamlit_app.py          # Web interface (root level)
├── tests/                    # Test files
│   ├── test_astronomy.py
│   ├── test_constellations.py
│   ├── test_detection.py
│   ├── test_matching.py
│   ├── test_visibility.py
│   ├── test_visualization.py
│   └── fixtures/             # Test data (small, version-controlled)
│       └── README.md
├── data/                     # Data files (git-ignored)
│   ├── raw/                  # Original uploaded images
│   ├── input/                # Preprocessed images
│   ├── output/               # Detection results, visualizations
│   └── supplemental/         # Star catalog cache, notebooks
├── docs/                     # Documentation
│   ├── QUICKSTART.md
│   ├── ALGORITHMS.md         # This document's companion
│   └── images/               # Screenshots and examples
├── .docker/                  # Docker development environment
├── .config/                  # Configuration files
├── .github/                  # GitHub Copilot integration
├── .claude/                  # Claude Code instructions
└── .specify/                 # Spec-kit files
```

---

## Core Architectural Decisions

### 1. Flat Module Structure

**Decision**: Single package (`drinking_galaxies/`) with 7 modules, no subpackages.

**Rationale**:
- **Small codebase**: ~2000 lines total (detection: 300, matching: 250, astronomy: 400, etc.)
- **Single purpose**: Application does one thing well (constellation matching)
- **Clear separation**: Each module has distinct responsibility (detection, matching, astronomy, etc.)
- **No interdependencies**: Modules interact through simple function calls, no complex class hierarchies

**Constitution compliance** (v1.0.0):
> "For small projects (< 500 lines, single purpose), use flat structure."

**When to consider subpackages**:
- Total lines > 2000 and growing
- Multiple teams working on different features
- Need for plugin architecture or extensibility

**References**:
- Python Packaging User Guide: <https://packaging.python.org/en/latest/>
- "Structuring Your Project" - The Hitchhiker's Guide to Python

---

### 2. Streamlit App in Root

**Decision**: `streamlit_app.py` placed in project root, not in `src/`.

**Rationale**:

**Streamlit convention**:
```bash
streamlit run streamlit_app.py
```

Framework expects app file in current directory or specified path. Root placement enables:
- Simple command: `streamlit run streamlit_app.py` (no path navigation)
- Clear separation: Application layer (UI) vs. library layer (business logic)
- Importability: `from drinking_galaxies import detection` works cleanly

**Alternative considered**: `src/drinking_galaxies/streamlit_app.py`

Rejected because:
- Violates Streamlit convention (requires `streamlit run src/drinking_galaxies/streamlit_app.py`)
- Mixes UI code with library code
- Complicates future packaging (library vs. application)

**References**:
- Streamlit documentation: <https://docs.streamlit.io/library/get-started/main-concepts>
- Python package structure best practices (separate app from library)

---

### 3. Main Entry Point: main.py

**Decision**: CLI entry point is `main.py`, not `__main__.py` or `drinking_galaxies.py`.

**Rationale**:

**Constitution requirement** (v1.0.0):
> "Always name the main entry point file main.py (never use the package name)"

**Benefits**:
- **Clarity**: Immediately obvious which file starts the application
- **Consistency**: Same pattern across all projects using this template
- **No confusion**: Avoids circular import issues with module named same as package

**Usage patterns**:
```bash
# CLI (minimal, delegates to library)
python src/drinking_galaxies/main.py photo.jpg

# Streamlit (primary interface)
streamlit run streamlit_app.py

# Library import (for scripts or notebooks)
from drinking_galaxies import detect_and_extract
```

**References**:
- Project constitution v1.0.0, "Python Package Structure Selection"
- PEP 338 (Executing modules as scripts)

---

### 4. Test Structure: Synthetic Data Only

**Decision**: Tests use programmatically generated data (numpy arrays), not fixture files.

**Rationale**:

**Constitution requirement** (v1.0.0):
> "Tests MUST use synthetic test data, not fixture files."

**Benefits**:
- **Reproducible**: No dependency on external image files
- **Version control**: No large binary files in git
- **Fast execution**: No I/O overhead, tests run in-memory
- **Flexible**: Easy to create edge cases (empty images, specific patterns)
- **Portable**: Tests work on any system without data download

**Implementation pattern**:
```python
# Create synthetic test image with known circle
def create_test_image_with_circle(size=300, radius=50):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    y, x = np.ogrid[:size, :size]
    mask = (x - size//2)**2 + (y - size//2)**2 <= radius**2
    image[mask] = [255, 255, 255]
    return image

# Use in test
def test_detect_circles():
    image = create_test_image_with_circle()
    circles = detect_circles(image)
    assert circles is not None
    assert len(circles) == 1
```

**Exception**: `tests/fixtures/` may contain small reference data (< 100 KB) for integration tests, but synthetic data is preferred.

**References**:
- Project constitution v1.0.0, "Test Structure"
- pytest best practices: <https://docs.pytest.org/en/stable/goodpractices.html>

---

### 5. Data Directory Organization

**Decision**: Three-tier structure: `raw/`, `input/`, `output/`.

**Structure**:
```
data/
├── raw/         # Original, immutable source data
├── input/       # Preprocessed, ready for processing
├── output/      # Results, visualizations, reports
└── supplemental/# Star catalog cache, notebooks
```

**Rationale**:

**raw/ (immutable)**:
- Original uploaded images
- Never modified programmatically
- Serves as backup and audit trail

**input/ (processed)**:
- Resized or normalized images
- Cached preprocessing results
- Ready for detection pipeline

**output/ (generated)**:
- Detection results (circles, centers)
- Constellation match visualizations
- Exported data (CSV, JSON)

**Git exclusion**: Entire `data/` directory git-ignored to prevent:
- Large file bloat (images can be 1-5 MB each)
- Sensitive data leakage (user-uploaded photos)
- Merge conflicts on binary files

**Exception**: `data/supplemental/vizier_v50.ipynb` version-controlled for reproducibility (star catalog download script).

**References**:
- Project constitution v1.0.0, "Data Organization"
- Data science best practices (raw/processed/results separation)

---

### 6. Configuration and Environment

**Decision**: All config files in `.config/`, secrets in `.env` (git-ignored).

**Structure**:
```
.config/
├── environment.yml      # Conda dependencies
├── .env.example        # Environment variable template
└── ai-instructions.md  # AI assistant guidelines
```

**Environment variables** (.env):
```bash
PROJECT_NAME=drinking-galaxies
VIZIER_TIMEOUT=10          # Star catalog query timeout
CACHE_DIR=~/.drinking_galaxies/star_cache/
LOG_LEVEL=INFO
```

**Security**:
- `.env` git-ignored (never commit secrets)
- `.env.example` version-controlled (template for developers)
- Validate required variables at startup

**References**:
- 12-Factor App methodology: <https://12factor.net/config>
- Project constitution v1.0.0, "Configuration and Security"

---

### 7. Docker Development Environment

**Decision**: All Docker files in `.docker/`, not in project root.

**Rationale**:

**Constitution requirement** (v1.0.0):
> "Keep project root CLEAN: Only essential files (README.md, .gitignore, optionally pyproject.toml)"

**Structure**:
```
.docker/
├── Dockerfile              # Multi-stage build
├── docker-compose.yml      # Service orchestration
└── scripts/                # Container setup scripts
```

**Benefits**:
- Clean root directory
- Grouped Docker-related files
- Easy to locate and modify

**Usage**:
```bash
cd .docker
docker compose up -d
docker compose exec dev bash
```

**Note**: Use `docker compose` (with space), not `docker-compose` (deprecated hyphenated command).

**References**:
- Docker Compose v2 specification: <https://docs.docker.com/compose/>
- Project constitution v1.0.0, "Project Structure"

---

### 8. Documentation Structure

**Decision**: Technical docs in `docs/`, specifications in `.specify/`.

**docs/ (project documentation)**:
- QUICKSTART.md - Getting started guide
- ALGORITHMS.md - Algorithm explanations
- ARCHITECTURE.md - This document
- images/ - Screenshots and examples

**.specify/ (spec-kit)**:
- constitution.md - Project principles
- specifications/ - Requirements and user stories
- plans/ - Technical implementation plans
- tasks/ - Implementation task breakdowns

**Separation rationale**:
- `docs/` - For users and contributors (public-facing)
- `.specify/` - For development workflow (internal process)

**References**:
- GitHub spec-kit: <https://github.com/github/spec-kit>
- Documentation best practices (user docs vs. internal specs)

---

### 9. AI Assistant Integration

**Decision**: Dual AI configuration (GitHub Copilot + Claude Code).

**Structure**:
```
.github/
├── copilot-instructions.md     # Repository-wide instructions
├── agents/                     # Custom agent profiles
│   ├── docs-agent.md
│   ├── test-agent.md
│   ├── data-agent.md
│   └── lint-agent.md
└── instructions/               # Path-specific instructions

.claude/
└── CLAUDE.md                   # Claude Code comprehensive instructions
```

**Rationale**:
- **GitHub Copilot**: Inline completions, chat, repository context
- **Claude Code**: Long-form reasoning, file creation, multi-step workflows
- Both follow project constitution and coding standards

**Custom agents**: Specialized AI profiles for focused tasks:
- `@docs-agent` - Technical writing
- `@test-agent` - Quality assurance
- `@data-agent` - Spatial analysis
- `@lint-agent` - Code formatting

**References**:
- Project constitution v1.0.0, "AI Assistant Integration"
- .github/AGENTS.md (agent documentation)

---

## Package Structure Selection Criteria

### When to Use Flat Structure (Current)

**Criteria** (from constitution):
- Total lines < 2000
- Single purpose application
- 1-2 developers
- No plugin system needed

**Drinking Galaxies fits all criteria**:
- ~2000 lines total
- Single purpose (constellation matching)
- Individual developer project
- No extensibility requirements

### When to Consider Modular Structure

**Migrate if**:
- Total lines > 2000 and growing
- Multiple distinct features (e.g., add planets, deep-sky objects)
- Team of 3+ developers
- Need for plugin architecture

**Example modular structure** (future):
```
src/drinking_galaxies/
├── __init__.py
├── main.py
├── core/           # Detection and matching
│   ├── detection.py
│   ├── matching.py
│   └── preprocessing.py
├── astronomy/      # Star catalogs and projections
│   ├── catalogs.py
│   ├── constellations.py
│   └── visibility.py
└── ui/             # User interfaces
    ├── streamlit_app.py
    └── cli.py
```

**References**:
- Project constitution v1.0.0, "Python Package Structure Selection"
- "Scaling Python Codebases" - Real Python

---

## Constitution Compliance Checklist

✅ **Workflow**: Explore → Plan → Code → Commit (followed for this clarification feature)  
✅ **Code Quality**: Black formatting (88 chars), type hints, early returns  
✅ **Environment**: Conda/mamba only (no pip venv/virtualenv/poetry)  
✅ **Folder Structure**: All files in correct locations (src/, docs/, .docker/, .config/)  
✅ **Root Cleanliness**: Only README.md, .gitignore, streamlit_app.py, setup.py in root  
✅ **Security**: No secrets in code, .env git-ignored, validation at startup  
✅ **Testing**: Synthetic test data (no large fixture files)  
✅ **Documentation**: Technical docs in docs/, specifications in .specify/

**Constitution version**: 1.0.0  
**Last verified**: 2025-11-30

---

## Assumptions and Design Constraints

### Assumption 1: Single-User Desktop Application

**Context**: No multi-user scenarios, no server deployment.

**Architectural implications**:
- No database (local file cache only)
- No authentication or user management
- No concurrent access handling
- Streamlit single-process model sufficient

**File impact**:
- Star catalog cache: `~/.drinking_galaxies/star_cache/` (user home directory)
- No need for multi-tenant data isolation
- Simple file-based configuration

### Assumption 2: Offline Operation Preferred

**Context**: v0.3.0 added offline mode with bundled star catalog.

**Architectural implications**:
- VizieR query timeout acceptable (10s with cache fallback)
- Star catalog bundled in `data/supplemental/`
- No real-time catalog updates required

**File impact**:
- `astronomy.py`: Local cache takes precedence over VizieR
- `data/supplemental/vizier_v50.ipynb`: Catalog download and caching logic
- No network dependency after first download

### Assumption 3: Modern Desktop Hardware

**Context**: Target is 2018+ desktop/laptop with 8+ GB RAM.

**Architectural implications**:
- Image processing uses full CPU (no battery optimization)
- 6000×6000 pixel images processable in-memory
- 60-second processing time acceptable

**File impact**:
- No memory-mapped files needed
- No image streaming or chunking
- Simple numpy arrays for all processing

---

## Future Architecture Considerations

### Potential Enhancements

**If adding mobile support**:
- Restructure to separate web API (`api/`) from core library
- Add image streaming for large uploads
- Optimize for lower memory footprint

**If adding database**:
- Move star catalog to PostgreSQL/PostGIS
- Add user management and match history
- Require migration to modular structure (`core/`, `data/`, `api/`)

**If adding real-time features**:
- WebSocket integration for progress updates
- Background job queue (Celery, RQ)
- Separate Streamlit limitations (single-process)

**All changes must**:
- Follow project constitution v1.0.0
- Maintain clean root directory
- Use synthetic test data
- Document architectural decisions

---

**Version**: 1.0.0  
**Date**: 2025-11-30  
**Based on**: Drinking Galaxies v0.3.0 architecture analysis
