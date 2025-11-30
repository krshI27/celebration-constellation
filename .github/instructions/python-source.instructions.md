---
applyTo: "src/**/*.py"
---

# Python Source Code Instructions

These instructions apply to all Python source files in `src/project_name/`.

## Code Style Requirements

### Black Formatting
- Line length: 88 characters
- Use double quotes for strings
- Format code before committing: `black src/`

### Import Organization
```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import Optional, Union

# Third-party imports
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine

# Local application imports
from project_name.config import settings
from project_name.utils import validate_input
```

## Type Hints

**ALWAYS add type hints to function signatures:**

```python
# Good: Complete type hints
def process_data(
    input_path: Path,
    output_path: Path,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Process spatial data with coordinate transformation.
    
    Args:
        input_path: Path to input spatial data file
        output_path: Path where processed data will be saved
        crs: Target coordinate reference system
        
    Returns:
        Processed GeoDataFrame with specified CRS
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If data cannot be processed
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    gdf = gpd.read_file(input_path)
    
    if gdf.crs and gdf.crs.to_string() != crs:
        gdf = gdf.to_crs(crs)
    
    gdf.to_file(output_path)
    return gdf

# Bad: No type hints
def process_data(input_path, output_path, crs="EPSG:4326"):
    """Process data."""
    gdf = gpd.read_file(input_path)
    gdf = gdf.to_crs(crs)
    gdf.to_file(output_path)
    return gdf
```

## Function Design

### Early Returns
```python
# Good: Early returns for validation
def calculate_area(gdf: gpd.GeoDataFrame) -> float:
    """Calculate total area of geometries."""
    if gdf is None:
        return 0.0
    
    if gdf.empty:
        return 0.0
    
    if not gdf.geometry.is_valid.all():
        raise ValueError("Invalid geometries found")
    
    return gdf.geometry.area.sum()

# Bad: Nested if/else
def calculate_area(gdf: gpd.GeoDataFrame) -> float:
    """Calculate total area."""
    if gdf is not None:
        if not gdf.empty:
            if gdf.geometry.is_valid.all():
                return gdf.geometry.area.sum()
            else:
                raise ValueError("Invalid geometries")
        else:
            return 0.0
    else:
        return 0.0
```

### Function Length
- Keep functions under 50 lines when possible
- If longer, consider breaking into smaller functions
- Each function should have a single, clear responsibility

```python
# Good: Single responsibility, concise
def validate_geometry(gdf: gpd.GeoDataFrame) -> None:
    """Validate geometry column."""
    if gdf.geometry.isna().any():
        raise ValueError("Null geometries found")
    
    if not gdf.geometry.is_valid.all():
        raise ValueError("Invalid geometries found")

def validate_crs(gdf: gpd.GeoDataFrame, expected_crs: str) -> None:
    """Validate coordinate reference system."""
    if gdf.crs is None:
        raise ValueError("No CRS defined")
    
    if gdf.crs.to_string() != expected_crs:
        raise ValueError(f"Expected {expected_crs}, got {gdf.crs}")

def load_and_validate(file_path: Path) -> gpd.GeoDataFrame:
    """Load and validate spatial data."""
    gdf = gpd.read_file(file_path)
    validate_geometry(gdf)
    validate_crs(gdf, "EPSG:4326")
    return gdf

# Bad: Too many responsibilities in one function
def load_and_validate(file_path: Path) -> gpd.GeoDataFrame:
    """Load and validate data."""
    gdf = gpd.read_file(file_path)
    if gdf.geometry.isna().any():
        raise ValueError("Null geometries")
    if not gdf.geometry.is_valid.all():
        raise ValueError("Invalid geometries")
    if gdf.crs is None:
        raise ValueError("No CRS")
    if gdf.crs.to_string() != "EPSG:4326":
        raise ValueError("Wrong CRS")
    # ... more validation logic
    return gdf
```

## Error Handling

```python
# Good: Specific exceptions with clear messages
def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"Invalid YAML in configuration file: {e}"
        ) from e
    
    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must be a dictionary"
        )
    
    return config

# Bad: Generic exceptions, no context
def load_config(config_path: Path) -> dict:
    """Load config."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except:
        raise Exception("Error loading config")
```

## Docstrings

```python
# Good: Clear docstring with examples
def buffer_geometries(
    gdf: gpd.GeoDataFrame,
    distance: float,
    resolution: int = 16,
) -> gpd.GeoDataFrame:
    """Create buffer around geometries.
    
    Args:
        gdf: Input GeoDataFrame with geometries
        distance: Buffer distance in units of the CRS
        resolution: Number of segments per quadrant (default 16)
        
    Returns:
        GeoDataFrame with buffered geometries
        
    Example:
        >>> gdf = gpd.read_file("points.geojson")
        >>> buffered = buffer_geometries(gdf, distance=100)
        >>> print(buffered.geometry.area.sum())
        31415.926
    """
    buffered = gdf.copy()
    buffered.geometry = gdf.geometry.buffer(
        distance,
        resolution=resolution,
    )
    return buffered

# Bad: Minimal or missing docstring
def buffer_geometries(gdf, distance, resolution=16):
    """Buffer geometries."""
    return gdf.geometry.buffer(distance, resolution=resolution)
```

## Naming Conventions

```python
# Constants at module level
MAX_RETRIES = 3
DEFAULT_CRS = "EPSG:4326"
DATABASE_URL = os.getenv("DATABASE_URL")

# Functions and methods
def calculate_total_area(gdf: gpd.GeoDataFrame) -> float:
    """Calculate total area."""
    pass

# Classes
class SpatialDataProcessor:
    """Process spatial data."""
    
    def __init__(self, config: dict) -> None:
        self.config = config
    
    def process_file(self, file_path: Path) -> gpd.GeoDataFrame:
        """Process single file."""
        pass

# Private methods
def _validate_internal(data: Any) -> bool:
    """Internal validation (not part of public API)."""
    pass
```

## What NOT to Do

❌ Don't use nested if/else (use early returns)  
❌ Don't create functions longer than 50 lines without good reason  
❌ Don't omit type hints on function signatures  
❌ Don't hardcode configuration values  
❌ Don't use bare `except:` clauses  
❌ Don't mix tabs and spaces  
❌ Don't import with `*`  
❌ Don't use mutable default arguments  
❌ Don't ignore or suppress linting errors without explanation

## Review Checklist

Before committing Python source code:

- [ ] Black formatting applied: `black src/`
- [ ] Flake8 passes: `flake8 src/`
- [ ] Type hints added to all functions
- [ ] Docstrings written for public functions
- [ ] Error handling implemented with specific exceptions
- [ ] Tests written and passing: `pytest`
- [ ] No hardcoded secrets or configuration
- [ ] Functions under 50 lines (or justified)
- [ ] Early returns used instead of nesting
