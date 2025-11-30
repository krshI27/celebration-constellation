---
applyTo: "**/tests/**/*.py"
---

# Testing Instructions

These instructions apply to all test files in the project.

## Testing Framework

**Use pytest exclusively:**
- No unittest, nose, or other frameworks
- Use pytest fixtures for test data
- Use pytest parametrize for multiple test cases

## Test File Naming

```
tests/
├── test_module.py         # Unit tests for src/project_name/module.py
├── test_integration.py    # Integration tests
├── fixtures/              # Test data (small files only)
│   └── sample_data.geojson
└── conftest.py            # Shared fixtures
```

## Test Function Structure

**ALWAYS use arrange-act-assert pattern:**

```python
def test_function_does_something_specific() -> None:
    """Test description in present tense.
    
    More detailed explanation if needed.
    """
    # Arrange: Set up test data and preconditions
    input_data = create_test_data()
    expected_result = "expected value"
    
    # Act: Execute the function being tested
    actual_result = function_under_test(input_data)
    
    # Assert: Verify the results
    assert actual_result == expected_result
```

## Required Patterns

### Type Hints
```python
# Good: Type hints on test functions
def test_load_data(tmp_path: Path) -> None:
    """Test data loading."""
    result = load_data(tmp_path / "test.csv")
    assert isinstance(result, pd.DataFrame)

# Bad: No type hints
def test_load_data(tmp_path):
    result = load_data(tmp_path / "test.csv")
    assert result is not None
```

### Descriptive Names
```python
# Good: Clear, specific test name
def test_load_spatial_data_raises_error_when_file_not_found() -> None:
    """Test FileNotFoundError is raised for missing files."""
    with pytest.raises(FileNotFoundError):
        load_spatial_data(Path("nonexistent.geojson"))

# Bad: Vague test name
def test_error() -> None:
    """Test error."""
    # What error? When?
    pass
```

### Fixtures for Test Data
```python
# Good: Use fixtures for reusable test data
@pytest.fixture
def sample_geodataframe() -> gpd.GeoDataFrame:
    """Create sample GeoDataFrame for testing."""
    return gpd.GeoDataFrame(
        {"id": [1, 2], "name": ["A", "B"]},
        geometry=gpd.points_from_xy([0, 1], [0, 1]),
        crs="EPSG:4326",
    )

def test_process_data(sample_geodataframe: gpd.GeoDataFrame) -> None:
    """Test data processing."""
    result = process_data(sample_geodataframe)
    assert len(result) == 2

# Bad: Recreate data in every test
def test_process_data() -> None:
    gdf = gpd.GeoDataFrame(...)  # Repeated in every test
    result = process_data(gdf)
    assert len(result) == 2
```

### Parametrize for Multiple Cases
```python
# Good: Use parametrize for similar test cases
@pytest.mark.parametrize(
    "input_value,expected_output",
    [
        (0, "zero"),
        (1, "one"),
        (-1, "negative"),
    ],
)
def test_number_to_string(input_value: int, expected_output: str) -> None:
    """Test number to string conversion."""
    assert number_to_string(input_value) == expected_output

# Bad: Separate test for each case
def test_zero():
    assert number_to_string(0) == "zero"

def test_one():
    assert number_to_string(1) == "one"

def test_negative():
    assert number_to_string(-1) == "negative"
```

## Testing Spatial Operations

```python
"""Test spatial data operations."""

import pytest
import geopandas as gpd
from shapely.geometry import Point, Polygon


def test_spatial_join_with_matching_crs() -> None:
    """Test spatial join ensures CRS alignment."""
    # Arrange
    points = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    polygons = gpd.GeoDataFrame(
        {"name": ["Area"]},
        geometry=[Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])],
        crs="EPSG:3857",  # Different CRS
    )
    
    # Act
    result = spatial_join(points, polygons)
    
    # Assert
    assert result.crs == points.crs  # Should match left GeoDataFrame
    assert len(result) > 0  # Join should find matches
```

## Test Coverage Requirements

- Every public function must have at least one test
- Test both success and failure cases
- Test edge cases: empty inputs, None values, invalid types
- Test error conditions: verify correct exceptions are raised
- Integration tests for database operations
- Performance tests for operations on large datasets

## Commands for Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/project_name --cov-report=term-missing

# Run specific test file
pytest tests/test_module.py

# Run specific test function
pytest tests/test_module.py::test_function_name

# Run verbose mode
pytest -v

# Run with output
pytest -s
```

## What NOT to Do

❌ Don't modify source code from test files  
❌ Don't use production data in tests  
❌ Don't commit large test files (use small fixtures)  
❌ Don't skip tests without explanation  
❌ Don't test implementation details (test behavior)  
❌ Don't create tests without docstrings  
❌ Don't use generic assertions like `assert x` or `assert x is not None`
