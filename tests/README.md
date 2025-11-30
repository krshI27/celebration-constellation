# Tests

This directory contains all test files for the Drinking Galaxies project.

## Structure

```
tests/
├── fixtures/              # Test data files (small, version controlled)
├── test_detection.py      # Circle detection algorithm tests
├── test_astronomy.py      # Star catalog and data management tests
├── test_matching.py       # RANSAC constellation matching tests
└── test_integration.py    # End-to-end integration tests
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_detection.py

# Run with verbose output
pytest -v
```

## Test Fixtures

Small test images and data files are stored in `fixtures/` for repeatable testing.
