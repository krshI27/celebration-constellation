# Data Directory

This directory contains project data files with a simplified, three-folder organization.

**Important**: The entire `data/` directory is excluded from version control to prevent accidental commits of large or sensitive data files.

## Directory Structure

```
data/
├── raw/          # Original, unprocessed data files
├── input/        # Data prepared for processing/analysis
└── output/       # Results, reports, and generated datasets
```

## Guidelines

### Raw Data (`raw/`)

- Original data files as received from sources
- Never modify files in this directory
- Document data sources and acquisition dates
- Keep original file names when possible

### Input Data (`input/`)

- Data that is ready for processing or analysis
- May include cleaned, filtered, or preprocessed data
- Can be derived from raw data or external sources
- Should be in analysis-ready format

### Output Data (`output/`)

- Final results, reports, and generated datasets
- Model outputs, predictions, and analysis results
- Processed data ready for sharing or publication
- Should be clearly documented and reproducible

## Data Management

### File Naming

- Use descriptive, consistent naming conventions
- Include dates in format YYYY-MM-DD
- Avoid spaces and special characters
- Use lowercase with underscores

Examples:

- Raw: `forest_inventory_2025-09-23_source.csv`
- Input: `forest_data_processed_2025-09-23.parquet`
- Output: `analysis_results_2025-09-23_v1.xlsx`

### Documentation

- Include README files in each subdirectory
- Document data sources and processing steps
- Maintain data dictionaries for complex datasets
- Track data lineage and transformations

### Security and Version Control

- **Entire data/ directory is excluded from git**
- Only README files in subdirectories are version controlled
- Use external data storage for large datasets
- Consider data encryption for sensitive datasets
- Document data sources and how to reproduce datasets

## Test Data

For test data and fixtures, use the `tests/fixtures/` directory instead of the main data folder. Test fixtures are version controlled and should be small, focused datasets.

## Docker Integration

Data directories are accessible from the Docker container:

```bash
# Access data in container
docker compose exec dev bash
cd /workspace/data

# Data persists across container rebuilds
docker compose down
docker compose up -d
```
