# 🌌 Drinking Galaxies: Star Constellation Matcher

Find your table's hidden constellation! Upload a photo of bottles, plates, and glasses to discover which star pattern they match in the night sky.

## Overview

**Drinking Galaxies** detects circular objects (bottles, plates, glasses) in table photos and matches their spatial arrangement to star constellations visible from Earth using computer vision and RANSAC-based point cloud registration. The app identifies constellation names using IAU official boundaries and provides educational information about each match.

## Features

✨ **NEW in v0.3.0**: Viewing location calculator - discover where on Earth you can see your matched constellation!

- Shows latitude ranges where constellations are visible
- Provides example cities and geographic regions
- Displays best viewing months based on RA/Dec coordinates
- Optimal viewing locations for maximum altitude

✨ **v0.2.0**: Constellation name identification with full IAU constellation data

## Example Output

### Circle Detection

The app uses quality filtering and non-maximum suppression to detect high-quality circular objects:

![Detection Example](docs/images/detection_example.jpg)

**46 circular objects detected with quality filtering (edge strength + contrast analysis)**

### Constellation Matching

RANSAC-based point cloud matching finds the best-fitting star constellation:

![Match Example](docs/images/match_example.jpg)

**Best match: Score 23.25, 46 matching stars at RA 193.1°, Dec -12.9°**

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

## What's Included

### Directory Structure

```text
python-project-template/
├── .claude/                   # Claude Code instructions
│   └── CLAUDE.md             # Primary Claude Code instructions (auto-loaded)
├── .devcontainer/             # VS Code devcontainer config
├── .docker/                   # Docker development environment
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/                   # GitHub Copilot configuration
│   ├── agents/               # Custom agent profiles
│   │   ├── docs-agent.md     # Documentation specialist
│   │   ├── test-agent.md     # Testing specialist
│   │   ├── data-agent.md     # Data engineering specialist
│   │   └── lint-agent.md     # Code quality specialist
│   ├── instructions/         # Path-specific instructions
│   │   ├── python-source.instructions.md
│   │   └── test-files.instructions.md
│   ├── prompts/              # Spec-kit slash command prompts
│   │   ├── specify.prompt.md
│   │   ├── plan.prompt.md
│   │   └── tasks.prompt.md
│   ├── copilot-instructions.md  # Repository-wide instructions
│   ├── AGENTS.md             # Agent quick reference
│   ├── AI-INSTRUCTIONS-MAP.md   # Complete AI instructions overview
│   └── README.md             # GitHub configuration documentation
├── .specify/                  # Spec-kit files (created by init)
│   ├── constitution.md       # Project principles and standards
│   ├── specifications/       # Requirements and user stories
│   ├── plans/                # Technical implementation plans
│   ├── tasks/                # Implementation tasks
│   ├── artifacts/            # Generated diagrams and templates
│   ├── templates/            # Agent and specification templates
│   └── scripts/              # Automation scripts
├── .vscode/                   # VS Code settings for Python/R
├── .config/                   # Template configuration files
│   ├── ai-instructions.md    # Unified AI assistant instructions
│   ├── environment.yml       # Conda environment
│   └── .env.example          # Environment variables template
├── data/                      # Data files and datasets (excluded from git)
│   ├── raw/                  # Original, unprocessed data
│   ├── input/                # Data prepared for processing/analysis
│   └── output/               # Results, reports, and generated datasets
├── docs/                      # Project documentation
├── src/                       # Source code (package layout)
│   └── project_name/         # Main package directory
```

**Important**: This folder structure should be maintained consistently across all projects. AI assistants are configured to automatically place new files in the appropriate directories.

### Configured Features

- **Python/R Development**: Optimized for data science and spatial analysis
- **Spec-Driven Development**: Integrated GitHub spec-kit for systematic development (requires `specify init` for each AI)
- **AI Assistant Integration**: Configured for both GitHub Copilot and Claude Code with enhanced instructions
- **Custom AI Agents**: Specialized agents for documentation, testing, data processing, and code quality
- **Path-Specific Instructions**: File pattern-based guidance for targeted AI assistance
- **Linear MCP Integration**: Automatic Linear workflow with issue management
- **Conda Environment**: Project-specific mamba/conda environment with development tools
- **Docker Development**: Containerized development with volume mounts
- **VS Code Enhancement**: Python debugging, Black formatting
- **Spatial Data Support**: PostGIS and R integration ready
- **Standardized Structure**: Consistent folder organization across all projects
- **Security Best Practices**: Environment variable management and automated security scanning

### Folder Structure Guidelines

The template enforces a standardized folder structure that should be maintained across all projects:

#### Core Directories

- **`src/`**: All source code in package structure (`src/project_name/`)

- **`data/`**: All data files with clear subdirectory organization
- **`docs/`**: Project documentation, specifications, guides, and templates
- **`.docker/`**: Docker-related files (Dockerfile, compose files, scripts)

#### Python Packaging

- **`main.py`**: Always use `main.py` as the primary entry point file (not package name) for clear project navigation
- **`src/project_name/pyproject.toml.template`**: Template for Python package configuration
- **`pyproject.toml`**: Can be placed in project root OR `src/package_name/` directory
- **Package placement**: Use `src/package_name/pyproject.toml` for cleaner project root

#### Data Organization

- **`data/raw/`**: Original, unprocessed data files
- **`data/input/`**: Data prepared for processing and analysis
- **`data/output/`**: Results, reports, and generated datasets

**Important**: The entire `data/` directory is excluded from version control.

#### AI Assistant Compliance

AI assistants are configured to automatically:

- Place new Python modules in `src/project_name/`
- Create tests in `tests/`
- Store data files in appropriate `data/` subdirectories (raw/, input/, output/)
- Use `tests/fixtures/` for test data (version controlled)
- Place documentation in `docs/`
- Handle Docker configurations in `.docker/`

This structure ensures consistency, maintainability, and seamless collaboration across different projects and team members.

## Usage Scenarios

### Scenario 1: Starting a New Project

1. **Clone or copy this template**

   ```bash
   # Clone the template (use SSH for private repos)
   git clone git@gitlab.com:xr-future-forests-lab/vscode-project-template.git my-new-project
   cd my-new-project

   # Or copy files manually
   cp -r vscode-project-template/ my-new-project/
   cd my-new-project/
   ```

2. **Customize for your project**

   ```bash
   # Set your project name
   PROJECT_NAME="my-new-project"

   # Replace PROJECT_NAME placeholders in all files
   find . -type f -name "*.yml" -o -name "*.json" -o -name "*.md" -o -name "Dockerfile" | \
       xargs sed -i "s/PROJECT_NAME/$PROJECT_NAME/g"

   # Rename source directory
   mv src/project_name src/$PROJECT_NAME
   ```

3. **Set up environment**

   ```bash
   # Create conda environment
   mamba env create -f .config/environment.yml
   conda activate $PROJECT_NAME

   # Or use Docker
   cd .docker
   docker compose up -d
   docker compose exec dev bash
   ```

4. **Initialize git repository**

   ```bash
   rm -rf .git  # Remove template git history
   git init
   git add .
   git commit -m "feat: initial project setup

   - Add conda environment with development dependencies
   - Configure VS Code with Copilot and Claude Code integration
   - Set up Docker development environment
   - Include Linear MCP workflow integration"
   ```

### Scenario 2: Adding Template to Existing Repository

#### Option A: Direct Copy Approach

1. **Clone your existing repository**

   ```bash
   git clone git@gitlab.com:your-username/your-existing-repo.git
   cd your-existing-repo
   ```

2. **Copy template files selectively**

   ```bash
   # Copy development environment files
   cp -r /path/to/vscode-project-template/.vscode ./
   cp -r /path/to/vscode-project-template/.docker ./
   cp -r /path/to/vscode-project-template/.devcontainer ./
   cp -r /path/to/vscode-project-template/.github ./
   cp -r /path/to/vscode-project-template/.claude ./

   # Copy folder structure (if not existing)
   cp -r /path/to/vscode-project-template/data ./
   cp -r /path/to/vscode-project-template/docs ./
   cp -r /path/to/vscode-project-template/src ./

   # Copy configuration files
   cp -r /path/to/vscode-project-template/.config ./
   cp /path/to/vscode-project-template/.gitignore ./
   ```

   **Note**: If you already have `data/`, `docs/`, or `src/` folders, you can selectively copy only the missing ones or merge the contents as needed.

3. **Customize for your project**

   ```bash
   # Replace PROJECT_NAME with your actual project name
   PROJECT_NAME="your-project-name"
   find . -type f -name "*.yml" -o -name "*.json" -o -name "*.md" -o -name "Dockerfile" | \
       xargs sed -i "s/PROJECT_NAME/$PROJECT_NAME/g"
   ```

4. **Update environment.yml**

   ```bash
   # Edit environment.yml to match your project name
   sed -i "s/name: PROJECT_NAME/name: $PROJECT_NAME/" .config/environment.yml
   ```

5. **Commit the changes**

   ```bash
   git add .
   git commit -m "feat: add development environment template

   - Add VS Code configuration with Copilot and Claude Code settings
   - Add Docker development environment
   - Add conda environment configuration
   - Add devcontainer support
   - Add standardized folder structure (data/, docs/, src/)"
   ```

#### Option B: Sub-repository Approach

For ongoing template updates and easier maintenance, you can integrate the template as a sub-repository:

1. **Clone your existing repository**

   ```bash
   git clone git@gitlab.com:your-username/your-existing-repo.git
   cd your-existing-repo
   ```

2. **Clone the template repository**

   ```bash
   # Clone the template to a temporary directory
   git clone git@gitlab.com:xr-future-forests-lab/vscode-project-template.git /tmp/vscode-template

   # Or use HTTPS if you don't have SSH access
   # git clone https://gitlab.com/xr-future-forests-lab/vscode-project-template.git /tmp/vscode-template
   ```

3. **Copy template files to your project**

   ```bash
   # Copy configuration directories
   cp -r /tmp/vscode-template/.vscode ./
   cp -r /tmp/vscode-template/.docker ./
   cp -r /tmp/vscode-template/.devcontainer ./
   cp -r /tmp/vscode-template/.github ./
   cp -r /tmp/vscode-template/.claude ./
   cp -r /tmp/vscode-template/.config ./

   # Copy and merge .gitignore
   cat /tmp/vscode-template/.gitignore >> .gitignore

   # Create folder structure if needed
   mkdir -p data docs src
   cp -r /tmp/vscode-template/data/* ./data/ 2>/dev/null || true
   cp -r /tmp/vscode-template/docs/* ./docs/ 2>/dev/null || true
   cp -r /tmp/vscode-template/src/* ./src/ 2>/dev/null || true

   # Clean up temporary directory
   rm -rf /tmp/vscode-template
   ```

4. **Update template when needed**

   ```bash
   # Update to latest template version
   git clone git@gitlab.com:xr-future-forests-lab/vscode-project-template.git /tmp/vscode-template

   # Selectively copy updated files (be careful not to overwrite customizations)
   cp -r /tmp/vscode-template/.docker ./
   cp -r /tmp/vscode-template/.vscode ./
   # ... copy other directories as needed

   # Clean up
   rm -rf /tmp/vscode-template

   git add .
   git commit -m "update: sync with latest template version"
   ```

5. **Customize for your project**

   ```bash
   # Replace PROJECT_NAME with your actual project name
   PROJECT_NAME="your-project-name"
   sed -i "s/name: PROJECT_NAME/name: $PROJECT_NAME/" .config/environment.yml
   ```

6. **Commit the integration**

   ```bash
   git add .
   git commit -m "feat: integrate development template

   - Add template files for standardized development environment
   - Include Docker, VS Code, and configuration setup"
   - Link development environment configurations
   - Add standardized folder structure
   - Enable template version tracking"
   ```

## Development Workflow

### Spec-Driven Development

This template includes GitHub's spec-kit for systematic development. The `.specify/` folder structure is created during initialization:

#### 1. Install spec-kit (one-time setup per machine)

```bash
# Install the spec-kit CLI tool globally (requires uv)
pip install uv  # If not already installed
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git

# Verify installation
specify --help
```

#### 2. Initialize spec-kit (required for each AI assistant)

**Important**: The `specify init` command can only configure one AI assistant at a time. You need to run it separately for each assistant you plan to use.

```bash
# For GitHub Copilot
specify init
# Select GitHub Copilot when prompted

# For Claude Code (separate initialization)
specify init
# Select Claude Code when prompted
```

This creates the `.specify/` folder with:

- `constitution.md` - Project principles and standards
- `specifications/` - Requirements and user stories
- `plans/` - Technical implementation plans
- `tasks/` - Implementation tasks
- `artifacts/` - Generated diagrams and templates

#### 3. Create Specifications

```bash
# Use spec-kit commands or edit manually
specify specification

# Or edit .specify/specifications/PROJECT_NAME.md directly
```

#### 4. Generate Technical Plans

```bash
specify plan
# Creates detailed implementation plans in .specify/plans/
```

#### 5. Break Into Tasks

```bash
specify tasks
# Generates actionable tasks in .specify/tasks/
```

#### 6. Implement with AI

```bash
specify implement
# Execute tasks with AI assistance
```

### Environment Setup

#### Option 1: Conda Environment

```bash
# Create and activate environment
mamba env create -f .config/environment.yml
conda activate your-project-name

# Install development tools
pip install -e .
```

#### Option 2: Docker Environment

```bash
# Start development environment
cd .docker
docker compose up -d

# Access container shell
docker compose exec dev bash
```

### Development Commands

#### Local Development

```bash
# Testing
pytest

# Code formatting and quality checks
black src/ tests/
flake8 src/ tests/

# Manual formatting
black src/ tests/

# Linting
flake8 src/ tests/

# Security scanning
bandit -r src/
```

#### Docker Development

```bash
# Using docker compose directly
docker compose exec dev pytest
docker compose exec dev black src/ tests/
```

### Available Services

- **Jupyter Lab**: <http://localhost:8888>
- **Dev Server**: <http://localhost:8000> (FastAPI/Flask)
- **Alternative Server**: <http://localhost:5000>

### Custom AI Agents

The template includes specialized AI agents for focused tasks:

**Available Agents:**

- **@docs-agent** - Technical writing and documentation specialist
- **@test-agent** - Quality assurance and testing specialist
- **@data-agent** - Data engineering and spatial analysis specialist
- **@lint-agent** - Code quality and formatting specialist

**Usage:**

```bash
# In GitHub Copilot Chat or Claude Code
@docs-agent Write API documentation for load_spatial_data function
@test-agent Create tests for the spatial join operation
@data-agent Build a pipeline to process boundary data
@lint-agent Fix formatting issues in src/module.py
```

**Creating New Agents:**

1. Copy template from `.specify/templates/agent-template.md`
2. Create new file in `.github/agents/your-agent.md`
3. Define role, boundaries, and examples
4. Update `.github/AGENTS.md` with new agent

See [.github/AGENTS.md](.github/AGENTS.md) for complete agent documentation.

## Configuration

### Conda Environment

Edit `.config/environment.yml` to customize dependencies:

```yaml
name: your-project
dependencies:
  - python=3.11
  - pandas
  - geopandas  # for spatial data
  - your-packages
```

### AI Assistant Instructions

**For GitHub Copilot:** Modify `.github/copilot-instructions.md`
**For Claude Code:** Modify `.claude/CLAUDE.md`

**Note**: Claude Code automatically loads `.claude/CLAUDE.md` when starting. You can create `.claude/CLAUDE.local.md` for personal customizations (automatically gitignored).

Both should include:

- Project-specific coding standards
- Technology stack preferences
- Linear project context
- Team-specific guidelines
- No emoji/icons policy

**Custom Agents:** Define specialized agents in `.github/agents/` for focused tasks (documentation, testing, data processing, code quality)

**Path-Specific Instructions:** Add file pattern-specific instructions in `.github/instructions/` for targeted guidance

### Docker Environment

Customize `.docker/Dockerfile` and `.docker/docker-compose.yml` for:

- Additional system dependencies
- Database connections (PostgreSQL/PostGIS)
- R environment setup
- Volume mounts for data

### VS Code Settings

The `.vscode/settings.json` includes:

- Python interpreter configuration
- Black formatting (88 char line length)
- pytest integration
- Copilot settings
- WSL terminal configuration

## Customization Points

1. **.config/environment.yml**: Add/remove Python packages
2. **.github/copilot-instructions.md**: Customize Copilot behavior for your project
3. **.claude/CLAUDE.md**: Customize Claude Code behavior (auto-loaded)
4. **.github/agents/**: Create specialized agents for specific tasks
5. **.github/instructions/**: Add path-specific instructions
6. **.docker/Dockerfile**: Add system dependencies
7. **.vscode/settings.json**: Adjust editor preferences
8. **.config/.env.example**: Add project-specific environment variables
9. **src/project_name/pyproject.toml.template**: Template for Python package configuration

### Python Package Configuration (pyproject.toml)

The template includes a comprehensive `src/project_name/pyproject.toml.template` file for Python packaging. This template can be used in two ways:

#### Option 1: Project Root (Traditional)

```bash
# Copy template to project root
cp src/project_name/pyproject.toml.template pyproject.toml

# Customize for your project
# Edit: project name, version, dependencies, etc.
```

#### Option 2: Package Directory (Recommended for Clean Root)

```bash
# Copy template to package directory
cp src/project_name/pyproject.toml.template src/your-package-name/pyproject.toml

# Customize for your specific package
# This approach keeps the project root clean and supports multiple packages
```

#### Template Features

The pyproject.toml template includes:

- **Modern setuptools configuration** with PEP 621 metadata
- **Development dependencies** (pytest, black, mypy, flake8)
- **Optional dependency groups** (dev, test, docs)
- **Tool configurations** for Black, pytest, coverage, and mypy
- **Package discovery** configured for `src/` layout
- **Comprehensive metadata** fields (authors, classifiers, URLs)
- **Console scripts** section for CLI tools

#### Customization Steps

1. **Replace placeholders**:
   - `project-name` → your actual package name
   - `Your Name` and email → your details
   - URLs → your repository and documentation links

2. **Update dependencies**:
   - Add your core dependencies to `dependencies`
   - Modify optional dependency groups as needed
   - Adjust Python version requirements

3. **Configure tools**:
   - Black formatting settings (line length, target versions)
   - pytest configuration (test paths, options)
   - MyPy type checking settings
   - Coverage reporting options

4. **Set up console scripts** (if needed):

   ```toml
   [project.scripts]
   my-cli-tool = "package_name.cli:main"
   ```

This template follows modern Python packaging standards and integrates seamlessly with the conda/mamba environment management enforced by this template.

### Environment Variables (.config/.env.example)

The `.config/.env.example` file serves as a template for environment-specific configuration. This file should be copied to `.env` and customized for your local development environment.

#### Purpose and Usage

Environment variables are used to:

- **Store sensitive information** (API keys, passwords, tokens) outside of code
- **Configure different environments** (development, staging, production)
- **Manage external service connections** (databases, APIs, cloud services)
- **Control application behavior** without code changes

#### Setup Process

1. **Copy the template**:

   ```bash
   cp .config/.env.example .env
   ```

2. **Customize your .env file**:

   ```bash
   # Edit with your actual values
   nano .env  # or use your preferred editor
   ```

3. **Keep .env private**: The `.env` file is already excluded from version control via `.gitignore`

#### Common Variable Types

**Project Configuration**:

```bash
PROJECT_NAME=my-awesome-project
ENVIRONMENT=development  # development, staging, production
DEBUG=true              # Enable debug mode
LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
```

**Database Connections**:

```bash
DATABASE_URL=postgresql://username:password@localhost:5432/mydb
DB_HOST=localhost
DB_PORT=5432
DB_NAME=myproject
DB_USER=myuser
DB_PASSWORD=secure_password
```

**API Keys and Tokens**:

```bash
API_KEY=sk-1234567890abcdef  # External API key
OPENAI_API_KEY=sk-...        # OpenAI API key
GITHUB_TOKEN=ghp_...         # GitHub personal access token
AWS_ACCESS_KEY_ID=AKIA...    # AWS credentials
AWS_SECRET_ACCESS_KEY=...
```

**External Services**:

```bash
REDIS_URL=redis://localhost:6379
ELASTICSEARCH_URL=http://localhost:9200
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=app-specific-password
```

**Application Settings**:

```bash
SECRET_KEY=your-super-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
MAX_UPLOAD_SIZE=10485760  # 10MB in bytes
```

#### Security Best Practices

- **Never commit .env files** to version control
- **Use strong, unique passwords** and rotate them regularly
- **Limit permissions** - only include variables needed for each environment
- **Use secrets management** in production (AWS Secrets Manager, Azure Key Vault, etc.)
- **Validate environment variables** in your application startup code

#### Docker Integration

Environment variables work seamlessly with Docker:

```bash
# Docker will automatically load .env file
docker compose up -d

# Or explicitly specify env file
docker compose --env-file .env.production up -d
```

#### Python Usage Example

```python
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access variables with defaults
PROJECT_NAME = os.getenv('PROJECT_NAME', 'default-project')
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
DATABASE_URL = os.getenv('DATABASE_URL')

# Validate required variables
required_vars = ['PROJECT_NAME', 'DATABASE_URL']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {missing_vars}")
```

#### Different Environments

You can maintain multiple environment files:

```bash
.env.example         # Template (version controlled)
.env                # Local development (not version controlled)
.env.staging        # Staging environment
.env.production     # Production environment
.env.test          # Testing environment
```

Load specific environment files:

```bash
# In Python
load_dotenv('.env.staging')

# With Docker
docker compose --env-file .env.production up -d
```

## Key Features

### WSL + Windows Integration

- **Terminal**: Configured for WSL Ubuntu
- **Docker**: WSL Docker integration
- **Git**: Meaningful commits and clean history

### Python Best Practices

- **Black Formatting**: 88 character line length
- **pytest Testing**: Comprehensive test structure
- **Type Hints**: Encouraged for better code clarity
- **Minimal Comments**: Self-documenting code approach

### Linear MCP Workflow

- **Issue Context**: Automatic Linear integration in AI assistants
- **Team Coordination**: Built-in team ID and project references
- **Progress Tracking**: MCP commands for status updates

### Data Science Ready

- **Spatial Data**: PostGIS integration patterns
- **R Integration**: R environment configuration
- **Data Handling**: Structured data/ directory
- **Analysis Tools**: Jupyter, pandas, geopandas ready

## Prerequisites

- **Python 3.11+** and **mamba/conda** for environment management
- **Docker** and **docker compose** for containerized development
- **VS Code** with Python/R extensions for IDE features
- **spec-kit CLI** for spec-driven development workflow
- **Git** for version control
- **WSL2** (if on Windows) for Linux compatibility

## Technical Stack

- **Languages**: Python 3.11+, R (optional)
- **Package Manager**: mamba/conda
- **Testing**: pytest with coverage
- **Formatting**: Black (88 chars), isort for imports
- **Linting**: flake8 with Black-compatible settings
- **Security**: bandit for vulnerability scanning
- **Containers**: Docker with docker compose
- **Database**: PostgreSQL/PostGIS ready
- **IDE**: VS Code with Python/R extensions

## Notes

- The template is designed to be non-invasive to existing code
- You can pick and choose which parts to integrate
- All configurations use relative paths and will work with your existing structure
- The setup maintains your existing git history
- Spec-kit is installed once per machine, not per project

---

*This template provides a complete Python development environment with AI assistant integration, optimized for data science, spatial analysis, and collaborative development.*
