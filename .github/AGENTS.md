# GitHub Copilot Agents

> For Claude Code, see [.claude/CLAUDE.md](../.claude/CLAUDE.md) | For complete instructions, see [copilot-instructions.md](copilot-instructions.md)

This file defines custom agent profiles for GitHub Copilot with specialized expertise and focused responsibilities.

## Available Agents

### @docs-agent

**Purpose:** Technical writing and documentation  
**File:** `.github/agents/docs-agent.md`

Specializes in:

- Creating clear, concise documentation
- Documenting spatial data workflows
- Writing API documentation
- Creating usage examples

### @test-agent

**Purpose:** Quality assurance and testing  
**File:** `.github/agents/test-agent.md`

Specializes in:

- Writing pytest tests
- Test-driven development (TDD)
- Testing spatial data operations
- Ensuring test coverage

### @data-agent

**Purpose:** Data engineering and spatial analysis  
**File:** `.github/agents/data-agent.md`

Specializes in:

- Spatial data processing with geopandas
- PostGIS database operations
- ETL pipeline design
- Data validation and quality

### @lint-agent

**Purpose:** Code quality and formatting  
**File:** `.github/agents/lint-agent.md`

Specializes in:

- Black code formatting
- flake8 linting
- Import organization
- Style enforcement

## How to Use Agents

### In GitHub Copilot Chat

```
@docs-agent Write documentation for the load_spatial_data function
@test-agent Create tests for the spatial join operation
@data-agent Build a pipeline to process boundary data
@lint-agent Fix formatting issues in src/module.py
```

### In Claude Code (VS Code Extension)

```
Use @docs-agent to document the data processing module
Ask @test-agent to add edge case tests
Have @data-agent optimize the PostGIS query
Request @lint-agent to format all Python files
```

## Project Context

**Type:** Python/R data science and spatial analysis template
**Stack:** Python 3.11+, PostGIS, mamba/conda, Docker, pytest, Black (88 chars)
**Management:** Linear (MCP), GitLab, VS Code
**Workflow:** Explore → Plan → Code → Commit

### Essential Commands

```bash
# Environment (ALWAYS check before Python)
conda env list && conda activate <project-name>

# Code quality (MUST run before commits)
black src/ tests/ && flake8 src/ tests/ && pytest

# Docker
docker compose -f .docker/docker-compose.yml up -d
```

### MCP Integration

This project uses MCP servers running in Docker:

- **Linear** - Issue tracking (`mcp_linear_list_my_issues` to check issues)
- **YouTube Transcript** - Extract and analyze video content
- **Sequential Thinking** - Extended reasoning workflows
- **Perplexity** - Web search and research

## Agent Capabilities

**All agents always:**

- Follow Black formatting (88 char line length)
- Add type hints to all function signatures
- Use early returns (not nested if/else)
- Check conda environment before Python operations
- Follow the project constitution ([.specify/constitution.md](.specify/constitution.md))

**All agents never:**

- Suggest pip venv/virtualenv/poetry/pipenv (ONLY conda/mamba)
- Hardcode secrets or API keys
- Modify `data/raw/` (immutable source data)
- Place Docker/config files in root
- Change code logic when fixing style
- Use emojis in code or documentation

**Ask first:**

- Before modifying database schemas
- Before adding external dependencies
- Before processing large datasets (> 1GB)
- Before changing test structure significantly

**Each agent:**

- Has specialized domain knowledge
- Limited to specific file operations
- Follows role-specific boundaries
- Uses appropriate tools for their domain

## Creating New Agents

To create a new custom agent:

1. Copy the template: `.specify/templates/agent-template.md`
2. Create new file in `.github/agents/`
3. Define name, description, and frontmatter
4. Specify role, knowledge, and boundaries
5. Add examples and commands
6. Update this file with the new agent

Example structure:

```markdown
---
name: your_agent
description: One-line description
---

# Agent Name

You are an expert [role]...

## Your Role
- Responsibility 1
- Responsibility 2

## Commands You Can Use
\`\`\`bash
command --flags
\`\`\`

## Boundaries
✅ Always do...
⚠️ Ask first...
🚫 Never do...
```

## Agent Best Practices

### When to Use Agents

- **Specialized tasks**: Use the agent with relevant expertise
- **Focused work**: Agents stay within their domain
- **Code reviews**: Different agents can review different aspects

### When NOT to Use Agents

- **General questions**: Use the default assistant
- **Cross-domain work**: May need multiple agents or general assistant
- **Exploratory work**: Start with general assistant, then use specific agent

### Combining Agents

For complex tasks, you can work with multiple agents sequentially:

1. `@data-agent` - Build the data pipeline
2. `@test-agent` - Create comprehensive tests
3. `@docs-agent` - Document the implementation
4. `@lint-agent` - Polish the code style

## Workflow Integration

Agents integrate with the Explore → Plan → Code → Commit workflow:

**Explore phase:**

- Use general assistant to research and understand

**Plan phase:**

- Consult relevant agents for domain-specific planning
- `@data-agent` for data architecture
- `@docs-agent` for documentation structure

**Code phase:**

- Use specialized agents for implementation
- `@data-agent` for data processing
- `@test-agent` for test creation

**Commit phase:**

- `@lint-agent` for final formatting
- `@test-agent` to verify all tests pass
- `@docs-agent` for final documentation review

## File Structure

```text
.github/          # Copilot instructions, agents, prompts
  copilot-instructions.md     # Repository-wide instructions
  agents/                     # Custom agent profiles
  instructions/               # Path-specific instructions
  prompts/                    # Spec-kit slash command prompts
.claude/          # Claude Code comprehensive instructions
.specify/         # Spec-kit: specifications, plans, tasks
  constitution.md # Non-negotiable project principles
.docker/          # ALL Docker files (NEVER in root)
data/             # Data files (git-ignored: raw/, input/, output/)
src/project_name/ # Source code (main.py is entry point)
tests/            # Test files
```

**Root MUST only contain:** README.md, .gitignore, pyproject.toml (optional)

## Spec-Kit Integration

Use slash commands for spec-driven development:

```bash
/specify    # Create feature specification
/plan       # Generate technical implementation plan
/tasks      # Break down into actionable tasks
```

See [.github/prompts/](.github/prompts/) for slash command definitions.

## Configuration Files

Agent behavior is controlled by:

- **Repository-wide**: [copilot-instructions.md](copilot-instructions.md)
- **Path-specific**: [instructions/*.instructions.md](instructions/)
- **Constitution**: [.specify/constitution.md](../.specify/constitution.md)
- **Claude Code**: [.claude/CLAUDE.md](../.claude/CLAUDE.md)
- **This file**: .github/AGENTS.md (agent quick reference)

All agents respect these instruction hierarchies.

## Code Example

```python
"""Spatial data loading with type hints, validation, and error handling."""

from pathlib import Path
import geopandas as gpd


def load_spatial_data(
    file_path: Path,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Load and validate spatial data from file.

    Args:
        file_path: Path to spatial data file
        crs: Target coordinate reference system

    Returns:
        GeoDataFrame with validated and transformed data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data is invalid

    Example:
        >>> gdf = load_spatial_data(Path("data/raw/boundaries.geojson"))
        >>> print(gdf.crs)
        EPSG:4326
    """
    # Early return for validation
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    gdf = gpd.read_file(file_path)

    if gdf.empty:
        raise ValueError("File contains no data")

    # Transform CRS if needed
    if gdf.crs and gdf.crs.to_string() != crs:
        gdf = gdf.to_crs(crs)

    return gdf
```

---

**For more information:**

- [GitHub Copilot Documentation](https://docs.github.com/copilot)
- [Project Constitution](../.specify/constitution.md)
- [Claude Code Instructions](../.claude/CLAUDE.md)
- [Complete Copilot Instructions](copilot-instructions.md)
