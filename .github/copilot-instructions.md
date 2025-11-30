# GitHub Copilot Repository Instructions

For complete standards and guidelines, see: [`../.config/ai-instructions.md`](../.config/ai-instructions.md)

For agent-specific behavior, use custom agents: `@docs-agent`, `@test-agent`, `@data-agent`, `@lint-agent`

For Claude Code, see: [`../.claude/CLAUDE.md`](../.claude/CLAUDE.md)

## Project Context

**Type:** Python/R data science and spatial analysis template
**Environment:** WSL with Docker integration
**Stack:** Python 3.11+, PostGIS, mamba/conda, Docker, pytest, Black (88 chars)
**Management:** Linear (MCP), GitLab for source control
**Workflow:** Explore → Plan → Code → Commit (spec-driven development)

### MCP Server Integration

This project uses Model Context Protocol (MCP) servers running in Docker:

- **Linear** - Issue tracking (`mcp_linear_list_my_issues` to check issues)
- **YouTube Transcript** - Extract and analyze video content
- **Sequential Thinking** - Extended reasoning workflows
- **Perplexity** - Web search and research

Always check Linear issues at session start. Create issues assigned to current user (never unassigned).

## Code Completion Standards

When providing inline completions:
- Follow Black formatting (88 char line length)
- Use early returns instead of nested if/else
- Add type hints for function signatures
- Use snake_case naming
- Keep suggestions minimal and clean

## Chat Response Standards

### When user invokes `/explain`
- Focus on purpose and logic flow
- Keep explanations concise
- Avoid unnecessary verbosity

### When user invokes `/fix`
- Identify root cause, not just symptoms
- Suggest minimal fixes
- Explain why the fix works

### When user invokes `/tests`
- Generate pytest tests
- Use arrange-act-assert structure
- Include edge cases
- Focus on behavior, not implementation

### When user invokes `/doc`
- Write docstrings focusing on purpose and usage
- Provide examples, not just parameter descriptions
- Keep documentation concise

## Commands

**Before every commit:**
```bash
black src/ tests/         # Format code (88 char line length)
flake8 src/ tests/        # Check linting
pytest                    # Run tests
```

**Environment management:**
```bash
conda env list            # List available environments
conda activate <project-name>
mamba install <package>   # Prefer mamba over pip
```

**Docker:**
```bash
docker compose -f .docker/docker-compose.yml up -d  # Note: space not hyphen
```

**Spec-kit workflow:**
- Use `/specify` to create feature specifications
- Use `/plan` to create technical implementation plans
- Use `/tasks` to break down into actionable tasks
- Slash commands defined in `.github/prompts/`

## Critical Folder Structure

ALWAYS place files in correct locations:

- `src/project_name/` - Python modules and application code
- `.docker/` - ALL Docker files (NEVER in root)
- `.config/` - Configuration files
- `data/` - Data files (raw/, input/, output/) - excluded from git
- `docs/` - Documentation
- `tests/` - Test files
- `.github/agents/` - Custom agent profiles
- `.github/instructions/` - Path-specific instructions
- `.github/prompts/` - Spec-kit slash command prompts
- `.specify/` - Specifications, plans, tasks (constitution.md, specifications/, plans/, tasks/)
- `.claude/` - Claude Code comprehensive instructions

**Project root should only contain**: README.md, .gitignore, pyproject.toml (optional)

## Environment Rules

MANDATORY:
- ONLY suggest mamba/conda for environments
- NEVER suggest pip venv, virtualenv, poetry, pipenv
- ALWAYS verify environment before running Python
- Use `docker compose` (with space), NOT `docker-compose`

## Development Workflow

Follow this sequence:

1. **Explore**: Research the problem, read context, think through approach
2. **Plan**: Create specification and technical plan (use `/specify` and `/plan`)
3. **Code**: Implement following the plan, write tests
4. **Commit**: Format, lint, test, then commit with concise message

For complex problems, use extended thinking by adding "think" to your prompt.

## Code Generation Rules

When generating code:

- **Use Black formatting** (88 char line length)
- **Add type hints** to function signatures
- **Use early returns** instead of nested if/else
- **Keep functions under 50 lines** when possible
- **Write minimal comments** (only for complex business logic)
- **Validate all user inputs**
- **Use parameterized queries** for databases
- **NEVER hardcode secrets or API keys**

**Example:**
```python
from pathlib import Path
import geopandas as gpd

def load_spatial_data(file_path: Path, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Load and validate spatial data.
    
    Args:
        file_path: Path to spatial data file
        crs: Target coordinate reference system
        
    Returns:
        Validated GeoDataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    # Early return for validation
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    gdf = gpd.read_file(file_path)
    
    if gdf.crs and gdf.crs.to_string() != crs:
        gdf = gdf.to_crs(crs)
    
    return gdf
```

## Custom Agents

Use specialized agents for specific tasks (see [AGENTS.md](AGENTS.md) for details):

- `@docs-agent` - Technical writing and documentation
- `@test-agent` - Test creation and quality assurance
- `@data-agent` - Spatial data processing and PostGIS
- `@lint-agent` - Code formatting and style fixes

**Agent boundaries:**
- All agents follow Black formatting (88 chars) and add type hints
- All agents NEVER suggest pip venv/virtualenv/poetry (ONLY conda/mamba)
- All agents check conda environment before Python operations
- Each agent has specialized domain knowledge and file boundaries

## Path-Specific Instructions

Refer to path-specific instruction files:

- `.github/instructions/python-source.instructions.md` - For `src/**/*.py`
- `.github/instructions/test-files.instructions.md` - For `tests/**/*.py`

## Workspace Context Usage

When user references files:

- Read full context before responding
- Search for similar patterns using `@workspace`
- Maintain consistency with existing code style
- Verify file placement matches template structure
- Check `.specify/` for existing specifications and plans

## Response Guidelines

When answering:
- Be specific and actionable
- Reference actual code patterns in workspace
- Cite files and line numbers when relevant
- Use @workspace, @terminal, @vscode contexts appropriately
- Always verify against template structure before suggesting new files

---

*These instructions apply to all Copilot responses. See `../.config/ai-instructions.md` for complete standards.*
