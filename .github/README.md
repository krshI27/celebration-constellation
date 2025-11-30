# AI Instructions Setup

This directory contains all AI assistant configuration files for GitHub Copilot and Claude Code.

## Structure

```
.github/
├── agents/                          # Custom agent profiles
│   ├── docs-agent.md               # Documentation specialist
│   ├── test-agent.md               # Testing specialist
│   ├── data-agent.md               # Data engineering specialist
│   └── lint-agent.md               # Code quality specialist
├── instructions/                    # Path-specific instructions
│   ├── python-source.instructions.md    # For src/**/*.py
│   └── test-files.instructions.md       # For tests/**/*.py
├── prompts/                        # Spec-kit slash commands
│   ├── specify.prompt.md
│   ├── plan.prompt.md
│   └── tasks.prompt.md
└── copilot-instructions.md         # Repository-wide instructions
```

## Files Overview

### Repository-Wide Instructions

**`.github/copilot-instructions.md`**

- Primary instruction file for GitHub Copilot
- Applies to entire repository
- Includes commands, workflow, code standards
- References other instruction files

### Custom Agents

**`.github/agents/*.md`**

- Specialized agent profiles for specific tasks
- Each agent has focused expertise and boundaries
- Compatible with both Copilot and Claude Code
- Use with `@agent-name` syntax

Available agents:

- `@docs-agent` - Technical writing and documentation
- `@test-agent` - Test creation and quality assurance
- `@data-agent` - Spatial data processing and analysis
- `@lint-agent` - Code formatting and style enforcement

### Path-Specific Instructions

**`.github/instructions/*.instructions.md`**

- Apply to specific file patterns
- More detailed than repository-wide instructions
- Override general instructions for specific file types

Current instructions:

- `python-source.instructions.md` - For `src/**/*.py`
- `test-files.instructions.md` - For `tests/**/*.py`

### Spec-Kit Prompts

**`.github/prompts/*.prompt.md`**

- Slash command definitions for spec-driven development
- Used with `/specify`, `/plan`, `/tasks` commands
- Integrate with GitHub Copilot coding agent

## Related Configuration

### Claude Code Instructions

**`.claude/CLAUDE.md`**

- Primary instruction file for Claude Code
- Treated as immutable system rules
- Higher priority than user prompts
- Modular sections to prevent instruction bleeding

### Project Constitution

**`.specify/constitution.md`**

- Non-negotiable project principles
- Applies to all AI assistants
- Defines Explore → Plan → Code → Commit workflow
- Immutable standards for code quality

### General AI Instructions

**`.config/ai-instructions.md`**

- Comprehensive development guidelines
- Technical standards and patterns
- Complete reference documentation
- Used by both Copilot and Claude Code

## How AI Reads Instructions

### GitHub Copilot

1. Repository-wide: `.github/copilot-instructions.md`
2. Path-specific: `.github/instructions/*.instructions.md` (if path matches)
3. Agent-specific: `.github/agents/*.md` (when `@agent` is invoked)
4. Constitution: `.specify/constitution.md`

### Claude Code

1. System rules: `.claude/CLAUDE.md` (highest priority)
2. Constitution: `.specify/constitution.md`
3. Repository context: `.github/copilot-instructions.md`
4. Path-specific: Reads automatically when working with matching files
5. Agent profiles: `.github/agents/*.md` (can be referenced)

## Usage Examples

### Using Repository-Wide Instructions

```
# Copilot uses automatically
# Claude Code loads CLAUDE.md automatically

Ask about project structure
Request code following standards
```

### Using Custom Agents

```
@docs-agent Write API documentation for load_spatial_data()
@test-agent Create unit tests for buffer_geometries()
@data-agent Build ETL pipeline for boundary data
@lint-agent Fix formatting in src/module.py
```

### Using Spec-Kit Commands

```
/specify     # Create feature specification
/plan        # Create technical plan
/tasks       # Break down into tasks
```

## Creating New Instructions

### New Custom Agent

1. Copy template: `.specify/templates/agent-template.md`
2. Create file in `.github/agents/`
3. Define frontmatter with name and description
4. Specify role, commands, and boundaries
5. Add examples showing good/bad patterns
6. Update `AGENTS.md` with new agent

### New Path-Specific Instructions

1. Create file: `.github/instructions/name.instructions.md`
2. Add frontmatter with `applyTo` glob pattern
3. Define standards specific to those files
4. Include examples and patterns
5. Specify what NOT to do

Example frontmatter:

```markdown
---
applyTo: "src/**/models/*.py"
---

# Model File Instructions
...
```

## Best Practices

### Keep Instructions Focused

- One concern per instruction file
- Clear, specific guidance
- Executable commands, not just descriptions

### Use Examples

- Show good vs bad code patterns
- Include real examples from the project
- Demonstrate expected behavior

### Define Boundaries

- Clearly state what agents can/cannot do
- Use ✅ Always, ⚠️ Ask First, 🚫 Never format
- Prevent agents from modifying wrong files

### Maintain Consistency

- All instructions should align with constitution
- Use same terminology across files
- Reference related instruction files

## Updating Instructions

When updating instruction files:

1. Review current behavior and issues
2. Make targeted, specific changes
3. Test with example prompts
4. Verify agents follow new instructions
5. Update related files if needed
6. Document changes in commit message

Instruction updates should be:

- **Specific**: Target particular behaviors
- **Testable**: Can verify compliance
- **Consistent**: Align with other instructions
- **Minimal**: Only add what's necessary

## Troubleshooting

### Agent Not Following Instructions

1. Check instruction file exists and is readable
2. Verify frontmatter is correct
3. Ensure instructions are specific and clear
4. Add emphasis (ALWAYS, NEVER, MUST)
5. Include more examples

### Conflicting Instructions

1. Check instruction hierarchy
2. Path-specific overrides repository-wide
3. Agent-specific overrides general
4. Claude Code: CLAUDE.md has highest priority
5. Resolve conflicts by updating appropriate file

### Instructions Too Complex

1. Break into smaller, focused sections
2. Use clear markdown headers
3. Separate concerns into different files
4. Keep each section under 100 lines
5. Use bullet points and examples

---

For more information:

- [GitHub Copilot Documentation](https://docs.github.com/copilot)
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Project Constitution](../.specify/constitution.md)
- [Main AGENTS.md](../AGENTS.md)
