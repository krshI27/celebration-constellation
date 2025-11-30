# AI Instructions Hierarchy Map

This document maps the instruction file hierarchy for both GitHub Copilot and Claude Code, showing how different files work together.

## Quick Reference

| AI Tool | Quick Start | Comprehensive Guide |
|---------|-------------|---------------------|
| **GitHub Copilot** | [.github/AGENTS.md](AGENTS.md) | [.github/copilot-instructions.md](copilot-instructions.md) |
| **Claude Code** | [.claude/CLAUDE.md](../.claude/CLAUDE.md) | [.claude/CLAUDE.md](../.claude/CLAUDE.md) |
| **Both** | [.specify/constitution.md](../.specify/constitution.md) | Project principles and standards |

## File Structure & Priority

```text
Repository Root
│
├── .claude/
│   └── CLAUDE.md                      # Claude Code comprehensive instructions (immutable rules)
│
├── .github/
│   ├── AGENTS.md                      # GitHub Copilot agents overview (concise)
│   ├── copilot-instructions.md        # GitHub Copilot repository-wide instructions
│   ├── agents/                        # Custom agent profiles (both AI tools)
│   │   ├── docs-agent.md              # Documentation specialist
│   │   ├── test-agent.md              # Testing specialist
│   │   ├── data-agent.md              # Data engineering specialist
│   │   └── lint-agent.md              # Code quality specialist
│   ├── instructions/                  # Path-specific instructions
│   │   ├── python-source.instructions.md    # For src/**/*.py
│   │   └── test-files.instructions.md       # For tests/**/*.py
│   └── prompts/                       # Spec-kit slash command definitions
│       ├── specify.prompt.md          # /specify command
│       ├── plan.prompt.md             # /plan command
│       ├── tasks.prompt.md            # /tasks command
│       └── ...                        # Other slash commands
│
└── .specify/
    ├── constitution.md                # Non-negotiable project principles (highest authority)
    ├── specifications/                # Feature requirements
    ├── plans/                         # Technical implementation plans
    └── tasks/                         # Actionable task breakdowns
```

## Instruction Priority Hierarchy

**For Claude Code:**

1. **`.claude/CLAUDE.md`** - Comprehensive instructions (loaded automatically at startup)
2. **`.specify/constitution.md`** - Project principles (applies to all work)
3. **User prompts** - Specific instructions (work within established rules)

**For GitHub Copilot:**

1. **`.specify/constitution.md`** - Project principles (highest authority)
2. **`.github/copilot-instructions.md`** - Repository-wide instructions
3. **`.github/instructions/*.instructions.md`** - Path-specific instructions (override general for specific paths)
4. **`.github/agents/*.md`** - Agent-specific behavior (when using @agent)
5. **`.github/AGENTS.md`** - Agent quick reference (found via **/AGENTS.md pattern)

**For Both:**

- **`.specify/constitution.md`** defines non-negotiable standards that both AI tools must follow
- **Custom agents** (`.github/agents/*.md`) work with both Copilot and Claude Code
- **Spec-kit prompts** (`.github/prompts/*.md`) define slash command behavior

## When to Use Which File

### User Reading (Quick Start)

**"I want to quickly understand how to use AI tools in this project"**

- GitHub Copilot users → [.github/AGENTS.md](AGENTS.md)
- Claude Code users → [.claude/CLAUDE.md](../.claude/CLAUDE.md)
- Both → [.specify/constitution.md](../.specify/constitution.md)

### AI Tool Reading (Comprehensive)

**GitHub Copilot loads:**

1. [.github/copilot-instructions.md](copilot-instructions.md) - Always
2. [.github/instructions/*.instructions.md](instructions/) - For matching file paths
3. [.github/agents/*.md](agents/) - When @agent is invoked
4. [.specify/constitution.md](../.specify/constitution.md) - Referenced for standards

**Claude Code loads:**

1. [.claude/CLAUDE.md](../.claude/CLAUDE.md) - Always (comprehensive instructions, loaded at startup)
2. [.specify/constitution.md](../.specify/constitution.md) - Referenced for standards

### Developer Modifying Instructions

**"I want to add a new coding standard"**

→ Update [.specify/constitution.md](../.specify/constitution.md) (affects both AI tools)

**"I want to change how Copilot behaves across the project"**

→ Update [.github/copilot-instructions.md](copilot-instructions.md)

**"I want to change how Claude Code behaves"**

→ Update [.claude/CLAUDE.md](../.claude/CLAUDE.md)

**"I want to create a new specialized agent"**

→ Create new file in [.github/agents/](agents/) using [.specify/templates/agent-template.md](../.specify/templates/agent-template.md)

**"I want to add rules for specific file types"**

→ Create new file in [.github/instructions/](instructions/) with `applyTo: "glob-pattern"` frontmatter

## Content Guidelines

### Quick Reference Files

**.github/AGENTS.md** (~320 lines):

- Agent descriptions
- Project context
- Essential commands
- MCP integration
- Agent boundaries
- Code example

### Comprehensive Guides

**.claude/CLAUDE.md** (~690 lines):

- Complete development workflow
- Detailed code standards
- Extensive examples
- Security practices
- Testing standards
- Database patterns
- Review checklists

**.github/copilot-instructions.md** (~190 lines):

- Code completion standards
- Chat response guidelines
- Slash command usage
- File structure rules
- Example code patterns

### Project Standards

**.specify/constitution.md** (~215 lines):

- Core development principles
- Immutable standards
- Technology stack
- Spec-driven workflow
- Non-negotiable rules
- Review checklists

## MCP Server Integration

All instruction files now reference these MCP servers running in Docker:

- **Linear** - Issue tracking (`mcp_linear_list_my_issues`)
- **YouTube Transcript** - Extract video content
- **Sequential Thinking** - Extended reasoning
- **Perplexity** - Web search and research

Each agent profile includes specific guidance on when and how to use MCP servers.

## Spec-Kit Integration

Slash commands for spec-driven development:

- `/specify` - Create feature specification
- `/plan` - Create technical implementation plan
- `/tasks` - Break down into actionable tasks

Definitions: [.github/prompts/](prompts/)

## Custom Agents

| Agent | File | Purpose | MCP Usage |
|-------|------|---------|-----------|
| @docs-agent | [agents/docs-agent.md](agents/docs-agent.md) | Technical documentation | Perplexity, Linear, YouTube |
| @test-agent | [agents/test-agent.md](agents/test-agent.md) | Test creation & QA | Perplexity, Linear |
| @data-agent | [agents/data-agent.md](agents/data-agent.md) | Spatial data & PostGIS | Perplexity, Linear, Sequential |
| @lint-agent | [agents/lint-agent.md](agents/lint-agent.md) | Code formatting | Perplexity (optional) |

All agents share common boundaries:

- ONLY conda/mamba (never pip venv/virtualenv/poetry)
- Black formatting (88 chars)
- Type hints required
- Early returns (not nested if/else)
- Check conda environment before Python

## Maintenance

**Keep in sync:**

- .github/AGENTS.md should summarize agent profiles in .github/agents/
- Agent boundaries should align with constitution standards
- MCP integration should be consistent across all files
- Code examples should follow the same patterns

**When updating:**

1. Update constitution for project-wide standards
2. Update .claude/CLAUDE.md for Claude Code instructions
3. Update .github/copilot-instructions.md for Copilot instructions
4. Update .github/AGENTS.md to summarize agent changes
5. Update agent profiles for specialized guidance
6. Verify markdown linting passes

---

**Last Updated:** 2025-11-20

**For Questions:**

- GitHub Copilot: See [GitHub Copilot Documentation](https://docs.github.com/copilot)
- Claude Code: See [.claude/CLAUDE.md](../.claude/CLAUDE.md)
- Project: See [.specify/constitution.md](../.specify/constitution.md)
