# Hidden Folders in Repository

This document explains which hidden folders are tracked in git and which are excluded for public release.

## ✅ Tracked (Version Controlled)

These folders are essential for the project and safe to share publicly:

### `.streamlit/`
- **Purpose**: Streamlit app configuration
- **Contents**: `config.toml` with PWA theme settings
- **Why public**: Required for consistent app appearance and functionality

### `.github/`
- **Purpose**: GitHub workflows and CI/CD
- **Contents**: `.gitlab-ci.yml` for Black linting
- **Why public**: Transparent CI/CD process, helps contributors understand code quality checks

### `.git/`
- **Purpose**: Git version control metadata
- **Why public**: Automatically managed by Git, contains commit history

## 🚫 Excluded (Git-Ignored)

These folders contain local development configs and are excluded via `.gitignore`:

### `.vscode/`
- **Purpose**: VS Code editor settings
- **Why excluded**: User-specific IDE preferences

### `.config/`
- **Purpose**: Template configuration files
- **Why excluded**: Contains development template configs not needed for this standalone project

### `.specify/`
- **Purpose**: Spec-kit specifications and plans
- **Why excluded**: Internal development workflow tracking

### `.claude/`
- **Purpose**: Claude Code AI instructions
- **Why excluded**: Local AI assistant configuration

### `.devcontainer/`
- **Purpose**: VS Code DevContainer configuration
- **Why excluded**: Optional development environment setup

### `.docker/`
- **Purpose**: Docker development environment
- **Why excluded**: Not used in Streamlit Cloud deployment

### `.pytest_cache/`
- **Purpose**: Pytest cache files
- **Why excluded**: Generated at runtime, no need to version

### `__pycache__/`
- **Purpose**: Python bytecode cache
- **Why excluded**: Generated at runtime, no need to version

### `specs/`
- **Purpose**: Development specifications
- **Why excluded**: Internal planning documents

## 🔐 Security Considerations

Excluded folders may contain:
- Local file paths
- Development-specific configurations
- Internal workflow documentation
- User-specific settings

None of these contain secrets or credentials, but excluding them keeps the public repo clean and focused on the application code.

## 📝 Notes for Contributors

If you're forking or cloning this repo:
- You won't have the excluded folders - that's intentional
- Create your own `.vscode/settings.json` if needed
- The app will work without any of the excluded folders
- All runtime data (star cache) is created automatically in `~/.celebration_constellation/`
