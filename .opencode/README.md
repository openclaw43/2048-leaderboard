# OpenCode Custom Commands

This repository includes custom slash commands for OpenCode to streamline development workflows.

## Setup

These commands are automatically available when using OpenCode with this repository. The command definitions are in `.opencode/commands/`.

## Commands

### `/implement-issue <issue-number>`

Automatically implements a GitHub issue by:
1. Reading the issue details
2. Creating a feature branch
3. Implementing the feature (manual step)
4. Running tests and benchmarks
5. Committing changes
6. Opening a pull request

**Usage:**
```bash
opencode /implement-issue 2
```

**Example:**
```bash
# Implement the Corner Strategy agent from issue #2
opencode /implement-issue 2

# This will:
# - Read issue #2
# - Create branch: feature/issue-2-corner-strategy  
# - Guide you through implementation
# - Commit changes
# - Open PR
```

**Requirements:**
- GitHub CLI (`gh`) authenticated
- UV installed (`pip install uv`)
- Repository cloned locally

## Manual Usage (Script)

You can also use the script directly:

```bash
python scripts/implement_issue.py <issue-number>
```

This is useful for:
- CI/CD pipelines
- Automation
- When OpenCode is not available

## Adding New Commands

To add a custom slash command:

1. Create a markdown file in `.opencode/commands/<command-name>.md`
2. Document the command behavior, parameters, and examples
3. Optionally create a Python script in `scripts/` for standalone execution
4. Update this README

### Command Structure

```markdown
# /command-name

Brief description.

## Usage
```bash
opencode /command-name [args]
```

## Behavior
1. Step 1
2. Step 2
3. Step 3

## Requirements
- Requirement 1
```

## Directory Structure

```
.opencode/
├── README.md              # This file
└── commands/
    └── implement-issue.md # Command definition

scripts/
└── implement_issue.py     # Executable script
```

## Notes

- Commands are declarative - they describe what OpenCode should do
- The actual implementation is handled by OpenCode's AI
- Scripts provide a fallback for non-OpenCode environments
- All commands follow the repository's code style and patterns
