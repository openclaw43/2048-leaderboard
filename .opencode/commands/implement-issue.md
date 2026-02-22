# /implement-issue

Implements a feature from a GitHub issue by reading the issue, creating a branch, developing the feature, and opening a PR.

## Usage

```bash
opencode /implement-issue <issue-number>
```

## Parameters

- `issue-number` (required): The GitHub issue number to implement

## Behavior

When this command is invoked with an issue number:

1. **Fetch Issue Details**
   ```bash
   gh issue view <issue-number> --json number,title,body,labels
   ```
   - Extract issue title and description
   - Parse acceptance criteria
   - Identify relevant files to modify

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/issue-<number>-<short-title>
   ```
   - Branch name format: `feature/issue-{number}-{kebab-case-title}`
   - Base branch: current branch (typically main/master)

3. **Analyze Issue Requirements**
   - Read the algorithm design section
   - Identify files to create/modify
   - Understand acceptance criteria
   - Extract code examples if present

4. **Implement the Feature**
   - Create/modify files as specified in issue
   - Follow existing code patterns and style
   - Add appropriate imports and exports
   - Include docstrings and type hints
   - Match the project's architecture

5. **Add Tests**
   - Create tests in `test_game.py` or appropriate file
   - Follow existing test patterns
   - Ensure all acceptance criteria are tested

6. **Run Verification**
   ```bash
   uv run pytest test_game.py -v
   uv run python benchmark.py --json
   ```
   - All tests must pass
   - Benchmark should meet expected performance

7. **Commit Changes**
   ```bash
   git add -A
   git commit -m "feat: implement <issue-title> (#<issue-number>)
   
   - <change 1>
   - <change 2>
   - <change 3>
   
   Closes #<issue-number>"
   ```

8. **Push Branch**
   ```bash
   git push -u origin feature/issue-<number>-<short-title>
   ```

9. **Open Pull Request**
   ```bash
   gh pr create \
     --title "feat: <issue-title>" \
     --body "## Summary
   Implements #<issue-number>
   
   ## Changes
   - <description of changes>
   
   ## Acceptance Criteria
   - [x] <criterion 1>
   - [x] <criterion 2>
   - [x] <criterion 3>
   
   ## Test Results
   - All tests passing
   - Benchmark results: <score>
   
   ## Notes
   <any implementation notes>" \
     --base main
   ```

## Requirements

- GitHub CLI (`gh`) installed and authenticated
- UV installed for dependency management
- Running from repository root
- Issue should have:
  - Clear algorithm design
  - File structure requirements
  - Acceptance criteria
  - Expected performance metrics

## Example

```bash
# Implement the Corner Strategy agent (issue #2)
opencode /implement-issue 2

# This will:
# 1. Read issue #2 details
# 2. Create branch: feature/issue-2-corner-strategy
# 3. Implement CornerAgent in game2048/agents/corner_agent.py
# 4. Add tests
# 5. Run verification
# 6. Open PR
```

## Error Handling

- If issue not found: Display error and exit
- If branch already exists: Use alternative name or fail gracefully
- If tests fail: Report failures and do not create PR
- If push fails: Display git error message

## Notes

- Always follow existing code patterns in the repository
- Maintain type hints and docstrings
- Update `__init__.py` files for new exports
- Ensure agents are registered with `@register_agent`
- Add the agent to benchmark comparisons
