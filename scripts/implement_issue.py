#!/usr/bin/env python3
"""
OpenCode Custom Command: /implement-issue

Implements a feature from a GitHub issue by:
1. Reading the issue
2. Creating a branch
3. Implementing the feature
4. Opening a PR

Usage:
    python scripts/implement_issue.py <issue-number>
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_command(cmd: list[str], check: bool = True) -> tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False
    )
    if check and result.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    return result.returncode, result.stdout, result.stderr


def get_issue_details(issue_number: int) -> dict:
    """Fetch issue details from GitHub."""
    code, stdout, stderr = run_command(
        ["gh", "issue", "view", str(issue_number), "--json", "number,title,body,labels"],
        check=False
    )
    if code != 0:
        print(f"Error: Could not fetch issue #{issue_number}")
        print(f"Make sure you're in a git repository and gh CLI is authenticated")
        sys.exit(1)
    
    return json.loads(stdout)


def create_branch_name(issue_number: int, title: str) -> str:
    """Create a valid branch name from issue number and title."""
    # Convert title to kebab-case, limit length
    short_title = re.sub(r'[^\w\s-]', '', title.lower())
    short_title = re.sub(r'[-\s]+', '-', short_title)
    short_title = short_title[:50].strip('-')
    return f"feature/issue-{issue_number}-{short_title}"


def create_branch(branch_name: str) -> None:
    """Create and checkout a new branch."""
    print(f"Creating branch: {branch_name}")
    run_command(["git", "checkout", "-b", branch_name])


def parse_issue_body(body: str) -> dict:
    """Parse issue body to extract key information."""
    info = {
        "algorithm": "",
        "files": [],
        "acceptance_criteria": [],
        "code_examples": []
    }
    
    # Extract code examples
    code_blocks = re.findall(r'```python\n(.*?)```', body, re.DOTALL)
    info["code_examples"] = code_blocks
    
    # Try to find files to modify
    file_patterns = [
        r'`(\S+\.py)`',
        r'\*\*`(\S+\.py)`\*\*',
        r'\[\]`(\S+\.py)`',
    ]
    for pattern in file_patterns:
        matches = re.findall(pattern, body)
        info["files"].extend(matches)
    
    # Extract acceptance criteria
    if "## Acceptance Criteria" in body:
        criteria_section = body.split("## Acceptance Criteria")[1].split("##")[0]
        criteria = re.findall(r'- \[ \] (.+)', criteria_section)
        info["acceptance_criteria"] = criteria
    
    return info


def commit_changes(issue_number: int, title: str, changes: list[str]) -> None:
    """Commit changes with a descriptive message."""
    change_list = "\n".join(f"- {c}" for c in changes)
    message = f"""feat: {title} (#{issue_number})

{change_list}

Closes #{issue_number}"""
    
    run_command(["git", "add", "-A"])
    run_command(["git", "commit", "-m", message])


def push_branch(branch_name: str) -> None:
    """Push branch to origin."""
    print(f"Pushing branch: {branch_name}")
    run_command(["git", "push", "-u", "origin", branch_name])


def create_pull_request(issue_number: int, title: str, body: str, changes: list[str], criteria: list[str]) -> str:
    """Create a pull request on GitHub."""
    
    # Build PR body
    criteria_checklist = "\n".join(f"- [x] {c}" for c in criteria)
    changes_list = "\n".join(f"- {c}" for c in changes)
    
    pr_body = f"""## Summary
Implements #{issue_number}

## Changes
{changes_list}

## Acceptance Criteria
{criteria_checklist}

## Test Results
<!-- To be filled by implementer -->
- [ ] All tests passing
- [ ] Benchmark results documented

## Notes
<!-- Any implementation notes -->
"""
    
    code, stdout, stderr = run_command([
        "gh", "pr", "create",
        "--title", f"feat: {title}",
        "--body", pr_body,
        "--base", "main"
    ])
    
    # Extract PR URL from stdout
    pr_url = stdout.strip()
    return pr_url


def main():
    parser = argparse.ArgumentParser(
        description="Implement a feature from a GitHub issue"
    )
    parser.add_argument(
        "issue_number",
        type=int,
        help="The GitHub issue number to implement"
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Implementing issue #{args.issue_number}")
    print("=" * 60)
    
    # Step 1: Fetch issue details
    print("\n📖 Fetching issue details...")
    issue = get_issue_details(args.issue_number)
    title = issue["title"]
    body = issue["body"]
    print(f"Title: {title}")
    
    # Step 2: Parse issue body
    print("\n🔍 Parsing issue requirements...")
    info = parse_issue_body(body)
    if info["files"]:
        print(f"Files to modify: {', '.join(info['files'])}")
    if info["acceptance_criteria"]:
        print(f"Acceptance criteria: {len(info['acceptance_criteria'])} items")
    
    # Step 3: Create branch
    print("\n🌿 Creating feature branch...")
    branch_name = create_branch_name(args.issue_number, title)
    create_branch(branch_name)
    
    # Step 4: Implementation placeholder
    print("\n⚠️  Implementation step")
    print("=" * 60)
    print("The automated implementation would:")
    print("1. Create/modify the necessary files")
    print("2. Add tests for the feature")
    print("3. Run tests to verify implementation")
    print("4. Run benchmarks to validate performance")
    print()
    print("Currently, this requires manual implementation.")
    print("Follow the issue description and acceptance criteria.")
    print("=" * 60)
    
    # Wait for user confirmation
    print("\n⏳ Please implement the feature manually.")
    print("When ready, the script will commit and create a PR.")
    
    # In a real implementation, OpenCode would do the work here
    # For now, we ask the user to confirm
    response = input("\nHave you completed the implementation? (yes/no): ")
    if response.lower() != "yes":
        print("Exiting. Run this script again when ready.")
        sys.exit(0)
    
    # Step 5: Commit changes
    print("\n💾 Committing changes...")
    changes = ["Add implementation files", "Add tests", "Update documentation"]
    commit_changes(args.issue_number, title, changes)
    
    # Step 6: Push branch
    push_branch(branch_name)
    
    # Step 7: Create PR
    print("\n📤 Creating pull request...")
    pr_url = create_pull_request(
        args.issue_number,
        title,
        body,
        changes,
        info["acceptance_criteria"]
    )
    
    print("\n" + "=" * 60)
    print("✅ Implementation workflow complete!")
    print(f"Branch: {branch_name}")
    print(f"PR: {pr_url}")
    print("=" * 60)


if __name__ == "__main__":
    main()
