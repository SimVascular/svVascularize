# Remove Binder Sandbox Implementation Plan

**Goal:** Retire the repository's Binder sandbox configuration and remove the `binder/` directory.

**Architecture:** Delete the three files that configure the Binder environment. The current main branch already has no Binder launch links or integrations, so the implementation is a configuration cleanup with no application changes.

**Tech Stack:** Git, shell configuration, Python dependency files, static HTML documentation, GitHub Actions.

**Spec:** The user's September 15, 2026 request to create the `remove-binder` worktree and branch and plan removal of the now-unneeded Binder sandbox and folder; the bounded design is recorded below.

## Workspace and baseline

- Branch: `remove-binder`.
- Worktree: `/home/zack/Downloads/svVascularize/.worktrees/remove-binder`.
- Base: freshly fetched `upstream/main`, commit `688bcb6fc715be516d427a8ac7cb3b97663c17b2`.
- The worktree was clean before this plan was added.
- This independent branch starts from main because the separate `docs/website-redesign` branch already removed these files in commit `8eb79e7`.
- Implementation complete: the three Binder files are removed and the planned checks pass.

## Bounded design

| File | Current purpose | Planned action |
| --- | --- | --- |
| `binder/postBuild` | Installs the checkout with `python -m pip install -e . --no-deps` | Delete |
| `binder/requirements.txt` | Lists dependencies for the Binder environment | Delete |
| `binder/runtime.txt` | Selects Python 3.11 for Binder | Delete |

The baseline audit found no Binder, mybinder, or repo2docker references in tracked text files, including application code, documentation, README, and workflows. There are no tracked notebooks or alternative `.binder/` configuration files. Package installation reads the root `requirements.txt` through `setup.py`; GitHub Pages publishes `docs/` directly.

## Global constraints

- Perform the implementation in the `remove-binder` worktree.
- Keep the existing checkout and its website redesign work intact.
- Limit the implementation to retiring Binder; preserve package APIs, numerical behavior, dependencies, and local installation instructions.
- Retain the root `requirements.txt`, `setup.py`, `pyproject.toml`, and `MANIFEST.in`.
- Preserve existing CI and GitHub Pages workflows.
- Do not add a replacement hosted sandbox or new dependencies.
- Do not rewrite Git history; historical versions remain available.
- Author every new commit as Zachary Sexton (`zsexton@stanford.edu`).

---

### Task 1: Remove Binder configuration and verify the result

**Files:**

- Delete: `binder/postBuild`.
- Delete: `binder/requirements.txt`.
- Delete: `binder/runtime.txt`.
- Planning record: `docs/superpowers/plans/2026-09-15-remove-binder.md`.

**Interfaces:**

- Consumes: the three Binder configuration files on the recorded base commit.
- Produces: a repository without dedicated Binder configuration or active Binder launch references.

- [x] **Step 1: Confirm the branch and repeat the reference audit.**

  Run from the worktree root:

  ```bash
  git branch --show-current
  git status --short
  git ls-files -- binder .binder
  git grep -n -I -i -E 'binder|mybinder|repo2docker' -- . ':!docs/superpowers/plans/**'
  ```

  Expected: branch `remove-binder`; only the planning document may be new or modified; exactly the three listed Binder files are tracked; the reference search returns no matches (exit code 1). A search error is not a successful audit. If the base changes before execution and adds active launch links, include their targeted removal and update this plan before proceeding.

- [x] **Step 2: Delete the three configuration files.**

  ```bash
  git rm -- binder/postBuild binder/requirements.txt binder/runtime.txt
  ```

  Git removes the directory when its final file is deleted. If any unexpected local files remain, inspect them before taking further action.

- [x] **Step 3: Verify removal and check the change scope.**

  ```bash
  test ! -e binder
  test ! -e .binder
  git ls-files -- binder .binder
  git grep -n -I -i -E 'binder|mybinder|repo2docker' -- . ':!docs/superpowers/plans/**'
  git diff --check
  git diff --cached --check
  git diff HEAD --name-status
  git diff HEAD -- README.md docs/index.html docs/install.html docs/quickstart.html docs/script.js docs/style.css .github setup.py pyproject.toml MANIFEST.in requirements.txt svv svv_accel test
  ```

  Expected: both directory checks succeed; the tracked-file listing is empty; the reference search has no matches (exit code 1); both whitespace checks pass. The implementation diff contains only the three deletions, plus this planning record if staged. The final command produces no output, confirming the inspected application, packaging, documentation, and CI paths are unchanged.

  These static checks are sufficient for this configuration-only deletion. No new regression test, dependency installation, compiled build, or GUI smoke run is needed. If implementation expands into application behavior, reassess validation for that added scope.

- [x] **Step 4: Review and commit the focused change.**

  Review `git diff --cached` and `git status --short`. Update this plan's checklist and implementation status to reflect completed work, then commit the three deletions and the planning record:

  ```bash
  git add -- docs/superpowers/plans/2026-09-15-remove-binder.md
  git diff --cached --check
  git diff --cached --stat
  git -c user.name="Zachary Sexton" -c user.email="zsexton@stanford.edu" commit --author="Zachary Sexton <zsexton@stanford.edu>" -m "chore: remove unused Binder sandbox configuration"
  git status --short --branch
  ```

  Expected: one focused commit and a clean worktree. Pushing, opening a pull request, and merging can follow when requested.

## Acceptance criteria

- [x] `binder/` and `.binder/` are absent from the worktree and Git index.
- [x] No active Binder launch links or integration references remain outside this planning record.
- [x] Package code, root dependencies, installation guidance, and workflows have no implementation changes.
- [x] The change is isolated on `remove-binder` and the original checkout is untouched.

## Planning validation

Verified the clean branch baseline, inspected all three configuration files, searched tracked text and documentation for Binder integrations, and inspected package dependency loading and Pages deployment. No application code or dependencies are changed by this cleanup, so validation uses the static checks specified in Step 3.

## Implementation validation

- Confirmed the `remove-binder` branch starts from the recorded base commit.
- Confirmed both Binder directories are absent and the Git index lists no Binder files.
- Repeated the reference audit: no matches outside this planning record, with the expected search exit code of 1.
- Passed staged and unstaged whitespace checks; the change contains only the three configuration deletions and this record.
- Confirmed all application, dependency, packaging, documentation, and workflow files are unchanged.
- Compared the original checkout's branch, revision, status, and diff checksum before and after removal; all are unchanged.
- Independent review approved the cleanup with no findings.
- Commit author and committer: Zachary Sexton (`zsexton@stanford.edu`).
- Application tests were not run for this configuration-only deletion, as specified in Step 3.
