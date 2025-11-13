# Contributing

## How to contribute?

This project is an open-source initiative actively seeking contributions from the community. This document outlines how to get involved and the process we follow for proposing, reviewing, and merging changes.

When contributing to this project, please first discuss the change you wish to make via a GitHub issue. For small fixes, opening a PR directly with clear context is fine; for larger changes or new features, please open an issue first to align on scope and approach.

Please note that all of your interactions in the project are subject to the [Code of Conduct](CODE_OF_CONDUCT.md). This includes creation of issues or pull requests, commenting on issues or pull requests, and extends to all interactions in any real-time space.

## Content Update Governance Process

This project uses a two-stage governance process for updates to the project’s rules, controls, and other structured content. This separates technical review from community governance, ensuring both content quality and project alignment.

### What Constitutes a Content Update

Content updates include changes to:
- Rules definitions, categories, and metadata
- Security control specifications and mappings
- Component or taxonomy elements and relationships
- Content-oriented documentation and guidance materials

### Two-Stage Process Overview

**Stage 1: Technical Review** – Content `feature` branches merge to the `develop` branch after standard PR review

**Stage 2: Community Review** – Bi-weekly governance review of the `develop` branch’s accumulated changes before release to `main`

```
feature-branch  →  develop    →    main
     ↑                 ↑             ↑
  Stage 1         Stage 2        Release
(Technical)       (Community)
```

### Non-Content Changes

The following types of changes are not covered by the two-stage content update process and continue to follow existing workflows:
- Bug fixes – technical corrections and error resolution
- Implementation changes – updates to code logic, algorithms, or system functionality
- Infrastructure updates – CI/CD, build processes, deployment configurations
- Documentation fixes – corrections to technical documentation, README updates, etc.
- Security patches – critical security-related fixes requiring immediate deployment
- Dependency updates – library upgrades and security patches for dependencies

These excluded change types may follow direct-to-`main` workflows as determined by repository maintainers and policies.

## Repository Structure

- `segment-1-introduction-and-foundations/` - Introduction and Foundations
- `segment-2-cloud-based-ai-labs/` - Cloud-Based AI Labs
- `segment-3-integrating-and-leveraging-ai-environments/` - Integrating and Leveraging AI Environments
- `segment-4-advanced-topics-and-practical-applications/` - Advanced Topics and Practical Applications

## First-time contributors

If you are new to the project and looking for an entry point, check the open issues. Issues tagged `good first issue` are meant to be small, well-scoped tasks suitable for first-time contributors. If you find one you’d like to work on, comment on the issue and/or assign yourself, and a maintainer can confirm assignment.

## Submitting a new issue

If you want to create a new issue that doesn't exist already, open a new one and include:
- a concise problem statement or feature request
- steps to reproduce (for bugs)
- expected vs. actual behavior
- environment details (versions, OS) if applicable
- proposed approach or alternatives (optional but helpful)

**If you discover a security bug, please do not report it through GitHub. Instead, please see security procedures in [SECURITY.md](SECURITY.md).**

## Submitting a new pull request and review process

The process for submitting pull requests depends on the type of change:

### For Content Updates (Two-Stage Process)

Follow these steps when submitting content updates:

1. Fork this repo into your GitHub account. If you have write access, you may create a branch directly.
2. Create a new branch, based on the `develop` branch, with a name that concisely describes what you’re working on.
3. Ensure that your changes pass validation and do not cause any existing tests to fail.
4. Submit a pull request against the `develop` branch.

#### Content Update PR Review

**Stage 1 Review**: Your PR to `develop` will be reviewed for technical criteria including content correctness, formatting and schema validation (where applicable), code hygiene, and commit message quality.

**Stage 2 Review**: On a regular cadence (typically bi-weekly), a PR will be created from `develop` to `main` containing all merged content changes from the cycle period. This undergoes community review by maintainers and contributors using lazy consensus and/or simple voting procedures as needed.

### For Non-Content Changes (Standard Process)

Follow these steps when submitting non-content changes (bug fixes, implementation changes, infrastructure updates, etc.):

1. Fork this repo into your GitHub account (or create a branch if you have write access).
2. Create a new branch, based on the `main` branch, with a name that concisely describes what you’re working on.
3. Ensure that your changes do not cause any existing tests to fail.
4. Submit a pull request against the `main` branch.

#### Non-Content PR Review
1. PRs will be reviewed by Omar Santos. Additional reviewers may be requested depending on scope.
2. Reviewer responses are typically due within 3 business days.


## Branch naming and commit messages

### Branch naming

- `main` – main development branch and authoritative source; updated after community approval for content changes
- `develop` – staging area for community review of content updates; feature branches for content changes target this branch
- `feature` – feature/this-is-a-new-feature-branch (target `develop` for content updates, `main` for non-content changes)
- `codebugfix` – codebugfix/description-of-the-bug (typically targets `main`)
- `languagefix` – languagefix/description-of-the-language-fix (typically targets `main`)
- `docs` – docs/description-of-the-documentation-change (typically targets `main`; documentation changes are exempt from the content update rule above)
- `release` – release/description-of-the-release – cut from `main` when ready

### Commit messages format

Write commit messages that clearly explain the change by continuing the sentence “This commit …”.

Examples of good commit messages:
- "This commit renames the examples folder to reference-implementations."
- "This commit bumps dependency versions to address security advisories."
