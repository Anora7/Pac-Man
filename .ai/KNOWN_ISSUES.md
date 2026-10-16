# Known Issues

## Purpose

This document tracks unresolved issues, bugs, technical debt, and potential improvements throughout development.

The goal is to avoid losing track of problems while allowing development to continue.

Resolved issues should either be removed or moved to the Resolution History section.

---

# Issue Template

---

## Issue ID

ISSUE-001

### Title

Short descriptive title

### Status

Open

Possible values:

- Open
- Investigating
- In Progress
- Resolved
- Deferred

### Priority

Critical

High

Medium

Low

### Category

Examples:

- Environment
- Rendering
- PPO
- Reward Function
- Ghost Logic
- Performance
- Documentation

### Description

Describe the problem clearly.

### Steps to Reproduce

1.
2.
3.

### Expected Behavior

Describe the expected result.

### Actual Behavior

Describe what currently happens.

### Possible Cause

Known or suspected root cause.

### Temporary Workaround

Optional.

### Assigned Phase

Indicate during which implementation phase this issue should ideally be resolved.

---

# Current Issues

At project initialization there are no known implementation issues.

Add issues here as they are discovered.

---

# Future Improvement Ideas

These are not bugs but possible enhancements.

Examples include:

- Random maze generation
- Curriculum learning
- Teleport tunnels
- Multiple ghost personalities
- Bonus fruit
- Dynamic difficulty
- PPO vs DQN comparison
- Gameplay recording
- Better evaluation statistics
- Additional visualization tools

---

# Resolution History

Move resolved issues here instead of deleting them.

---

## Example

### ISSUE-001

Ghost occasionally attempts to move into a wall repeatedly.

**Resolution**

Updated movement selection to retry valid neighboring cells before ending the turn.

Status:

Resolved

Date:

YYYY-MM-DD

---

# Technical Debt

Record implementation shortcuts that should be revisited.

Examples:

- Temporary constants that should move to configuration.
- Duplicate utility functions.
- Performance optimizations postponed until after functionality is complete.
- Placeholder rendering assets.
- Simplified ghost decision heuristics.

Technical debt should not prevent project completion but should be documented for transparency.

---

# Maintenance Notes

Review this document:

- Before each new implementation phase.
- Before major refactoring.
- Before the final demonstration.

Whenever an issue is resolved:

1. Update its status.
2. Document the resolution.
3. Update PROGRESS.md if the fix changes project status.
4. Record any architectural implications in DECISIONS.md if appropriate.