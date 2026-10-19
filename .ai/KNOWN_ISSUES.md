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

## ISSUE-002

### Title

Pac-Man May Oscillate After Nearby Pellets Are Exhausted

### Status

Investigating

### Priority

Medium

### Category

PPO

### Description

Observed gameplay can oscillate between nearby cells after local pellets are
consumed and does not clear the baseline maze. Opt-in nearest-pellet
observation and progress/revisit shaping variants have been evaluated. The
three saved `features_shaping` runs averaged 52.93 pellets and 11.67 clears
per 30 deterministic episodes; the three ghost-aware runs averaged 52.60
pellets and 11.33 clears. These results do not indicate an additional gain
from the nearest-ghost feature, and the issue remains open for the baseline.

### Expected Behavior

The variant evaluation should show reduced loop rate and ideally improved
pellet collection or maze clears without changing the baseline model or
defaults.

### Actual Behavior

The baseline reference rows in the three `features_shaping` CSVs averaged
44.43 pellets and zero clears. The selected 1M demo model cleared 13/30
against rule ghosts in the current robustness matrix, but only 0/30 against
chase ghosts; behavior is opponent-dependent.

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

### ISSUE-003

Learned ghosts repeatedly selected wall-directed actions, and evaluation
controlled all ghosts with PPO simultaneously despite shared training
controlling one ghost per step.

**Resolution**

Evaluation now rotates learned control across ghosts in the training schedule.
Wall-directed PPO actions use a random legal-neighbor fallback in both
training and evaluation. In three diagnostic episodes, the pre-fix models
selected a single direction continuously and had only 0–1.1% actual movement
per ghost. After correction, each ghost moved on 100% of its decisions. The
three final ghosts were re-evaluated for 30 episodes against the 1M demo
Pac-Man model; catch rates were 0.0%, 23.3%, and 66.7% for seeds 42, 43, and
44 respectively. Learned actions are chosen from the matching pre-step
observation.

Status:

Resolved

Date:

2026-10-19

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