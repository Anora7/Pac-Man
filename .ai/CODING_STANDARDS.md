# Coding Standards

## Purpose

This document defines the coding standards for the Pac-Man Reinforcement Learning project.

All source code should follow these guidelines to maintain readability, consistency, modularity, and maintainability.

---

# General Principles

The project should prioritize:

- Readability
- Simplicity
- Maintainability
- Modularity
- Reproducibility

Code should be written for humans first and computers second.

---

# Python Standards

Target Version

Python 3.12+

Follow:

- PEP 8
- PEP 257 (Docstrings)
- Type Hints

---

# File Organization

Each module should have one primary responsibility.

Avoid creating "utility dumping grounds."

Preferred structure:

Imports

Constants

Dataclasses

Classes

Functions

Main Entry Point (if applicable)

---

# Naming Conventions

## Classes

PascalCase

Example

Maze

Ghost

PacmanEnvironment

---

## Functions

snake_case

Example

move_pacman()

calculate_reward()

reset_environment()

---

## Variables

snake_case

Use descriptive names.

Avoid:

x

tmp

data

Prefer:

pellet_count

remaining_steps

ghost_positions

---

## Constants

UPPER_CASE

Example

MAX_STEPS

POWER_DURATION

PELLET_REWARD

---

# Type Hints

All public functions should include type hints.

Example

def reset(self) -> tuple:
    ...

Avoid untyped public APIs unless absolutely necessary.

---

# Documentation

Every public class should include a docstring.

Every public function should include:

Purpose

Arguments

Returns

Exceptions (if applicable)

Use Google-style docstrings consistently.

---

# Comments

Write comments explaining **why**, not **what**.

Avoid obvious comments.

Good:

Explain reward shaping rationale.

Avoid:

Increment x by one.

---

# Functions

Prefer small focused functions.

Ideal length:

10–40 lines

Large functions should be decomposed into helper methods.

---

# Classes

Prefer composition over inheritance.

Each class should own one responsibility.

Avoid "God classes."

---

# Configuration

Avoid hardcoded values.

Centralize configurable values such as:

- Rewards
- Hyperparameters
- Maze size
- Ghost probability
- Rendering settings

---

# Error Handling

Fail with meaningful exceptions.

Avoid silent failures.

Validate inputs where appropriate.

---

# Randomness

All randomness should support reproducibility.

Prefer configurable random seeds.

Avoid uncontrolled randomness during evaluation.

---

# Reinforcement Learning

Never encode an optimal Pac-Man strategy manually.

Learning must emerge from training.

Ghosts remain rule-based.

Reward shaping should be configurable.

---

# Testing

New functionality should be verified before continuing.

Prefer small, focused tests.

Do not modify unrelated modules while fixing bugs.

---

# Documentation Policy

Whenever code changes:

Update documentation if the change affects:

- Architecture
- RL design
- Public APIs
- Reward function
- Evaluation process

---

# Performance

Optimize only after correctness.

Avoid premature optimization.

The environment should remain lightweight enough for CPU training.

---

# Git Workflow

Commit only working code.

Use descriptive commit messages.

Recommended format:

feat:

fix:

docs:

refactor:

test:

chore:

---

# AI Development Guidelines

Before implementing a feature:

1. Read the AI documentation.
2. Produce a short implementation plan.
3. Identify affected files.
4. Make the smallest complete change.
5. Verify functionality.
6. Update documentation.
7. Stop and summarize.

Never rewrite unrelated code.

Never duplicate functionality.

Never introduce breaking architectural changes without updating documentation.