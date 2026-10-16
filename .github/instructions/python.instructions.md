# Python Development Instructions

## Target Version

Python 3.12+

---

# Style

Follow:

- PEP 8
- PEP 257
- Type hints
- Meaningful naming

---

# File Structure

Preferred order:

1. Imports
2. Constants
3. Dataclasses
4. Classes
5. Functions
6. Main entry point

---

# Functions

- Small and focused
- Single responsibility
- Descriptive names
- Type hints
- Google-style docstrings

---

# Classes

Prefer:

- Dataclasses where appropriate
- Composition
- Clear ownership

Avoid:

- God classes
- Deep inheritance

---

# Constants

Avoid magic numbers.

Centralize:

- Rewards
- Grid sizes
- Hyperparameters
- Colors
- Timing values

---

# Exceptions

Raise meaningful exceptions.

Never silently ignore errors.

---

# Randomness

Support deterministic execution when requested.

Expose configurable seeds.

---

# Performance

Prioritize correctness before optimization.

Avoid premature optimization.

---

# Code Quality

Avoid:

- Duplicate code
- Global mutable state
- Long functions
- Unused variables
- Dead code

Prefer reusable helper functions.