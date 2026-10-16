# Testing Instructions

Every implementation should be validated before completion.

---

# Environment

Verify:

- reset()
- step()
- close()
- render()

Follow Gymnasium API.

---

# Movement

Check:

- Wall collisions
- Invalid moves
- Valid moves
- Grid boundaries

---

# Rewards

Verify:

- Pellet reward
- Power pellet reward
- Ghost reward
- Wall penalty
- Step penalty
- Completion reward

---

# Ghosts

Verify:

- Random movement
- Chase behavior
- Vulnerable mode
- Respawn behavior

---

# Episode Lifecycle

Confirm:

- Maze completion
- Ghost collision
- Step limit
- Time limit

---

# Rendering

Renderer should:

- Open successfully
- Draw all entities
- Close cleanly
- Not modify environment state

---

# PPO

Smoke test:

- Environment creation
- PPO initialization
- Short training run
- Model saving
- Model loading

---

# Evaluation

Verify:

- Metrics generated
- Models load
- Gameplay executes

---

# Regression Testing

Whenever possible:

Ensure previous functionality continues working after changes.

Avoid introducing regressions.

---

# Completion

Before marking a task complete:

- Verify implementation
- Verify documentation
- Verify imports
- Verify project still builds