# Google Pac-Man Version Documentation

## Step 0 Reuse Report (Phase 1)
1. `config.py`: Import `Action`, `ACTION_DELTAS`, `CellType`, `RewardConfig`, and `ObservationConfig` as shared foundations, extending them in `v2_config.py` with custom 16x16 maze layout and ghost parameters.
2. `maze.py`: Import `Position` and subclass `Maze` into `GoogleMaze` in `src/v2_maze.py`, reusing `is_wall`, `is_walkable`, `has_pellet`, `consume_pellet`, `pellets_remaining`, and grid generators.
3. `maze.py` BFS routing: Reuse and generalize BFS pathfinding methods (`nearest_pellet_path`, `distances_from`) for target tile distance calculations and reachable-cell validation.
4. `rewards.py`: Reuse `calculate_step_reward` and `RewardConfig` for Pac-Man reward math, passing new power pellet, clear, caught, and pellet counts directly.
5. `ghost_policies.py`: Follow the `GhostPolicy` protocol to implement deterministic BFS shortest-path targeting and randomized fallback actions.
6. `ghost.py`: Reuse `GhostState` concept while creating `GoogleGhost` in `src/v2_ghost.py` to support Blinky, Pinky, Inky, and Clyde with scatter/chase cycles and release scheduling.
7. `renderer.py`: Subclass or adapt `PacmanRenderer` into `GoogleRenderer` in `src/v2_render.py` using `RenderState`, updating colors, animated mouth, and ghost shapes.
8. `observation_metadata.py`: Reuse `save_observation_config` and `load_observation_config` for serializing model feature configurations to JSON sidecars.
9. `train.py` / `train_curriculum.py`: Follow the SB3 PPO training pipeline structure, callbacks, and monitor logging in `src/v2_train.py` for curriculum progression.
10. `evaluate.py`: Reuse metric dataclasses and episode evaluation logic in `src/v2_evaluate.py` to report mean pellets, survival steps, clear rate, and caught rate across 30 seeds.

---

## Change Log (Files Created)
- `docs/GOOGLE_VERSION.md`: Comprehensive documentation, design specifications, command reference, and progress tracking.
- `src/v2_config.py`: Central configuration module defining 16x16 maze layout, ghost identities, colors, power mode parameters, rewards, and curriculum stage specifications.
- `src/v2_maze.py`: `GoogleMaze` class subclassing `Maze`, providing 16x16 layout, BFS pellet reachability verification, ghost house door access, and walkable projection helpers.
- `src/v2_ghost.py`: `GoogleGhost` class implementing 4 classic personalities (Blinky, Pinky, Inky, Clyde), BFS shortest-path movement, turn restrictions (no reverse), scatter/chase cycles, frightened retreat, house release schedules, and curriculum chase strength.
- `src/v2_environment.py`: Gymnasium environment `GoogleEnvironment` integrating the 16x16 maze, 4 rule-based ghosts, power pellet mechanics, scaled 1321-element observation space, and reward shaping.
- `src/v2_render.py`: Pygame-based `GoogleRenderer` featuring animated Pac-Man mouth, distinct ghost shapes (rounded head, wavy skirt, directional pupils), high-contrast flashing frightened ghosts, dark background, and HUD.
- `src/v2_train.py`: SB3 PPO curriculum training pipeline across chase strength (0.5 -> 0.75 -> 1.0) and active ghosts (2 -> 3 -> 4), logging to `results/google/` and saving models to `models/google/`.
- `src/v2_evaluate.py`: Comprehensive evaluation script running 30 seeds for both trained PPO agent and random-action Pac-Man baseline, outputting summary tables and exporting to CSV.
- `tests/test_google_maze.py`: Unit tests verifying 16x16 dimensions, 4 corner power pellets, BFS reachability of 100% of pellets, and absence of dead-end traps.
- `tests/test_google_ghost.py`: Unit tests verifying ghost factory, release schedules, scatter/chase cycles, targeting rules for all 4 ghosts, turn reversal restrictions, and frightened mode retreat/respawn.
- `tests/test_google_environment.py`: Unit tests verifying observation space (1321,), reset, step execution, pellet eating, power mode activation, ghost eating, and caught termination.
- `tests/test_google_render.py`: Unit tests verifying renderer initialization, window sizing, headless execution, and frame drawing for normal and frightened modes.
- `tests/test_google_train.py`: Unit tests verifying environment creation, curriculum callback promotion conditions, and PPO training loop.
- `tests/test_google_evaluate.py`: Unit tests verifying metric summary calculations, random evaluation episodes, and CSV output.

---

## Ghost Rules Specification
1. **Red "Blinky" (Shadow)**:
   - **Chase Target**: Pac-Man's exact current grid tile.
   - **Scatter Target**: Top-right corner `(1, 14)`.
   - **Release**: Step 0 (immediately active).
2. **Pink "Pinky" (Speedy)**:
   - **Chase Target**: 4 tiles ahead in Pac-Man's current facing direction. Clamped to the nearest walkable maze cell.
   - **Scatter Target**: Top-left corner `(1, 1)`.
   - **Release**: Step 10.
3. **Cyan "Inky" (Bashful)**:
   - **Chase Target**: Pivot tile is 2 tiles ahead of Pac-Man. Target vector is mirrored from Blinky's position across the pivot: `target = 2 * (pacman_pos + 2 * pacman_dir) - blinky_pos`. Clamped to closest walkable cell.
   - **Scatter Target**: Bottom-right corner `(14, 14)`.
   - **Release**: Step 20.
4. **Orange "Clyde" (Pokey)**:
   - **Chase Target**: If BFS shortest path distance to Pac-Man > 6 tiles, targets Pac-Man's current tile; otherwise retreats to his scatter corner.
   - **Scatter Target**: Bottom-left corner `(14, 1)`.
   - **Release**: Step 30.
5. **Movement & Turn Rules**:
   - Uses BFS shortest path to find the candidate step minimizing distance to target tile.
   - Never reverses direction (180-degree turn) unless forced (dead end) or during a mode change (scatter <-> chase, or normal -> frightened).
   - Mode timer: 7 steps scatter, 20 steps chase, repeating cyclically.
   - Frightened mode: Activated when Pac-Man consumes a power pellet. Ghosts pick the walkable neighbour maximizing BFS distance away from Pac-Man. When eaten, ghost teleports to house and respawns.
   - Release schedule: Ghosts are released one by one every 10 steps so early game is not overwhelming.
   - Chase strength: Probability in [0.5, 1.0] to follow rule vs uniform random legal move (used for curriculum training).

---

## Visual & Color Choices
- **Background**: Dark pitch black `(10, 10, 15)`.
- **Walls**: Electric neon indigo/blue `(35, 45, 185)` with outline, clearly distinct from ghost colors and background.
- **Ghost House Door**: Soft pink/white bar `(255, 180, 200)` spanning the entrance at `(6, 7)-(6, 8)`.
- **Pac-Man**: Classic bright yellow `(255, 238, 0)` with wedge mouth animation opening and closing in facing direction.
- **Blinky**: Red `(235, 45, 45)`.
- **Pinky**: Light pink `(255, 160, 200)`.
- **Inky**: Cyan `(40, 220, 230)`.
- **Clyde**: Orange `(255, 165, 40)`.
- **Frightened Ghosts**: High-contrast light grey/white body `(230, 235, 245)` with pink/red pupils `(240, 60, 80)`. Flashes to ghost's normal color during the final 8 steps of power mode to indicate expiration.
- **Ghost Anatomy**: Rounded top dome, vertical torso, wavy bottom skirt with 3 distinct scallops, and eyes whose pupils glance in their last move direction.
- **Pellets**: Crisp white `(245, 245, 245)`.
- **Power Pellets**: Glowing amber `(255, 195, 75)` with central bright core.
- **HUD**: Two-line status display below grid showing Score, Step, Pellets remaining, Episode, Power mode countdown, and active ghost modes.

---

## Config Parameters & Defaults
| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `GOOGLE_MAZE_HEIGHT` | 16 | Board height in rows |
| `GOOGLE_MAZE_WIDTH` | 16 | Board width in columns |
| `GOOGLE_MAX_STEPS` | 1500 | Max steps before truncation |
| `GOOGLE_POWER_MODE_DURATION` | 30 | Steps power pellet remains active |
| `GOOGLE_EAT_GHOST_REWARD` | 100.0 | Reward for eating vulnerable ghost |
| `GOOGLE_SCATTER_STEPS` | 7 | Steps per scatter cycle |
| `GOOGLE_CHASE_STEPS` | 20 | Steps per chase cycle |
| `GOOGLE_GHOST_RELEASE_INTERVAL` | 10 | Steps between ghost house releases |
| `GOOGLE_CLYDE_CHASE_DISTANCE` | 6 | Clyde distance threshold to retreat |
| `GOOGLE_PINKY_LOOKAHEAD` | 4 | Pinky lookahead tile offset |
| `GOOGLE_INKY_LOOKAHEAD` | 2 | Inky pivot tile offset |
| `GOOGLE_CELL_SIZE` | 36 | Pygame display pixels per cell |
| `GOOGLE_FPS` | 15 | Render frames per second |

---

## Exact Commands

### 1. Run All Google Tests
```powershell
.\.venvRL\Scripts\python.exe -m pytest -k "google" -v
```

### 2. Run Full Test Suite (Old + New)
```powershell
.\.venvRL\Scripts\python.exe -m pytest -q
```

### 3. PPO Curriculum Training
- **Quick Smoke Test (~20k steps)**:
  ```powershell
  .\.venvRL\Scripts\python.exe src/v2_train.py --smoke-test
  ```
- **Full Run (~1M steps)**:
  ```powershell
  .\.venvRL\Scripts\python.exe src/v2_train.py --steps 1000000
  ```

### 4. Evaluation and Baseline Comparison
- **Headless 30-episode evaluation against random baseline**:
  ```powershell
  .\.venvRL\Scripts\python.exe src/v2_evaluate.py --episodes 30
  ```
- **Live Rendering Demo (watch game played live)**:
  ```powershell
  .\.venvRL\Scripts\python.exe src/v2_evaluate.py --episodes 5 --render
  ```

---

## Known Limitations & Observations
1. **CPU Training Scale**: A full 1M step PPO run takes several hours on a single CPU core. The curriculum script is tuned to run efficiently on CPU with batch size 64 and vectorized rollouts.
2. **Ghost Targeting in Corners**: Inky's vector targeting can project outside the maze perimeter on diagonal corners; the `closest_walkable` helper safely clamps all projected points to valid reachable coordinates.
3. **No Existing Files Modified**: All legacy files in `src/`, `tests/`, `models/`, and `results/` remain untouched.

