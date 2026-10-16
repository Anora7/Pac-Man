# Copilot Master Prompt: Pacman-RL Extensions

Paste everything inside the block below into Copilot Chat (Agent mode if available), with the project folder open.

````
You are extending my existing Pac-Man reinforcement learning project (Stable-Baselines3 PPO, Gymnasium, Pygame, PyTorch, Windows/PowerShell, Python 3.13, CPU only, no GPU). Start working immediately and carry out ALL of the tasks below end to end. Do not stop to ask questions between tasks; where something is ambiguous, make the most reasonable assumption, state it in one line in the docs, and keep going.

========================================
0. GROUND RULES (read first)
========================================
- First read RL_DESIGN.md, DECISIONS.md, ARCHITECTURE.md, environment.md, training.md, reward_function.md, api_reference.md, then config.py, maze.py, ghost.py, rewards.py, environment.py, train.py, evaluate.py, plots.py. Follow ARCHITECTURE.md's dependency rules (renderer never modifies state; environment never imports training code; ghosts never access rewards). If an extension needs a rule bent, add a new module instead of breaking the rule.
- DO NOT break the existing project. The 447 existing tests must still pass (python -m pytest tests\ -q). Run them after each major task and fix regressions before moving on.
- Keep the ORIGINAL setup as the default and as the baseline: rule-based ghosts (70% random / 30% chase, radius 5), the original 723-element observation, and the original reward values. Everything new must be opt-in through config flags or CLI arguments so the baseline stays reproducible. Existing models (models\best_model.zip) must keep loading and evaluating with the original observation.
- Put all new tunable values in config.py dataclasses (no magic numbers scattered in code), matching the project's existing style (dataclasses, keyword-only args, type hints, docstrings).
- Force device="cpu" for all SB3 models. Use fixed seeds everywhere for reproducibility.
- Training is slow (~90 steps/sec, 500k steps is about 1.5 hours). EVERY long-running script must have a --quick mode (tiny timesteps, e.g. 5k to 20k, 1 seed, 1 round) for smoke testing, and a --full mode with sensible defaults. Print progress and estimated time. Make runs resumable where practical (save checkpoints).
- Never overwrite models\best_model.zip or models\final_model.zip. Save all new artifacts under new folders: models\experiments\..., logs\experiments\..., results\..., plots\experiments\....
- Write tests for every new module (pytest, in tests\), keep them fast (no real training in tests; use a few steps), and run them.

========================================
1. IMPROVED OBSERVATION (opt-in)
========================================
Add a config flag (e.g. ObservationConfig.include_ghost_direction, default False) that appends ghost movement information to the observation:
- For each ghost: its last movement direction (one-hot of 4 directions, or zeros if it has not moved yet), plus a vulnerable/frightened flag if the original observation does not already encode it per ghost.
- Optionally add a "previous Pac-Man action" one-hot.
Update observation_space shape accordingly and make it depend on the flag. Document the new size (state the formula, like the original 5x144+3). Existing 723-d behaviour must be unchanged when the flag is off. Add tests for both shapes.

========================================
2. LEARNED GHOST AGENT (alternating training, SB3 only)
========================================
Goal: ghosts become learning agents too, using only Stable-Baselines3 via alternating training (train one side while the other is frozen, then swap, repeat). Do NOT use PettingZoo/RLlib.

2a. Ghost policy interface
- Create a clean policy abstraction for ghost control (e.g. src/ghost_policies.py) with interchangeable implementations behind one interface:
  * RandomGhostPolicy (pure random walk)
  * ChaseGhostPolicy (always chase using the existing Manhattan chase logic)
  * RuleBasedGhostPolicy (the existing 70/30 behaviour, default)
  * LearnedGhostPolicy (wraps a loaded SB3 PPO model and uses deterministic or stochastic predict)
- Optionally a "personality" variant per ghost (e.g. different chase_probability / chase_radius / ambush-ahead target) so ghosts behave differently, as in the original game.
- Keep wall-aware movement: a learned ghost's chosen action that hits a wall should be handled safely (stay in place or fall back to a valid move) and must not crash. Make the existing ghost movement code use the policy object while keeping default behaviour identical.

2b. Ghost-side training environment
- Create a Gymnasium environment (e.g. src/ghost_environment.py, class GhostEnvironment) in which the LEARNING agent is the ghost(s) and Pac-Man is controlled by a FROZEN loaded PPO model (path configurable; if none is given, fall back to a simple scripted Pac-Man such as greedy nearest-pellet).
- Support two modes: (a) one shared policy controlling all ghosts, where each ghost gets its own ghost-centred observation (own position, Pac-Man position, other ghosts, pellets remaining, power-mode state) and the shared model is queried per ghost; (b) single-ghost mode controlling just one ghost while others use a chosen fixed policy. Default to the shared-policy mode.
- Ghost reward (all values in a GhostRewardConfig in config.py): large positive for catching Pac-Man, small negative per step, a penalty for each pellet Pac-Man eats, a penalty when the ghost is eaten during power mode, a small shaping bonus for reducing distance to Pac-Man (clearly flagged as optional shaping and configurable to zero), and a penalty if Pac-Man clears the maze.
- Respect episode termination rules from the original environment (caught, cleared, step limit truncation). Use terminated vs truncated correctly.
- Reuse the existing maze/game logic rather than duplicating it. Refactor minimally and safely if needed.

2c. Alternating training loop
- Create src/train_alternating.py with a config dataclass (AlternatingConfig): n_rounds, pacman_steps_per_round, ghost_steps_per_round, seeds, initial pacman model path (default models\best_model.zip, only usable if observation settings match; otherwise train from scratch), paths for output.
- Round structure: Round 0 = Pac-Man vs rule-based ghosts (can load existing best model). Then for each round: (1) freeze Pac-Man, train ghost PPO; (2) freeze ghost, continue training Pac-Man PPO against the latest learned ghost.
- To avoid ghost overfitting to one Pac-Man snapshot, keep an "opponent pool" of past Pac-Man checkpoints and sample from it (e.g. 70% latest, 30% older) when training the ghost, and likewise for ghost checkpoints when training Pac-Man. Make pool sampling configurable.
- Save every round's models separately (models\experiments\alternating\round_XX_pacman.zip and round_XX_ghost.zip), and log per-round evaluation of both sides (Pac-Man mean reward, success rate, mean pellets, survival steps; ghost catch rate) to CSV (results\alternating_rounds.csv).
- Use reset_num_timesteps=False when continuing training so step counters stay continuous. Use EvalCallback/CheckpointCallback as the project already does.
- Provide --quick and --full modes.

========================================
3. CURRICULUM LEARNING (optional but implement it)
========================================
- Make the number of ghosts configurable (inspect config.py and ghost handling for how it is currently set; add a config field if it is hard-coded; default unchanged).
- Create src/train_curriculum.py: start with 1 ghost, and promote to 2 (then to the full number) when the rolling success/survival metric passes a configurable threshold or after a max number of steps. Log when each promotion happens. Output a plot of reward vs timesteps with vertical lines marking each promotion.
- Provide --quick and --full modes.

========================================
4. PPO vs DQN COMPARISON
========================================
- Create src/train_dqn.py using SB3 DQN on the same environment, same observation, same reward, same total timesteps, MlpPolicy, CPU. Choose reasonable DQN hyperparameters (replay buffer, learning starts, exploration schedule, target update interval) and put them in a config dataclass with comments explaining each one.
- Create src/compare_algorithms.py that trains/loads PPO and DQN across the same seeds, evaluates both on the same evaluation episodes, and writes results\algorithm_comparison.csv plus plots (learning curves with mean and standard deviation bands across seeds; bar chart of final mean reward and success rate with error bars).
- Add a short written note (in the report) tying the result back to Decision 001 in DECISIONS.md (why PPO over DQN).

========================================
5. REWARD ABLATION STUDY
========================================
- Create src/ablation.py that runs a configurable set of reward variants (each trained with the same timesteps and seeds), at minimum: baseline; caught penalty -200 vs -50 vs -500; step penalty -1 vs 0 vs -2; no wall penalty; pellet reward +10 vs +5 vs +20; clear-maze bonus +500 vs +0 vs +1000. Variants defined in one table/dict so they are easy to edit.
- Evaluate each trained variant over N episodes and write results\ablation_results.csv and plots (grouped bar chart of mean reward, success rate, and mean pellets per variant; plus a short text summary of which reward terms mattered most). Note in the report that reward totals are not directly comparable across different reward scales, so also report scale-independent metrics: pellets eaten, survival steps, success rate.

========================================
6. MULTI-SEED EXPERIMENTS
========================================
- Create a shared helper (e.g. src/experiment_utils.py) for running the same experiment across several seeds (default 3, configurable), aggregating results (mean, standard deviation, min/max, 95% confidence interval) and plotting mean curves with shaded bands. Reuse it in the DQN comparison, ablation, curriculum, and alternating experiments.
- Each seed's training must use Monitor logging so learning curves can be reconstructed from monitor CSVs.

========================================
7. GHOST-TYPE ROBUSTNESS EVALUATION
========================================
- Create src/evaluate_matrix.py: evaluate each available Pac-Man model (baseline, improved-observation, alternating-round models, curriculum model, if they exist) against each ghost type (random, chase-only, rule-based 70/30, learned ghost from the latest round, personality ghosts). Produce a matrix (rows = Pac-Man model, columns = ghost type) of mean reward, success rate, and mean pellets, saved to results\robustness_matrix.csv and rendered as a heatmap image with numbers annotated in each cell (matplotlib only).
- Skip missing models gracefully with a clear message (do not crash).
- Models trained with different observation settings must be evaluated with the matching observation config (store observation config alongside each saved model, e.g. a small JSON next to the .zip, and read it back when loading).

========================================
8. VISUALIZATIONS
========================================
Add src/analysis_plots.py (matplotlib only, saved to plots\experiments\), each as its own function with a PlotConfig-style config consistent with existing plots.py:
1. Pac-Man visitation heatmap: run N evaluation episodes, count visits per maze cell, overlay walls; produce one heatmap per ghost type if feasible. Also a ghost-position heatmap showing where ghosts tend to be, and a "where Pac-Man gets caught" heatmap (cell of death counts).
2. Value-function danger map: for a trained PPO model, take representative states (e.g. fixed pellet layout, Pac-Man at each walkable cell, ghosts at a few fixed positions), query model.policy.predict_values(...) for the state, and plot the value as a heatmap over Pac-Man's position. Low value = the agent thinks the situation is dangerous. Provide versions for: ghost nearby, ghost far, and power mode on. Add a plain-English explanation in the docs of how to read this.
3. Learning curves: reward, success rate, episode length with moving average (extend existing plots, do not break them).
4. Alternating-training dashboard: Pac-Man success rate and ghost catch rate across rounds on one chart, showing co-evolution.
5. Per-episode death/outcome breakdown (caught / cleared / timeout) as stacked bars per model.
6. Optional: an animated GIF or a short sequence of PNG frames of one evaluation episode (use pygame surface capture if the renderer supports it; skip gracefully if not).

========================================
9. EVALUATION & RENDER IMPROVEMENTS
========================================
- In evaluate.py add (without changing existing default behaviour): --ghost-type {rule,random,chase,learned}, --ghost-model PATH, --obs-config options, --seed, and --render-fps N (default about 8 so playback is watchable; make sure the Pygame loop uses pygame.time.Clock().tick(render_fps) and still handles window close events).
- Make the evaluation summary additionally report: mean survival steps, success/caught/timeout rates, mean pellets, standard deviation, and optionally a CSV export via --save-csv PATH.

========================================
10. ONE-COMMAND RUNNERS
========================================
- Create run_all_experiments.py (project root or src, with the sys.path handled correctly because modules live in src\) with flags: --quick (everything tiny, finishes in minutes, for verification), --full, and per-experiment toggles (--only alternating|curriculum|dqn|ablation|robustness|plots). It must print a clear summary of what ran, what was skipped, and where the outputs are.
- Make sure running it with --quick completes successfully end to end on this machine and produces every CSV and plot at least once. Run it yourself and fix any errors.

========================================
11. DOCUMENTATION
========================================
Create docs\extensions.md (and update README.md with a short section linking to it) containing:
- What was added and why, one section per extension (learned ghosts, improved observation, curriculum, PPO vs DQN, ablation, multi-seed, robustness matrix, visualizations), each with: the idea in plain English, how it works in the code (which file), exact PowerShell commands to run it (quick and full), and where outputs land.
- A "results interpretation" template with guidance on how to read each plot and table, including the value-function danger map and the ghost/Pac-Man co-evolution chart.
- An explanation of why alternating training is used instead of simultaneous training (non-stationarity; SB3 is single-agent) and the known limitations (ghost may overfit to one Pac-Man snapshot, mitigated by the opponent pool; learned ghosts make the task harder so Pac-Man success may drop at first).
- Update DECISIONS.md with new decision entries (alternating multi-agent training with SB3; opt-in observation extension; DQN comparison; reward ablation; multi-seed protocol) in the same format as existing entries.
- A "lab demo cheat sheet": a suggested 5-minute extension demo order and likely professor questions with short answers (why alternating, what is non-stationarity, what does the value map show, why multiple seeds, what did the ablation reveal).

========================================
12. DELIVERY ORDER & DEFINITION OF DONE
========================================
Work in this order and run the full existing test suite after each step: (1) ground rules + config changes, (2) improved observation, (3) ghost policy abstraction, (4) ghost environment, (5) alternating training, (6) evaluation upgrades + robustness matrix, (7) visualizations, (8) multi-seed helper, (9) DQN comparison, (10) ablation, (11) curriculum, (12) run_all_experiments.py, (13) docs.
Done means: all old and new tests pass; python run_all_experiments.py --quick completes and generates all outputs; the original commands (python src\train.py, python src\evaluate.py --model models\best_model.zip --episodes 20) behave exactly as before; and docs\extensions.md lists every command.
At the end, give me a concise summary: files added/changed, the exact commands to run quick and full modes, estimated runtime for full mode on CPU, and anything you were unsure about.

Begin now with step 1.
````
