# Pac-Man + Stable-Baselines3: Complete Study Guide

Everything the team needs to present and defend this project. Read it top to bottom once, then use the Q&A (Section 14) and the demo script (Section 13) to practise.

**How to read the confidence tags**
- No tag: confirmed from the project's results, logs, or code we looked at.
- **[VERIFY]**: we believe this is true but have not seen the exact code. Check it before you say it out loud.
- **[PENDING]**: a result we do not have yet (the 4-ghost retrain). Fill in the blanks in Section 10.

**A note on file names:** some code files in the project start with `google_`. Those are the files for the redesigned 16×16 version. We avoid that word on slides, but you will see it in the folder, in terminal output, and in the window title unless you renamed it. Do not be surprised by it.

---

## 1. The project in one minute

- **Assigned topic:** the Stable-Baselines3 library (SB3). We were asked to build a mini game with it. We chose Pac-Man.
- **What we built:** a Pac-Man game as a custom Gymnasium environment, and a Pac-Man agent trained with SB3's PPO algorithm.
- **Only Pac-Man learns.** The ghosts are rule-based (hand-written targeting rules), like the real game.
- **Two versions:**
  - **Original version:** 12×12 maze, 2 weak ghosts. Taught us what goes wrong.
  - **Redesigned version:** 16×16 maze, 4 ghosts with different personalities, curriculum training. This is the main result.
- **Headline result (3 ghosts, 30 evaluation episodes):** the trained agent eats **110.7 of 127 pellets** and survives **145.5 steps**, versus **10.5 pellets and 40.5 steps** for random actions. It clears the maze in **3.3%** of games (1 of 30).
- **Honest limitation:** the 4-ghost stage was not solved in our first run. [PENDING: update after the retrain, Section 10.]

If you remember only one sentence: *"We used Stable-Baselines3's PPO to train Pac-Man, using a curriculum that starts with easy ghosts and gets harder, and the trained agent eats about ten times more pellets than random."*

---

## 2. Reinforcement learning basics (the vocabulary)

You need these terms to talk about SB3.

| Term | Meaning | In our project |
|---|---|---|
| **Agent** | The thing that learns and acts | Pac-Man |
| **Environment** | The world the agent acts in | The maze, pellets, ghosts |
| **State / Observation** | What the agent sees at one moment | A vector of 1321 numbers |
| **Action** | What the agent can do | Up, Down, Left, Right (4 discrete actions) |
| **Reward** | A number telling the agent how good the last step was | +10 for a pellet, -200 for being caught, etc. |
| **Policy** | The agent's strategy: observation → action | A small neural network |
| **Episode** | One game from start to end | Ends on capture, maze cleared, or step limit |
| **Timestep** | One action-and-response step | We train for hundreds of thousands to 1M of them |
| **Return** | Total reward over an episode | Shown as `ep_rew_mean` in the log |
| **Discount factor (gamma)** | How much future reward is valued vs immediate reward | 0.99 |
| **Exploration** | Trying new things to discover rewards | Needed early in training |
| **Curriculum** | Training on easy versions first, then harder ones | Our 3 stages |
| **Reward shaping** | Extra small rewards that guide learning | Moving closer to a pellet: +0.5 |
| **Deterministic policy** | Always picks the most likely action | Used in evaluation |
| **Seed** | A number that fixes the randomness | 42 is our default |

**The learning loop (say this in your own words):**
1. The environment gives the agent an observation.
2. The agent picks an action.
3. The environment moves the game forward and returns a new observation and a reward.
4. The learning algorithm adjusts the policy so actions that led to more total reward become more likely.
5. Repeat for hundreds of thousands of steps.

---

## 3. Stable-Baselines3 (the assigned topic)

### 3.1 What it is
- A Python library of **reliable, well-tested implementations of reinforcement learning algorithms**, built on **PyTorch**.
- It lets you train an agent with a few lines of code instead of writing the algorithm yourself.
- It follows a **scikit-learn-style interface**: create a model, `learn()`, then `predict()`, `save()`, `load()`.
- It works with **Gymnasium** environments. Any game that follows the Gymnasium interface can be trained on.

### 3.2 What a Gymnasium environment must provide
- `reset()`: starts a new episode and returns the first observation.
- `step(action)`: applies an action and returns `(observation, reward, terminated, truncated, info)`.
- `observation_space` and `action_space`: describe the shapes and types.
  - `terminated` means the episode ended naturally (caught, cleared).
  - `truncated` means it was cut off (step limit).
- Our environment: 4 discrete actions, a flat 1321-value float observation.

### 3.3 Algorithms in SB3
| Algorithm | Type | Notes |
|---|---|---|
| **PPO** | On-policy, policy gradient | Used in our project. Stable, a good default. |
| **A2C** | On-policy | Simpler relative of PPO |
| **DQN** | Off-policy, value-based | We compared against it. Discrete actions only. |
| **SAC, TD3, DDPG** | Off-policy | Mainly for continuous actions. We did not use them. |
| **MaskablePPO** | In the companion package `sb3-contrib` | Lets you mask illegal actions. We did not use it. |

Say "SB3 includes these, we used PPO and DQN." Do not claim results for the others.

### 3.4 Policies
- **MlpPolicy:** a fully connected network for vector observations. **We used this.**
- **CnnPolicy:** for image observations. We did not use it.
- **Why MlpPolicy:** our observation is already a flat vector, and the maze is small, so a convolutional network would add complexity without benefit.

### 3.5 Helpers SB3 provides
- **Vectorized environments** (we saw `DummyVecEnv` in the training log: "Wrapping the env in a DummyVecEnv"). They let SB3 handle one or many copies of the environment in a uniform way.
- **Monitor wrapper:** records episode reward and length. **[VERIFY]** that our training script wraps the environment with Monitor (the original plan said it would, and the stage summaries report episode statistics).
- **Callbacks:** hooks that run during training. We use one to run the curriculum (check performance, move to the next stage). **[VERIFY]** in `google_train.py`; the tests mention "promotion callbacks".
- **Logging:** the table printed during training, plus optional TensorBoard.
- **save() / load():** we save checkpoints and carry the same model from stage to stage.

### 3.6 The core code pattern (show this on a slide)
```python
env = PacmanEnvironment()                      # our Gymnasium environment
model = PPO("MlpPolicy", env)                  # SB3 creates the agent
model.learn(total_timesteps=1_000_000)         # training
model.save("pacman_ppo")                       # save
action, _ = model.predict(obs, deterministic=True)   # use it
```

### 3.7 What SB3 did vs what WE did (your professor will ask)
| SB3 gave us | We had to build ourselves |
|---|---|
| The PPO and DQN algorithms | The Pac-Man game (maze, pellets, power pellets) |
| The neural network (MlpPolicy) | The Gymnasium environment wrapper |
| Training loop, optimisation, logging | The observation design (1321 values) |
| save / load / predict | The reward function and shaping |
| Vectorized env and Monitor helpers | The 4 ghost personalities (rules) |
| Callback hooks | The curriculum and promotion logic |
| | The evaluation scripts and the random baseline |
| | The renderer (pygame) |

**Why we chose SB3:** it lets us focus on designing the game, rewards and training plan instead of writing RL algorithms from scratch. We also used it to compare two algorithms in a few lines.

---

## 4. PPO explained in plain language

**PPO = Proximal Policy Optimization** (Schulman et al., 2017).

- It is a **policy-gradient** method: it directly improves the policy (the network that picks actions), rather than first learning a table of values.
- It is **on-policy**: it learns from data gathered by the current policy, then throws that data away and collects fresh data.
- **The key idea, "proximal":** after each batch of experience, update the policy, but **clip** the update so the new policy cannot move too far from the old one. Big jumps can wreck learning. Clipping keeps updates small and safe.
- **Why it suits us:** it is stable, needs relatively little tuning, and works well with discrete action spaces like our 4 moves.
- **Actor-critic:** the network has two outputs. The **actor** chooses actions. The **critic** estimates how good a state is (the value), which helps judge whether an action was better than expected.

### PPO settings we used (SB3 defaults)
| Setting | Value | Meaning |
|---|---|---|
| Learning rate | 3e-4 | Size of each learning step |
| Gamma | 0.99 | How much the future matters |
| n_steps | 2048 | Steps collected before each update |
| Batch size | 64 | Mini-batch size for updates |
| n_epochs | 10 | Passes over each collected batch |
| Clip range | 0.2 | The "proximal" limit on policy change |
| Entropy coefficient | 0.01 | Small bonus for exploring (randomness) |

One "iteration" in the log = 2048 steps collected, then 10 epochs of updates. That is why the log's `total_timesteps` goes up in jumps of 2048.

### Why PPO beat DQN here
- DQN stores past experience in a replay buffer and learns values. In our matched test at 300k steps with 3 seeds, DQN averaged 19.7 pellets vs PPO 49.5.
- **Careful:** do not claim "PPO is better than DQN in general." Say "in our environment and settings, PPO did better, so we used PPO." We did not tune DQN heavily.

---

## 5. The game: the redesigned version

### 5.1 The maze
- **16×16 grid.** 127 pellets total: **123 regular + 4 power pellets** (one near each corner).
- A central **ghost house** with a door. Pac-Man starts at the bottom middle.
- Every pellet is reachable (checked by a test using BFS, which is shortest-path search).
- Maximum episode length: 1500 steps.

### 5.2 Actions
Up, Down, Left, Right. Walking into a wall wastes the move and costs a penalty.

### 5.3 Observation (what Pac-Man sees): 1321 numbers
All values are small floats. The vector contains:
- **Five 16×16 grid channels** (walls, regular pellets, power pellets, Pac-Man, ghosts). 5 × 256 = 1280 values.
- **Status values** (for example power mode on/off, power timer, fraction of maze completed).
- **Extra helper features:** ghost movement directions and vulnerability, nearest-pellet direction and distance, the previous action, and an "escape" feature about the nearest dangerous ghost.

You do not need to memorise the breakdown. Say: *"A flat vector of 1321 numbers: grids for walls, pellets, Pac-Man and ghosts, plus a few helper features like where the nearest pellet is."* **[VERIFY]** the exact split in `google_environment.py` if asked.

**Why a flat vector?** SB3's `MlpPolicy` takes vectors directly. No image processing needed.

### 5.4 Rewards
| Event | Reward |
|---|---|
| Eat a regular pellet | **+10** |
| Eat a power pellet | **+50** |
| Eat a frightened ghost | **+100** |
| Clear the maze | **+500** |
| Caught by a ghost | **-200** |
| Bump into a wall | **-5** |
| Every step | **-1** |
| Move closer to the nearest pellet (shaping) | +0.5 |
| Move away from the nearest pellet (shaping) | -0.5 |
| Revisit a recently visited cell (shaping) | -0.1 |

**Insights to say out loud:**
- The **-1 step penalty forces urgency**. An agent that stands still always loses score.
- The **-200 catch penalty** teaches danger. The **+500 clear bonus** is the big goal, but it is rarely reached, so the pellet rewards and shaping carry most of the learning.
- **Shaping** helps because the maze clear is a rare, distant reward. Small guiding rewards give the agent something to learn from early.

### 5.5 Power pellets and frightened mode
- Eating a power pellet starts **power mode for 30 steps**.
- Ghosts turn **light blue**, flee from Pac-Man, and can be eaten (+100). They **flash white** near the end of power mode. An eaten ghost respawns at the ghost house.

### 5.6 End of an episode
- **Caught by a ghost**, **maze cleared**, or the **step limit** (1500).
- **Why the "caught" rate is 100% even for a good agent:** the game only ends on capture or a clear. A trained agent usually dies before finishing all 127 pellets. So 100% caught does not mean it is bad. Look at pellets, survival steps and clears instead.

---

## 6. The ghosts (rule-based, never trained)

**Principle:** ghosts are rule-based. Pac-Man learns; ghosts follow rules. This mirrors the real game and contrasts "rule-based AI" with "learning AI".

### 6.1 Movement basics
- Each ghost uses **BFS (breadth-first search) shortest path** toward its **target tile**. It steps to the neighbour closest to the target.
- Ghosts **do not reverse direction** unless forced (dead end or mode change).
- If two moves are equally good, the ghost picks **randomly** between them.
- Each ghost has a **chase strength**: the probability it follows its rule instead of making a random legal move. This is the curriculum's difficulty dial.

### 6.2 The four personalities
| Ghost | Colour | Target in chase mode |
|---|---|---|
| **Blinky** | Red | Pac-Man's current tile (direct chaser) |
| **Pinky** | Pink | 4 tiles ahead of Pac-Man's current direction (ambusher) |
| **Inky** | Cyan | **Flanker.** When far from Pac-Man (more than 8 steps), targets the point on the opposite side of the board to cut him off. When close, chases directly. |
| **Clyde** | Orange | Chases when far (more than 6 steps away), retreats to his corner when close |

**Note on Inky:** in the original arcade game Inky mirrors Blinky's position. We changed it to the flanker version above because the original made Inky and Pinky bunch up on our small board. If asked, say: *"We adapted Inky to flank rather than follow, to stop the ghosts moving as a single pack."* (Inky's change was made late; the first full retrain after it had a weak 4-ghost stage. See Section 10.)

### 6.3 Modes
- **Scatter:** each ghost heads to its own corner (Blinky top-right, Pinky top-left, Inky bottom-right, Clyde bottom-left).
- **Chase:** ghosts go after Pac-Man using their personality targets.
- **Cycle:** scatter for about 10 steps, then chase for 20, repeating. Each ghost's timer is **offset** a little so they do not all switch at the same moment.
- **Frightened:** after a power pellet, ghosts move to maximise their distance from Pac-Man.
- **In house:** ghosts wait in the house and are **released one at a time** at steps **0, 30, 60, 90**, so the start is not overwhelming.

### 6.4 Per-ghost chase strength (adds variety)
Each ghost's effective strength = the stage's chase strength × a personality factor:
Blinky 1.0, Pinky 0.95, Inky 0.90, Clyde 0.85. So Blinky is the most relentless, Clyde the least. This is part of why the ghosts do not behave identically.

---

## 7. Curriculum training (the key idea)

### 7.1 The problem
Against 4 full-strength ghosts, a brand-new agent dies in about **20 steps** and never gets a chance to learn anything. A random agent survives about 23 steps at that setting.

### 7.2 The solution: train in stages
We use one PPO model that keeps training through all stages (`save`/`load` carry it over):

| Stage | Active ghosts | Chase strength | Share of steps |
|---|---|---|---|
| 1 | 2 | 0.5 | 50% |
| 2 | 3 | 0.75 | 33% |
| 3 | 4 | 0.85 [PENDING: was 1.0 in the first run] | 17% |

- **Total budget:** 1,000,000 steps, about 35 minutes on a laptop CPU (roughly 500 steps per second).
- Stage 1 gets the most steps because weak basics ruin everything after.
- A stage ends when it hits its step budget, **or** earlier if performance crosses a promotion threshold (70% success over a rolling window of 20 episodes). In our runs, stages always ended on the step budget (`promoted_reason = max_steps`), because the success rate never reached 70%. **[VERIFY]** in `google_train.py`.

### 7.3 Training metrics (in the log)
| Log field | What it means | What good looks like |
|---|---|---|
| `ep_len_mean` | Average episode length | Rising (survives longer) |
| `ep_rew_mean` | Average episode reward | Rising. Ours went from about -200 to +700 and above. |
| `success_rate` | Fraction of episodes clearing the maze | Small but above 0 |
| `fps` | Steps per second | About 400-700 on our laptop |
| `entropy_loss` | How random the policy is (negative of entropy) | Moving toward 0 means more decisive. Ours went from -1.38 to about -0.25. |
| `explained_variance` | How well the critic predicts returns (1 is perfect) | Ours went from about 0 to 0.95+ |
| `value_loss` | Critic error | Falling |
| `approx_kl`, `clip_fraction` | How much each update changed the policy | Small values mean stable updates |

**A good sentence:** *"Early in training, reward is negative, close to random. As the agent learns to collect pellets and survive, reward climbs, and the policy becomes less random (entropy falls)."*

---

## 8. Results

### 8.1 Main result: trained agent vs random (3 ghosts, chase 0.75, 30 episodes)
| Metric | Random | Trained PPO |
|---|---|---|
| Pellets (of 127) | 10.5 ± 4.3 | **110.7 ± 11.0** |
| Survival steps | 40.5 ± 21.2 | **145.5 ± 22.1** |
| Clear rate | 0% | **3.3% (1 of 30)** |
| Score | -190.0 | **+1015.7** |

- The agent eats about **10× more pellets** and survives about **3.6× longer**.
- It eats about **87%** of the maze on average.
- The "±" is the spread across the 30 episodes. It is not zero, so the episodes genuinely differ.
- This is the **stage 2 model** (3 ghosts), saved as the demo checkpoint.

### 8.2 Stage results from the first 1M-step run (training episodes)
| Stage | Setup | Mean pellets | Survival | Clears |
|---|---|---|---|---|
| 1 | 2 ghosts, 0.5 | 70.3 | 120 | 2.8% |
| 2 | 3 ghosts, 0.75 | 108.0 | 145 | 2.6% |
| 3 | 4 ghosts, 1.0 | **12.6** | 59 | 0% (collapsed) |

### 8.3 Earlier 300k run (before the final ghost changes)
- 4 ghosts, full strength, previous ghost design: about **86 pellets**, 105 survival steps, 0 clears. Then a later evaluation gave about 97 pellets and 109 steps (3 episodes only).
- This proves 4 ghosts at full strength was learnable with the earlier ghost design.

### 8.4 Original version results (12×12, 2 weak ghosts)
| Experiment | Result |
|---|---|
| **PPO vs DQN, 300k steps, 3 seeds (42, 43, 44)** | PPO 49.5 pellets / 74.1 steps. DQN 19.7 pellets / 43.0 steps. Neither cleared the maze. |
| **Medium comparison, 100k steps, 2 seeds** | PPO 15.5 pellets, DQN 10.1. Inconclusive (too few steps). |
| **Endgame variants (extra features + shaping), 300k, 3 seeds** | Best variant: about 53 pellets and about 11.7 clears per 30 games. |
| **1M-step demo model** | Cleared 13/30 vs default ghosts, 23/30 vs random ghosts, **0/30 vs chase ghosts**. |
| **Ghost-direction observation** | Worse than baseline (32 vs 45 pellets). |
| **Curriculum (original version, medium)** | Did not reach the promotion threshold. |
| **Reward ablations, medium** | Caught penalty -50 gave 26 pellets vs 15.5 baseline, but 0% clears. A lead, not proof. |

The "0/30 against chase ghosts" number is a great talking point: the original model did well against weak ghosts and failed against a real chaser. That is why we redesigned the ghosts.

---

## 9. The project story (what we tried and what we learned)

Tell this as a story. It is the most convincing way to present.

1. **Original version.** Trained PPO Pac-Man on a 12×12 maze against 2 ghosts that chase only 30% of the time and otherwise move randomly. Pac-Man followed one fixed route.
2. **PPO vs DQN.** Both are SB3 algorithms. PPO won clearly, so we dropped DQN.
3. **Learned ghosts.** We tried training ghosts with PPO too (alternating training: freeze one, train the other). They collapsed to repeating one direction, often into a wall.
4. **Bug 1: evaluation mismatch.** While diagnosing, we found the learned ghost was *tested* differently than it was *trained*: tested with every ghost moving each step and acting after Pac-Man moved, but trained with one ghost per step acting before Pac-Man moved. We fixed it and added a legal-move fallback. After the fix, learned-ghost catch rates over 30 games were **0%, 23%, 67%** for three seeds, versus **57%** for the rule-based ghost. So learned ghosts were inconsistent and not better than rules.
   - *Caution:* an older results file showed 16.7% / 83.3% / 90% against a different Pac-Man model, from before the fix. Do not quote those.
5. **Decision.** The ghosts were the weak link. The real game's ghosts are rule-based, so we redesigned with proper rule-based ghosts and kept Pac-Man as the only learner.
6. **Redesigned version.** 16×16 maze, 4 power pellets, 4 ghost personalities, scatter/chase, house release, frightened mode, new visuals.
7. **First training attempts.** The very first test model barely moved (a 3,000-step smoke test), which looked like failure but was just an untrained model.
8. **Speed-up.** After caching shortest-path distances, training went from about 90 to about 500 steps per second.
9. **Bug 2: identical evaluations.** All 30 evaluation episodes gave identical numbers (spread 0.0) because nothing was random. We added randomness (random tie-breaking, per-ghost chase strengths) so episodes differ.
10. **Ghost spacing.** Ghosts moved as a pack, so we added per-ghost timer offsets, different chase strengths, and changed Inky into a flanker.
11. **Stage 3 collapse.** After the Inky and timing changes, the 4-ghost stage collapsed to 12.6 pellets. The stage 2 model stayed strong (108 pellets). We saved it as the demo model.
12. **Fix in progress.** A gentler stage 3 (chase strength 0.85) and a retrain. [PENDING]

**Lesson to state:** checking evaluation carefully caught two real bugs. Reward alone did not give us good ghosts. Curriculum was essential for the agent to learn at all.

---

## 10. The 4-ghost stage [PENDING: fill this in]

**What happened in the first run:** stage 3 (4 ghosts, chase 1.0) averaged 12.6 pellets and 59 survival steps with 0 clears.

**Our explanation (a hypothesis, not proven):**
- The jump from 3 ghosts at 0.75 to 4 ghosts at 1.0 in one step was too big.
- We had also just made the ghosts harder to predict (Inky's flanking target).
- Supporting evidence: the same 4-ghost, full-strength stage reached about 86 pellets with the earlier ghost design, so the setting is learnable.

**The fix:** stage 3 chase strength lowered to 0.85, then retrain with 1M steps.

**Retrain result (fill in):**
- Stage 3 mean pellets: ______
- Stage 3 survival steps: ______
- Stage 3 clear rate: ______
- 30-episode evaluation at 4 ghosts, chase 0.85: pellets ______ / steps ______ / clears ______
- Status: ☐ worked (above about 80 pellets) ☐ still collapsed ☐ not run

**What to say if it did not work:** *"The 3-ghost model is our main result. The 4-ghost stage needs a finer curriculum, for example an extra stage between 3 and 4 ghosts, and that is our next step."*

---

## 11. Important honesty points (say these before someone asks)

- **Single training seed.** The redesigned version was trained with seed 42 only. Our results are **preliminary**. The original version used 3 seeds for PPO vs DQN.
- **Evaluation used 30 episodes with different seeds**, but they are all from one trained model.
- **The agent is evaluated deterministically** (always its most likely action), while the game has randomness from the ghosts.
- **100% caught is normal here** (Section 5.6).
- **Clears are rare** (3.3%). The strong claim is pellets and survival, not clearing.
- **The training curves come from training episodes** (the agent still exploring), while the evaluation table is from separate deterministic evaluation, so the two do not match exactly.
- **The ghosts are only partly "smart".** They use shortest-path rules plus some randomness. Do not claim they learn.
- **Do not claim** the agent "hunts" frightened ghosts, or that it generalises beyond the tested settings, unless you have seen it in the render.
- **Do not quote a test count.** Our documents disagree with each other.

---

## 12. Code map (what each file does)

**Redesigned version (main)**
| File | Purpose |
|---|---|
| `src/google_config.py` | Constants: maze layout, rewards, colours, ghost settings, curriculum stages |
| `src/google_maze.py` | Maze class: walls, pellets, power pellets, BFS distances, reachability check |
| `src/google_ghost.py` | The four ghosts: targeting, scatter/chase, frightened, release schedule |
| `src/google_environment.py` | The Gymnasium environment (observation, step, rewards) |
| `src/google_render.py` | Pygame renderer (speed keys: Up/Down to change speed, Space to pause) |
| `src/google_train.py` | SB3 PPO curriculum training, saves checkpoints and CSV logs |
| `src/google_evaluate.py` | 30-episode evaluation vs random baseline, `--render` option |
| `tests/test_google_*.py` | Unit tests for each part |
| `docs/GOOGLE_VERSION.md` | Design document (some values are stale: trust the code) |

**Original version**
| File | Purpose |
|---|---|
| `src/environment.py`, `maze.py`, `ghost.py`, `rewards.py` | The 12×12 game |
| `src/train.py`, `evaluate.py`, `renderer.py`, `plots.py` | Training, evaluation, visuals |
| `src/train_dqn.py`, `compare_algorithms.py` | The DQN comparison |
| `src/train_alternating.py`, `ghost_policies.py` | Learned-ghost experiments |
| `src/train_endgame_variants.py`, `train_curriculum.py` | Other experiments |

**Where results are saved**
- `results/google/curriculum_summary.csv`: one row per stage.
- `results/google/curriculum_episodes.csv`: one row per training episode.
- `results/google/evaluation_episodes.csv`: overwritten at every evaluation. Copy it if you want to keep it.
- `models/google/`: stage checkpoints and the final model. `models/google_demo_stage2.zip` is the protected demo model.

---

## 13. Demo guide

### Commands (run from the project folder in PowerShell)
**Main demo (3 ghosts, the strong model):**
```
.\.venvRL\Scripts\python.exe src\google_evaluate.py --model-path models\google_demo_stage2.zip --episodes 3 --render --ghosts 3 --chase-strength 0.75
```
**30-episode table:**
```
.\.venvRL\Scripts\python.exe src\google_evaluate.py --model-path models\google_demo_stage2.zip --episodes 30 --ghosts 3 --chase-strength 0.75
```
**Training (about 35 minutes):**
```
.\.venvRL\Scripts\python.exe src/google_train.py --steps 1000000
```
**Tests:**
```
.\.venvRL\Scripts\python.exe -m pytest -k "google" -q
```
**Original version demo:**
```
.\.venvRL\Scripts\python.exe src\evaluate.py --model models\best_model.zip --episodes 3 --render
```

### Render controls
- **Up / Down arrow** (or + / -): change speed.
- **Space:** pause.

### Before the demo
1. Plug in the laptop and turn off sleep.
2. Do one full dry run of the render command.
3. Copy `results\google\evaluation_episodes.csv` somewhere safe (each evaluation overwrites it).
4. Have the 30-episode table open on a slide in case the live render fails.
5. Do **not** run training right before the demo.

### What to say while it plays
1. "This is Pac-Man, the yellow one. It is controlled by a PPO policy trained with Stable-Baselines3."
2. "Each coloured ghost has its own targeting rule: red chases directly, pink ambushes ahead, cyan flanks, orange backs off when close."
3. Point out: Pac-Man moves toward pellets on purpose, it changes route when ghosts approach, and ghosts leave the house one at a time.
4. When a power pellet is eaten: "The ghosts turn light blue and flee, and flash white when it is about to end." Only describe what you actually see Pac-Man do next.
5. When it gets caught: "It usually gets caught eventually. What matters is that it eats about 110 pellets first, versus about 10 for a random agent."

### If something goes wrong
| Problem | Fix |
|---|---|
| Colours or speed look wrong | Check `google_render.py` was not partly reverted |
| `AttributeError: _mouth_phase` | The `self._mouth_phase: int = 0` line is missing from `GoogleRenderer.__init__` |
| Agent moves right and dies at step 20 | Wrong model file (an early or collapsed one). Use `models\google_demo_stage2.zip` |
| 4 pellets, 19 steps | Same cause: the stage 3 model from the collapsed run |
| A ghost test fails | Tests are pinned to current values (scatter 10 steps, Inky flanker). Update the expected values |
| Window says "Google Pac-Man RL" | Change the caption in `google_render.py` to "Pac-Man RL" |
| PowerShell error pasting code | The code line belongs in the file, not the terminal |

---

## 14. Likely questions and good answers

**General**
1. **What is Stable-Baselines3?** A library of reliable implementations of RL algorithms in PyTorch, with a simple `learn/predict/save/load` interface and Gymnasium compatibility.
2. **What did SB3 do for you, and what did you build?** SB3 provided PPO, the network, the training loop, logging and save/load. We built the game, environment, observation, rewards, ghost rules, curriculum and evaluation.
3. **Why PPO?** It is stable, needs little tuning, handles discrete actions, and beat DQN in our comparison (49.5 vs 19.7 pellets, 3 seeds, 300k steps).
4. **Why not DQN?** We tested it. It performed worse in our environment at the same budget. We did not heavily tune it, so we do not claim DQN is bad in general.
5. **Why MlpPolicy and not CnnPolicy?** Our observation is a flat vector and the maze is small. A CNN adds complexity without clear benefit.
6. **Did you use a GPU?** No. The network is small, and the slow part is simulating the game, not the network. CPU only, about 500 steps per second.

**About training**
7. **What is a curriculum and why use it?** Training on easy versions first. Against 4 full-strength ghosts a new agent dies in about 20 steps and cannot learn.
8. **How did you choose the stages?** By gradually raising the number of ghosts and their chase strength (2 → 3 → 4 ghosts; 0.5 → 0.75 → 0.85 or 1.0).
9. **What is chase strength?** The probability a ghost follows its rule rather than moving randomly.
10. **Why did stage 3 fail?** Our hypothesis: too big a jump in difficulty combined with a recent change to the ghost behaviour. We tested a gentler stage. [Fill in the outcome.]
11. **How long did training take?** About 35 minutes for 1M steps on a laptop CPU.
12. **How many seeds?** The redesigned version used seed 42 only, so results are preliminary.

**About results**
13. **Does it clear the maze?** Rarely (3.3% at 3 ghosts). Its strength is eating about 87% of the pellets and surviving 3.6× longer than random.
14. **Why is the caught rate 100%?** The game only ends on capture or a clear, and the agent usually dies before clearing all 127 pellets.
15. **How do you know it learned and did not just get lucky?** The random baseline eats 10.5 pellets and the trained agent 110.7, over 30 different seeds. The learning curve rises through training, and the critic's explained variance rose from about 0 to above 0.95.
16. **Why does the agent follow a similar route each time?** Evaluation is deterministic, and the maze is fixed. Ghost randomness makes the games differ, but the agent's strategy is consistent.
17. **Is the agent dodging ghosts or just eating pellets?** Its survival time is 3.6× longer than random and it escapes capture for about 145 steps against chasing ghosts, so it avoids danger at least partly. Do not overclaim anything about its tactics unless you saw them in the render.

**About the ghosts**
18. **Are the ghosts learning?** No. They are rule-based, like the real game. We tried learned ghosts first and they collapsed to one repeated action, so we switched.
19. **How do the ghosts find Pac-Man?** BFS shortest path to a target tile. The target depends on the personality and the mode.
20. **Why do they behave differently?** Different targets, different chase strengths, different release times and timer offsets.
21. **How are frightened ghosts handled?** They turn light blue, move to maximise distance from Pac-Man, and can be eaten for +100.

**About the problems**
22. **What went wrong?** Learned ghosts collapsed. We found an evaluation mismatch and fixed it. Evaluations were identical until we added randomness. Stage 3 collapsed after we changed the ghosts.
23. **What would you do with more time?** A finer curriculum (for example 3.5 ghosts, or smaller chase steps), longer training, several seeds with confidence intervals, and a larger maze.
24. **How did you validate the code?** Unit tests for the maze, ghosts, environment, renderer, training and evaluation. Do not state a number.

---

## 15. Glossary (quick reference)

- **BFS:** breadth-first search, finds the shortest path in a grid.
- **Callback:** code SB3 runs during training (we use it to promote curriculum stages).
- **Clip range:** PPO's limit on how much the policy may change per update.
- **Critic / Actor:** the value estimator and the action chooser inside PPO's network.
- **Curriculum learning:** training from easy to hard.
- **Entropy:** a measure of how random the policy is.
- **Explained variance:** how well the critic predicts returns (1 is perfect).
- **Gymnasium:** the standard interface for RL environments.
- **MlpPolicy:** a fully connected network policy for vector observations.
- **On-policy:** learns only from data collected by the current policy.
- **Replay buffer:** DQN's memory of past experience.
- **Rollout:** a batch of steps collected before one PPO update.
- **Scatter / Chase / Frightened:** the ghost modes.
- **Seed:** fixes randomness for reproducibility.
- **Smoke test:** a tiny run just to check the pipeline works.
- **Truncated vs terminated:** cut off by a step limit vs ended naturally.
- **Vectorized environment:** SB3's wrapper for running one or more environment copies.

---

## 16. One-page cheat sheet

- **Topic:** Stable-Baselines3. **Game:** Pac-Man. **Algorithm:** PPO, MlpPolicy, CPU.
- **Learner:** Pac-Man only. **Ghosts:** rule-based (red direct, pink ahead, cyan flank, orange shy).
- **Maze:** 16×16, 127 pellets, 4 power pellets. **Observation:** 1321 numbers. **Actions:** 4.
- **Rewards:** pellet +10, power +50, ghost +100, clear +500, caught -200, wall -5, step -1.
- **Curriculum:** 2 → 3 → 4 ghosts; chase 0.5 → 0.75 → 0.85. Steps split 50/33/17.
- **Training:** 1M steps, about 35 minutes, about 500 steps per second.
- **Result (3 ghosts):** 110.7 vs 10.5 pellets, 145.5 vs 40.5 steps, 3.3% clears, over 30 episodes.
- **Problem:** stage 3 (4 ghosts, full strength) collapsed to 12.6 pellets. Fix: gentler stage. [PENDING]
- **PPO vs DQN (original):** 49.5 vs 19.7 pellets.
- **Bugs we caught:** learned-ghost evaluation mismatch; identical evaluation episodes.
- **Caveat:** one training seed, so preliminary.
- **Lesson:** SB3 makes the algorithm easy; the environment, rewards, curriculum and evaluation are the real work.
