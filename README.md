# Quantifying Markov Violations in Reinforcement Learning — Source Code

Companion code for *"Quantifying Markov Violations in Reinforcement Learning"*
(Anonymous, 2025).

This package reproduces all three experiments from the paper:

1. **MVS Sensitivity (Section 5.1):** Post-hoc AR(1) noise injection shows MVS
   increases monotonically with noise strength across 6 environments and 3 RL
   algorithms.
2. **Reward Degradation (Section 5.2):** Training under AR(1) noise degrades
   policy performance, and the degree correlates with MVS.
3. **Utility / Velocity Masking (Section 5.3):** MVS detects when velocity
   masking creates a POMDP in CartPole, and frame-stacking guided by MVS
   recovers performance.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

Requires Python 3.9+. MuJoCo environments (HalfCheetah, Hopper, Walker2d)
require a working MuJoCo installation (bundled with the `mujoco` pip package).

## Quick Smoke Test (~2 minutes)

Validate that everything runs end-to-end with minimal compute:

```bash
python run_experiments.py --smoke
python analyze_results.py
python run_masking_utility.py --smoke
```

The smoke test uses 1 seed and 2048 training steps per environment.

## Full Reproduction

### Experiment 1 & 2: MVS Sensitivity + Reward Degradation

```bash
# All phases, all environments (10 seeds, ~48 CPU-hours)
python run_experiments.py --seeds 10 --workers 8

# Analyze and produce plots
python analyze_results.py
```

#### Filtering

Run a subset of environments or algorithms:

```bash
# Single environment
python run_experiments.py --env CartPole-v1 --seeds 10

# Single algorithm
python run_experiments.py --algo PPO --seeds 10

# Specific phase only
python run_experiments.py --phase phase1a   # train clean policies
python run_experiments.py --phase phase1b   # compute MVS (requires phase1a)
python run_experiments.py --phase phase2    # reward impact

# Resume interrupted Phase 2 runs
python run_experiments.py --phase phase2 --skip-existing
```

### Experiment 3: Velocity Masking Utility

```bash
# Full run (5 seeds, 100k steps per condition)
python run_masking_utility.py

# Custom configuration
python run_masking_utility.py --seeds 3 --steps 50000
```

## Expected Outputs

### run_experiments.py

All results are saved to `results/final/`:

| File | Description |
|------|-------------|
| `cache/` | Trained models + observation caches |
| `phase1_mvs.jsonl` | Per-(env, algo, seed, alpha) MVS scores |
| `phase2_reward.jsonl` | Per-(env, algo, seed, alpha) reward data |

### analyze_results.py

Reads from `results/final/` and produces:

| File | Description |
|------|-------------|
| `phase1_summary.csv` | MVS mean, std, 95% CI per (env, algo, alpha) |
| `phase1_monotonicity.csv` | Spearman rho of MVS vs alpha per (env, algo) |
| `phase1_welch.csv` | Welch t-test (clean vs alpha=0.9) with BH correction |
| `phase1_random.csv` | Random-policy MVS (specificity check) |
| `phase2_summary.csv` | Reward mean, std, 95% CI per (env, algo, alpha) |
| `phase2_degradation.csv` | Reward ratio vs clean baseline |
| `phase2_welch.csv` | Welch t-test on rewards with BH correction |
| `combined_correlation.csv` | Spearman rho(MVS, reward ratio) |
| `plots/phase1_mvs_vs_alpha.pdf` | Figure: MVS vs noise level |
| `plots/phase2_reward_vs_alpha.pdf` | Figure: reward vs noise level |
| `plots/combined_mvs_vs_reward.pdf` | Figure: MVS-reward scatter |

### run_masking_utility.py

| File | Description |
|------|-------------|
| `results/utility/masking_results.json` | MVS values + per-condition rewards |

## Hardware Estimates

| Configuration | Time |
|--------------|------|
| Smoke test (`--smoke`) | ~2 min |
| CartPole only, 10 seeds | ~30 min |
| Full (6 envs, 10 seeds) | ~48 CPU-hours |
| Utility experiment (5 seeds) | ~15 min |

Parallelism is controlled by `--workers` (default: 4). On a machine with 16+
cores, `--workers 16` significantly reduces wall-clock time.

## Code Structure

```
run_experiments.py              # Main experiment runner (Sections 5.1, 5.2)
analyze_results.py              # Analysis + plotting
run_masking_utility.py          # Utility experiment (Section 5.3)
markovianess/
  algorithms.py                 # SB3 algorithm registry (PPO, A2C, SAC)
  callbacks.py                  # Reward tracking callback
  ci/
    prediction_test.py          # Core MVS algorithm (Section 3)
  wrappers/
    simple_ar_wrapper.py        # AR(1) noise wrapper (Section 4.1)
```

### Paper Section Mapping

| Paper Section | Code |
|--------------|------|
| Section 3 (MVS definition) | `markovianess/ci/prediction_test.py` |
| Section 4.1 (AR noise model) | `markovianess/wrappers/simple_ar_wrapper.py`, `run_experiments.py:inject_ar_noise()` |
| Section 5.1 (MVS sensitivity) | `run_experiments.py` phases 1a + 1b |
| Section 5.2 (Reward degradation) | `run_experiments.py` phase 2 |
| Section 5.3 (Utility) | `run_masking_utility.py` |
| Tables & Figures | `analyze_results.py` |

## Platform Notes

- **macOS:** If you encounter `RuntimeError` related to multiprocessing, set
  `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` before running, or use
  `--workers 1`.
- **Windows:** File locking for parallel JSONL writes falls back to a no-op.
  This is safe for `--workers 1`; for higher parallelism on Windows, run
  phases sequentially.
- **GPU:** PyTorch is included for the optional NN stage of MVS (disabled by
  default). All experiments use CPU-only scikit-learn models.
