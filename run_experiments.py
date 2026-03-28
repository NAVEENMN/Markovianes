#!/usr/bin/env python3
"""
run_experiments.py
------------------
Experiment runner for "Quantifying Markov Violations in Reinforcement Learning"
(Anonymous, 2025).

Three phases:
  1a. Train clean policies + collect observations
  1b. Compute MVS with post-hoc AR(1) noise injection
  2.  Train policies under AR(1) noise, measure reward impact

Usage:
    python run_experiments.py                          # all phases
    python run_experiments.py --phase phase1a          # train clean models only
    python run_experiments.py --phase phase1b          # MVS only (requires 1a)
    python run_experiments.py --phase phase2           # reward impact only
    python run_experiments.py --smoke                  # 1 seed, fast validation
    python run_experiments.py --env CartPole-v1 --algo PPO --seeds 2
"""

import argparse
import json
import logging
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gymnasium as gym
import numpy as np

# ---------------------------------------------------------------------------
# Cross-platform file locking
# ---------------------------------------------------------------------------
try:
    import fcntl

    def _lock(f):
        fcntl.flock(f, fcntl.LOCK_EX)

    def _unlock(f):
        fcntl.flock(f, fcntl.LOCK_UN)

except ImportError:
    # Windows: no fcntl — fall back to no-op locking.
    # Safe when --workers 1 or when concurrent writes are unlikely to collide.
    def _lock(f):
        pass

    def _unlock(f):
        pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("run_experiments.log"),
    ],
)
log = logging.getLogger("run_experiments")
logging.getLogger("stable_baselines3").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results/final")
CACHE_DIR = RESULTS_DIR / "cache"
PHASE1_FILE = RESULTS_DIR / "phase1_mvs.jsonl"
PHASE2_FILE = RESULTS_DIR / "phase2_reward.jsonl"

ALPHAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
SIGMA_SCALE = 0.5
N_OBS = 5000
N_EVAL_EPISODES = 10

ENV_ALGO_MATRIX = [
    ("CartPole-v1", ["PPO", "A2C"]),
    ("Pendulum-v1", ["PPO", "A2C", "SAC"]),
    ("Acrobot-v1", ["PPO", "A2C"]),
    ("HalfCheetah-v4", ["PPO", "A2C", "SAC"]),
    ("Hopper-v4", ["PPO", "A2C", "SAC"]),
    ("Walker2d-v4", ["PPO", "A2C", "SAC"]),
]

# Training timesteps per environment (Table 1 in paper)
TIMESTEPS = {
    "CartPole-v1": 50_000,
    "Acrobot-v1": 50_000,
    "Pendulum-v1": 450_000,
    "HalfCheetah-v4": 1_000_000,
    "Hopper-v4": 1_000_000,
    "Walker2d-v4": 1_000_000,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _append_jsonl(filepath, record):
    """Append a JSON line (process-safe via file lock)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    with open(filepath, "a") as f:
        _lock(f)
        f.write(line)
        f.flush()
        _unlock(f)


def inject_ar_noise(observations, alpha, sigma_scale=0.5, seed=42):
    """Add AR(1) noise to ALL dimensions post-hoc."""
    T, N = observations.shape
    noised = observations.copy()
    rng = np.random.RandomState(seed)
    for d in range(N):
        sigma_d = sigma_scale * (np.std(observations[:, d]) + 1e-8)
        noise = np.zeros(T)
        for t in range(1, T):
            noise[t] = alpha * noise[t - 1] + rng.randn() * sigma_d
        noised[:, d] += noise
    return noised


def get_obs_std(env_name, n_steps=2000, seed=0):
    """Collect observations with random actions and return per-dim std."""
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    obs_list = [obs.copy()]
    for _ in range(n_steps - 1):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        obs_list.append(obs.copy())
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.std(np.array(obs_list), axis=0)


def collect_observations(model, env, n_steps=5000):
    """Collect observations from a trained model (deterministic)."""
    obs_list = []
    obs, _ = env.reset()
    for _ in range(n_steps):
        obs_list.append(obs.copy())
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.array(obs_list)


def evaluate_policy(model, env_name, n_episodes=10, wrapper=None, wrapper_kwargs=None):
    """Evaluate policy for n_episodes, return list of episode rewards."""
    env = gym.make(env_name)
    if wrapper is not None:
        env = wrapper(env, **wrapper_kwargs)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    env.close()
    return rewards


def _cache_dir(env_name, algo_name, seed):
    return CACHE_DIR / f"{env_name}_{algo_name}_s{seed}"


# ---------------------------------------------------------------------------
# Phase 1a: Train clean policies + collect observations
# ---------------------------------------------------------------------------

def _phase1a_worker(env_name, algo_name, seed, timesteps, n_obs, n_envs):
    """Train a clean policy, collect observations, save to cache."""
    tag = f"phase1a/{env_name}/{algo_name}/s{seed}"
    t0 = time.perf_counter()
    try:
        logging.getLogger("stable_baselines3").setLevel(logging.WARNING)
        from stable_baselines3.common.env_util import make_vec_env
        from markovianess.algorithms import create_model
        from markovianess.callbacks import RewardTrackingCallback

        cache = _cache_dir(env_name, algo_name, seed)
        cache.mkdir(parents=True, exist_ok=True)

        # Skip if already cached
        if (cache / "obs.npy").exists() and (cache / "model.zip").exists():
            log.info(f"[{tag}] Already cached, skipping.")
            return {"status": "cached", "tag": tag}

        # Train
        vec_env = make_vec_env(env_name, n_envs=n_envs, seed=seed)
        model = create_model(algo_name, vec_env, seed=seed)
        cb = RewardTrackingCallback()
        model.learn(total_timesteps=timesteps, callback=cb)
        vec_env.close()

        # Save model
        model.save(str(cache / "model"))

        # Collect observations with trained model
        eval_env = gym.make(env_name)
        eval_env.reset(seed=seed + 1000)
        observations = collect_observations(model, eval_env, n_steps=n_obs)
        np.save(str(cache / "obs.npy"), observations)

        # Evaluate clean reward
        clean_rewards = evaluate_policy(model, env_name, n_episodes=N_EVAL_EPISODES)
        np.save(str(cache / "rewards.npy"), np.array(clean_rewards))

        # Save training rewards
        np.save(str(cache / "train_rewards.npy"), np.array(cb.get_rewards()))

        dt = time.perf_counter() - t0
        log.info(f"[{tag}] Done in {dt:.0f}s. "
                 f"Mean reward={np.mean(clean_rewards):.1f}")
        return {"status": "ok", "tag": tag, "duration_s": round(dt, 1),
                "mean_reward": float(np.mean(clean_rewards))}

    except Exception:
        dt = time.perf_counter() - t0
        log.error(f"[{tag}] FAILED in {dt:.0f}s:\n{traceback.format_exc()}")
        return {"status": "error", "tag": tag, "error": traceback.format_exc()[:500]}


# ---------------------------------------------------------------------------
# Phase 1b: MVS sensitivity (post-hoc noise injection)
# ---------------------------------------------------------------------------

def _phase1b_worker(env_name, algo_name, seed, alpha, n_obs):
    """Load cached observations, inject AR noise, compute MVS."""
    tag = f"phase1b/{env_name}/{algo_name}/s{seed}/a{alpha}"
    t0 = time.perf_counter()
    try:
        logging.getLogger("stable_baselines3").setLevel(logging.WARNING)
        from markovianess.ci.prediction_test import PredictionMarkovTest

        cache = _cache_dir(env_name, algo_name, seed)
        observations = np.load(str(cache / "obs.npy"))

        # Inject noise (alpha=0.0 means clean)
        if alpha > 0:
            noised = inject_ar_noise(observations, alpha, SIGMA_SCALE,
                                     seed=seed * 1000 + int(alpha * 100))
        else:
            noised = observations

        # Compute MVS
        test = PredictionMarkovTest(use_nn=False)
        result = test.compute_mvs(noised[:n_obs])

        dt = time.perf_counter() - t0
        record = {
            "phase": "phase1b", "env": env_name, "algo": algo_name,
            "seed": seed, "alpha": alpha,
            "mvs": round(result["mvs"], 6),
            "mvs_ridge": round(result["mvs_ridge"], 6),
            "mse_markov": round(float(result["mse_markov"]), 8),
            "mse_history": round(float(result["mse_history"]), 8),
            "status": "ok", "duration_s": round(dt, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _append_jsonl(PHASE1_FILE, record)
        log.info(f"[{tag}] MVS={result['mvs']:.4f} ({dt:.1f}s)")
        return {"status": "ok", "tag": tag, "mvs": result["mvs"]}

    except Exception:
        dt = time.perf_counter() - t0
        record = {
            "phase": "phase1b", "env": env_name, "algo": algo_name,
            "seed": seed, "alpha": alpha,
            "status": "error", "error": traceback.format_exc()[:500],
            "duration_s": round(dt, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _append_jsonl(PHASE1_FILE, record)
        log.error(f"[{tag}] FAILED:\n{traceback.format_exc()}")
        return {"status": "error", "tag": tag}


# ---------------------------------------------------------------------------
# Phase 1b extra: random-policy MVS for specificity check
# ---------------------------------------------------------------------------

def _phase1b_random_worker(env_name, seed, n_obs):
    """Collect random-policy observations and compute MVS (specificity)."""
    tag = f"phase1b_random/{env_name}/s{seed}"
    t0 = time.perf_counter()
    try:
        from markovianess.ci.prediction_test import PredictionMarkovTest

        env = gym.make(env_name)
        obs, _ = env.reset(seed=seed)
        obs_list = [obs.copy()]
        for _ in range(n_obs - 1):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            obs_list.append(obs.copy())
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()
        observations = np.array(obs_list)

        test = PredictionMarkovTest(use_nn=False)
        result = test.compute_mvs(observations)

        dt = time.perf_counter() - t0
        record = {
            "phase": "phase1b_random", "env": env_name, "algo": "random",
            "seed": seed, "alpha": 0.0,
            "mvs": round(result["mvs"], 6),
            "mvs_ridge": round(result["mvs_ridge"], 6),
            "status": "ok", "duration_s": round(dt, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _append_jsonl(PHASE1_FILE, record)
        log.info(f"[{tag}] MVS={result['mvs']:.4f} ({dt:.1f}s)")
        return {"status": "ok", "tag": tag, "mvs": result["mvs"]}

    except Exception:
        dt = time.perf_counter() - t0
        record = {
            "phase": "phase1b_random", "env": env_name, "algo": "random",
            "seed": seed, "alpha": 0.0,
            "status": "error", "error": traceback.format_exc()[:500],
            "duration_s": round(dt, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _append_jsonl(PHASE1_FILE, record)
        log.error(f"[{tag}] FAILED:\n{traceback.format_exc()}")
        return {"status": "error", "tag": tag}


# ---------------------------------------------------------------------------
# Phase 2: Reward impact (train with noise)
# ---------------------------------------------------------------------------

def _phase2_worker(env_name, algo_name, alpha, seed, timesteps, n_envs):
    """Train a policy under AR(1) noise and evaluate reward."""
    tag = f"phase2/{env_name}/{algo_name}/a{alpha}/s{seed}"
    t0 = time.perf_counter()
    try:
        logging.getLogger("stable_baselines3").setLevel(logging.WARNING)
        from markovianess.algorithms import create_model
        from markovianess.callbacks import RewardTrackingCallback
        from markovianess.wrappers.simple_ar_wrapper import SimpleARWrapper

        # Get obs_std from a short clean rollout
        obs_std = get_obs_std(env_name, n_steps=2000, seed=seed)

        # Create wrapped env (n_envs=1 for accurate reward tracking)
        def make_env():
            env = gym.make(env_name)
            if alpha > 0:
                env = SimpleARWrapper(env, alpha=alpha, sigma_scale=SIGMA_SCALE,
                                      obs_std=obs_std, seed=seed * 1000 + int(alpha * 100))
            return env

        env = make_env()
        model = create_model(algo_name, env, seed=seed)
        cb = RewardTrackingCallback()
        model.learn(total_timesteps=timesteps, callback=cb)
        env.close()

        # Evaluate on wrapped env
        wrapper_kwargs = {"alpha": alpha, "sigma_scale": SIGMA_SCALE,
                          "obs_std": obs_std,
                          "seed": seed * 1000 + int(alpha * 100) + 1}
        if alpha > 0:
            eval_rewards = evaluate_policy(model, env_name, N_EVAL_EPISODES,
                                           wrapper=SimpleARWrapper,
                                           wrapper_kwargs=wrapper_kwargs)
        else:
            eval_rewards = evaluate_policy(model, env_name, N_EVAL_EPISODES)

        dt = time.perf_counter() - t0
        record = {
            "phase": "phase2", "env": env_name, "algo": algo_name,
            "alpha": alpha, "seed": seed,
            "mean_reward": round(float(np.mean(eval_rewards)), 4),
            "std_reward": round(float(np.std(eval_rewards)), 4),
            "episode_rewards": [round(float(r), 4) for r in eval_rewards],
            "n_training_episodes": len(cb.get_rewards()),
            "status": "ok", "duration_s": round(dt, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _append_jsonl(PHASE2_FILE, record)
        log.info(f"[{tag}] Reward={np.mean(eval_rewards):.1f}+/-{np.std(eval_rewards):.1f} ({dt:.0f}s)")
        return {"status": "ok", "tag": tag,
                "mean_reward": float(np.mean(eval_rewards))}

    except Exception:
        dt = time.perf_counter() - t0
        record = {
            "phase": "phase2", "env": env_name, "algo": algo_name,
            "alpha": alpha, "seed": seed,
            "status": "error", "error": traceback.format_exc()[:500],
            "duration_s": round(dt, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _append_jsonl(PHASE2_FILE, record)
        log.error(f"[{tag}] FAILED:\n{traceback.format_exc()}")
        return {"status": "error", "tag": tag}


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_phase1a(args, pairs):
    """Train clean policies and collect observations."""
    log.info(f"=== Phase 1a: Training {len(pairs) * args.seeds} clean policies ===")

    jobs = []
    for env_name, algo_name in pairs:
        ts = TIMESTEPS[env_name]
        if args.smoke:
            ts = min(ts, 2048)
        for seed in range(args.seeds):
            jobs.append((env_name, algo_name, seed, ts, N_OBS if not args.smoke else 500,
                         args.n_envs))

    ok, err = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_phase1a_worker, *j): j for j in jobs}
        for fut in as_completed(futures):
            result = fut.result()
            if result["status"] == "error":
                err += 1
            else:
                ok += 1
            log.info(f"Phase 1a progress: {ok + err}/{len(jobs)} "
                     f"(ok={ok}, err={err})")

    log.info(f"=== Phase 1a complete: {ok} ok, {err} errors ===")
    return err == 0


def run_phase1b(args, pairs):
    """Compute MVS with post-hoc noise injection."""
    n_obs = N_OBS if not args.smoke else 500
    total = len(pairs) * args.seeds * len(ALPHAS)
    log.info(f"=== Phase 1b: {total} MVS computations ===")

    jobs = []
    for env_name, algo_name in pairs:
        for seed in range(args.seeds):
            for alpha in ALPHAS:
                jobs.append((env_name, algo_name, seed, alpha, n_obs))

    # Add random-policy specificity checks
    random_jobs = []
    envs_seen = set()
    for env_name, _ in pairs:
        if env_name not in envs_seen:
            envs_seen.add(env_name)
            for seed in range(args.seeds):
                random_jobs.append((env_name, seed, n_obs))

    ok, err = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for j in jobs:
            futures[pool.submit(_phase1b_worker, *j)] = j
        for j in random_jobs:
            futures[pool.submit(_phase1b_random_worker, *j)] = j

        total_all = len(futures)
        for fut in as_completed(futures):
            result = fut.result()
            if result["status"] == "error":
                err += 1
            else:
                ok += 1
            if (ok + err) % 50 == 0 or (ok + err) == total_all:
                log.info(f"Phase 1b progress: {ok + err}/{total_all} "
                         f"(ok={ok}, err={err})")

    log.info(f"=== Phase 1b complete: {ok} ok, {err} errors ===")
    return err == 0


def run_phase2(args, pairs):
    """Train policies under noise and measure reward impact."""
    # Load completed jobs if --skip-existing
    completed = _load_completed_phase2() if args.skip_existing else set()
    if completed:
        log.info(f"--skip-existing: found {len(completed)} completed Phase 2 jobs")

    # Build jobs with priority ordering: PPO/A2C first, then SAC
    fast_jobs = []  # PPO, A2C
    slow_jobs = []  # SAC
    skipped = 0
    for env_name, algo_name in pairs:
        ts = TIMESTEPS[env_name]
        if args.smoke:
            ts = min(ts, 2048)
        for alpha in ALPHAS:
            for seed in range(args.seeds):
                if (env_name, algo_name, alpha, seed) in completed:
                    skipped += 1
                    continue
                job = (env_name, algo_name, alpha, seed, ts, 1)
                if algo_name == "SAC":
                    slow_jobs.append(job)
                else:
                    fast_jobs.append(job)

    jobs = fast_jobs + slow_jobs  # PPO/A2C run first
    total = len(jobs)
    log.info(f"=== Phase 2: {total} jobs ({len(fast_jobs)} PPO/A2C + "
             f"{len(slow_jobs)} SAC, {skipped} skipped) ===")

    if total == 0:
        log.info("No Phase 2 jobs to run.")
        return True

    ok, err = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_phase2_worker, *j): j for j in jobs}
        for fut in as_completed(futures):
            result = fut.result()
            if result["status"] == "error":
                err += 1
            else:
                ok += 1
            if (ok + err) % 20 == 0 or (ok + err) == total:
                log.info(f"Phase 2 progress: {ok + err}/{total} "
                         f"(ok={ok}, err={err})")

    log.info(f"=== Phase 2 complete: {ok} ok, {err} errors ===")
    return err == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_completed_phase2():
    """Load set of completed (env, algo, alpha, seed) tuples from JSONL."""
    completed = set()
    if PHASE2_FILE.exists():
        with open(PHASE2_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        completed.add((r["env"], r["algo"], r["alpha"], r["seed"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


def build_pairs(args):
    """Build (env, algo) pairs, respecting --env and --algo filters."""
    algo_filter = set(args.algo.split(",")) if args.algo else None
    pairs = []
    for env_name, algos in ENV_ALGO_MATRIX:
        if args.env and env_name != args.env:
            continue
        for algo in algos:
            if algo_filter and algo not in algo_filter:
                continue
            pairs.append((env_name, algo))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Experiment runner for Markov Violation Score paper")
    parser.add_argument("--phase", choices=["phase1a", "phase1b", "phase2", "all"],
                        default="all")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick validation: 1 seed, 2048 steps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Vectorized envs for phase1a training")
    parser.add_argument("--env", type=str, default=None,
                        help="Filter to single environment")
    parser.add_argument("--algo", type=str, default=None,
                        help="Filter to algo(s), comma-separated (e.g. PPO,A2C)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip Phase 2 jobs already in phase2_reward.jsonl")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 1
        log.info("SMOKE TEST MODE: 1 seed, minimal timesteps")

    pairs = build_pairs(args)
    if not pairs:
        log.error("No (env, algo) pairs match filters. Check --env / --algo.")
        return

    log.info(f"Running {len(pairs)} (env, algo) pairs: "
             f"{[(e, a) for e, a in pairs]}")
    log.info(f"Seeds={args.seeds}, Workers={args.workers}, "
             f"Phase={args.phase}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    if args.phase in ("phase1a", "all"):
        run_phase1a(args, pairs)

    if args.phase in ("phase1b", "all"):
        run_phase1b(args, pairs)

    if args.phase in ("phase2", "all"):
        run_phase2(args, pairs)

    total_time = time.perf_counter() - t_start
    log.info(f"=== All done in {total_time / 3600:.1f} hours ===")


if __name__ == "__main__":
    main()
