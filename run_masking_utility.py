#!/usr/bin/env python3
"""
run_masking_utility.py
----------------------
Velocity-masking utility experiment (Section 5.3 in the paper).

CartPole obs = [cart_pos, cart_vel, pole_angle, pole_angular_vel]
Masking velocities creates a POMDP where:
  - MLP can't infer velocity from a single obs -> poor performance
  - Frame stacking [o_t, o_{t-1}, o_{t-2}] lets MLP estimate velocity -> recovery

We also compute MVS on both conditions to show MVS detects the violation.

Usage:
    python run_masking_utility.py              # full run (5 seeds, 100k steps)
    python run_masking_utility.py --smoke      # quick test (1 seed, 10k steps)
    python run_masking_utility.py --seeds 3 --steps 50000
"""

import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class VelocityMaskWrapper(gym.ObservationWrapper):
    """Mask velocity dimensions from CartPole observations."""
    def __init__(self, env):
        super().__init__(env)
        # CartPole: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
        # Mask indices 1 and 3 (velocities)
        self.mask_indices = [1, 3]

    def observation(self, obs):
        masked = obs.copy()
        for i in self.mask_indices:
            masked[i] = 0.0
        return masked


class FrameStackFlatWrapper(gym.Wrapper):
    """Stack k frames into flat vector for MlpPolicy compatibility."""
    def __init__(self, env, k=3):
        super().__init__(env)
        self.k = k
        self.frames = []
        obs_dim = env.observation_space.shape[0]
        low = np.tile(env.observation_space.low, k)
        high = np.tile(env.observation_space.high, k)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs.copy() for _ in range(self.k)]
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.pop()
        self.frames.insert(0, obs.copy())
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(self.frames, axis=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(model, env, n_episodes=20):
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
    return rewards


def compute_mvs(env_maker, n_episodes=50, seed=42):
    """Compute MVS on trajectory data from the given env."""
    from markovianess.ci.prediction_test import PredictionMarkovTest

    # Collect trajectories and concatenate into one long sequence
    env = env_maker()
    all_obs = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_obs = [obs.copy()]
        done = False
        while not done:
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            ep_obs.append(obs.copy())
            done = terminated or truncated
        all_obs.append(np.array(ep_obs))
    env.close()

    # Concatenate all episodes into one 2D array (T, N)
    obs_concat = np.concatenate(all_obs, axis=0)

    # Compute MVS
    tester = PredictionMarkovTest(use_nn=False)
    result = tester.compute_mvs(obs_concat)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Velocity-masking utility experiment (CartPole)")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test: 1 seed, 10k steps")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds (default: 5)")
    parser.add_argument("--steps", type=int, default=100_000,
                        help="Training timesteps per seed (default: 100000)")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 1
        args.steps = 10_000

    from stable_baselines3 import PPO

    STEPS = args.steps
    SEEDS = args.seeds

    conditions = {
        'Full obs + MLP': {'mask': False, 'stack': False},
        'Full obs + FrameStack': {'mask': False, 'stack': True},
        'Masked + MLP': {'mask': True, 'stack': False},
        'Masked + FrameStack': {'mask': True, 'stack': True},
    }

    print('=' * 70)
    print('VELOCITY MASKING EXPERIMENT: CartPole-v1')
    print('=' * 70)
    print(f'Steps: {STEPS}, Seeds: {SEEDS}')
    print('Conditions:')
    print('  Full obs: [cart_pos, cart_vel, pole_angle, pole_angular_vel]')
    print('  Masked:   [cart_pos, 0,        pole_angle, 0]')
    print('  FrameStack: k=3, [o_t, o_{t-1}, o_{t-2}]')
    print()

    # Step 1: Compute MVS for both observation types
    print('--- Computing MVS ---')

    def make_full():
        return gym.make('CartPole-v1')

    def make_masked():
        return VelocityMaskWrapper(gym.make('CartPole-v1'))

    mvs_full = compute_mvs(make_full, n_episodes=50)
    print(f'  MVS (full obs):    {mvs_full["mvs"]:.4f}')

    mvs_masked = compute_mvs(make_masked, n_episodes=50)
    print(f'  MVS (masked vel):  {mvs_masked["mvs"]:.4f}')

    print()
    if mvs_masked['mvs'] > mvs_full['mvs']:
        print('  => MVS correctly detects that masking creates non-Markov obs!')
    else:
        print('  => MVS does NOT detect masking (unexpected)')
    print()

    # Step 2: Train policies under all conditions
    print('--- Training policies ---')

    results = {}
    for cond_name, cond in conditions.items():
        rewards_all = []
        for seed in range(SEEDS):
            t0 = time.time()

            # Training env
            env = gym.make('CartPole-v1')
            if cond['mask']:
                env = VelocityMaskWrapper(env)
            if cond['stack']:
                env = FrameStackFlatWrapper(env, k=3)

            model = PPO('MlpPolicy', env, verbose=0, seed=seed)
            model.learn(total_timesteps=STEPS)

            # Eval env (same setup)
            eval_env = gym.make('CartPole-v1')
            if cond['mask']:
                eval_env = VelocityMaskWrapper(eval_env)
            if cond['stack']:
                eval_env = FrameStackFlatWrapper(eval_env, k=3)

            ep_rewards = evaluate(model, eval_env, n_episodes=20)
            mean_r = np.mean(ep_rewards)
            rewards_all.append(mean_r)
            env.close()
            eval_env.close()
            dt = time.time() - t0
            print(f'  {cond_name} s={seed}: '
                  f'reward={mean_r:.1f} ({dt:.0f}s)')

        results[cond_name] = rewards_all
        mean = np.mean(rewards_all)
        std = np.std(rewards_all)
        print(f'  >> {cond_name}: {mean:.1f} +/- {std:.1f}')
        print()

    # Step 3: Summary
    print('=' * 70)
    print('RESULTS SUMMARY')
    print('=' * 70)
    print()
    print(f'MVS (full obs):   {mvs_full["mvs"]:.4f}  '
          f'=> {"Markov" if mvs_full["mvs"] < 0.05 else "Non-Markov"}')
    print(f'MVS (masked vel): {mvs_masked["mvs"]:.4f}  '
          f'=> {"Markov" if mvs_masked["mvs"] < 0.05 else "Non-Markov"}')
    print()

    print(f'{"Condition":<25} {"Mean Reward":>12} {"Std":>8}')
    print('-' * 50)
    for cond_name in conditions:
        vals = results[cond_name]
        print(f'{cond_name:<25} {np.mean(vals):>12.1f} {np.std(vals):>8.1f}')
    print()

    # Key comparisons
    full_mlp = np.mean(results['Full obs + MLP'])
    full_fs = np.mean(results['Full obs + FrameStack'])
    mask_mlp = np.mean(results['Masked + MLP'])
    mask_fs = np.mean(results['Masked + FrameStack'])

    print('KEY FINDINGS:')
    print(f'  1. Masking degrades MLP: {full_mlp:.0f} -> {mask_mlp:.0f} '
          f'({mask_mlp - full_mlp:+.0f})')
    print(f'  2. FrameStack recovers: {mask_mlp:.0f} -> {mask_fs:.0f} '
          f'({mask_fs - mask_mlp:+.0f})')
    print(f'  3. FrameStack on full obs: {full_mlp:.0f} -> {full_fs:.0f} '
          f'({full_fs - full_mlp:+.0f}) (unnecessary complexity)')
    print()

    if mask_fs > mask_mlp + 50 and full_mlp >= full_fs - 20:
        print('  => SUCCESS: MVS-guided strategy works!')
        print('     MVS ~ 0 -> use MLP (simpler, equally good)')
        print('     MVS > 0 -> use FrameStack (recovers performance)')
        guided_mean = (full_mlp * SEEDS + mask_fs * SEEDS) / (2 * SEEDS)
        always_mlp = (full_mlp * SEEDS + mask_mlp * SEEDS) / (2 * SEEDS)
        always_fs = (full_fs * SEEDS + mask_fs * SEEDS) / (2 * SEEDS)
        print()
        print(f'  Strategy comparison (avg across conditions):')
        print(f'    Always MLP:    {always_mlp:.1f}')
        print(f'    Always FS:     {always_fs:.1f}')
        print(f'    MVS-guided:    {guided_mean:.1f}  (best of both)')
    else:
        print('  => Mixed results. FrameStack recovery may need more steps.')

    # Save JSON summary
    summary = {
        "mvs_full_obs": round(mvs_full["mvs"], 6),
        "mvs_masked": round(mvs_masked["mvs"], 6),
        "conditions": {},
        "steps": STEPS,
        "seeds": SEEDS,
    }
    for cond_name in conditions:
        vals = results[cond_name]
        summary["conditions"][cond_name] = {
            "mean_reward": round(float(np.mean(vals)), 2),
            "std_reward": round(float(np.std(vals)), 2),
            "per_seed_rewards": [round(float(v), 2) for v in vals],
        }

    out_dir = "results/utility"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "masking_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved summary to {out_path}')


if __name__ == '__main__':
    main()
