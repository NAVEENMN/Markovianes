"""
simple_ar_wrapper.py
--------------------
Simple observation wrapper that adds AR(1) noise to ALL observation dims.

Unlike ARObservationWrapper (which uses learned transition models and blending),
this wrapper just injects correlated noise post-observation:
    noised_obs[d] = obs[d] + noise[d]
    noise[d]_t = alpha * noise[d]_{t-1} + eps_t
    eps_t ~ N(0, (sigma_scale * obs_std[d])^2)
"""

import gymnasium as gym
import numpy as np


class SimpleARWrapper(gym.ObservationWrapper):
    """Adds AR(1) noise to all observation dimensions."""

    def __init__(self, env, alpha, sigma_scale, obs_std, seed=None):
        """
        Parameters
        ----------
        env : gym.Env
        alpha : float
            AR(1) autocorrelation coefficient in [0, 1).
        sigma_scale : float
            Innovation std = sigma_scale * obs_std per dim.
        obs_std : np.ndarray
            Per-dimension observation std (pre-computed from clean rollout).
        seed : int or None
            RNG seed for reproducibility.
        """
        super().__init__(env)
        self.alpha = alpha
        self.sigma_per_dim = sigma_scale * (np.asarray(obs_std, dtype=np.float64) + 1e-8)
        self.rng = np.random.RandomState(seed)
        self.noise_state = np.zeros(self.observation_space.shape[0], dtype=np.float64)

    def reset(self, **kwargs):
        self.noise_state = np.zeros_like(self.noise_state)
        return super().reset(**kwargs)

    def observation(self, obs):
        eps = self.rng.randn(len(self.noise_state)) * self.sigma_per_dim
        self.noise_state = self.alpha * self.noise_state + eps
        return obs + self.noise_state.astype(obs.dtype)
