"""
algorithms.py
-------------
Algorithm registry for RL algorithms (PPO, SAC, A2C).
Handles compatibility filtering by action space type.
"""

import logging

import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC

logger = logging.getLogger("markovianess")

ALGORITHM_REGISTRY = {
    "PPO": {
        "class": PPO,
        "supports_discrete": True,
        "supports_continuous": True,
        "max_n_envs": None,  # no limit
    },
    "SAC": {
        "class": SAC,
        "supports_discrete": False,
        "supports_continuous": True,
        "max_n_envs": None,  # SB3 2.x supports vectorized SAC
    },
    "A2C": {
        "class": A2C,
        "supports_discrete": True,
        "supports_continuous": True,
        "max_n_envs": None,
    },
}


def _is_discrete(env_name):
    """Check if an environment has a discrete action space."""
    env = gym.make(env_name)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    env.close()
    return discrete


def get_compatible_algorithms(env_name, requested=None):
    """
    Return list of algorithm names compatible with the given environment.

    Parameters
    ----------
    env_name : str
    requested : list[str] or None
        If provided, filter to only these algorithms. Otherwise use all.

    Returns
    -------
    list[str]
    """
    discrete = _is_discrete(env_name)
    candidates = requested if requested else list(ALGORITHM_REGISTRY.keys())

    compatible = []
    for algo_name in candidates:
        entry = ALGORITHM_REGISTRY.get(algo_name)
        if entry is None:
            logger.warning(f"Unknown algorithm '{algo_name}', skipping.")
            continue
        if discrete and not entry["supports_discrete"]:
            logger.info(f"Skipping {algo_name} for {env_name} (discrete action space).")
            continue
        if not discrete and not entry["supports_continuous"]:
            logger.info(f"Skipping {algo_name} for {env_name} (continuous action space).")
            continue
        compatible.append(algo_name)

    return compatible


def get_n_envs(algo_name, default_n_envs):
    """
    Return the number of parallel envs to use, respecting algorithm limits.
    """
    entry = ALGORITHM_REGISTRY.get(algo_name, {})
    max_n = entry.get("max_n_envs")
    if max_n is not None:
        return min(default_n_envs, max_n)
    return default_n_envs


def create_model(algo_name, env, learning_rate=3e-4, **kwargs):
    """
    Instantiate an SB3 model from the registry.

    Parameters
    ----------
    algo_name : str
    env : stable_baselines3 VecEnv
    learning_rate : float
    **kwargs : passed to the algorithm constructor

    Returns
    -------
    stable_baselines3 model instance
    """
    entry = ALGORITHM_REGISTRY.get(algo_name)
    if entry is None:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    cls = entry["class"]
    model = cls("MlpPolicy", env, verbose=0, learning_rate=learning_rate, **kwargs)
    logger.info(f"Created {algo_name} model with lr={learning_rate}")
    return model
