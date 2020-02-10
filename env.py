from stable_baselines.common.vec_env import DummyVecEnv
from gym import Env


class CBNVecEnv(Env):
    def __init__(self):
        """Create a stable_baselines-compatible environment to train policies on"""
        # set action_space
        # set observation_space
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass
    
    def seed(self, seed=None):
        pass


def create_env(n_env):
    return DummyVecEnv([lambda: CBNVecEnv() for _ in range(n_env)])