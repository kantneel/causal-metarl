from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv

from .causal import CausalGraph


class CBNEnv(Env):
    def __init__(self):
        """Create a stable_baselines-compatible environment to train policies on"""
        self.action_space = Discrete(8)
        self.observation_space = Box(-5, 5, (8,))
        self.state = EnvState()

    @classmethod
    def create(cls, n_env):
        return DummyVecEnv([lambda: cls for _ in range(n_env)])

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


class EnvState(object):
    def __init__(self):
        """Create an object which holds the state of a CBNEnv"""
        self.info_phase = True
        self.graph = CausalGraph()

    def intervene(self, node_idx, val):
        self.graph.intervene(node_idx, val)
