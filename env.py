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
        selected = np.argmax(action)
        selected_node = selected % 4
        info = dict()
        if self.state.info_phase:
            if selected > 3:
                # inappropriate action for phase
                reward = -10
            else:
                reward = 0
                self.state.intervene(selected_node)
            observed_vals = self.state.sample_all()
            intervene_obs = np.zeros(4)
            done = False
        else:
            # quiz phase
            # create one-hot observation for node intervened on
            intervene_obs = np.zeros(4)
            intervened_node = np.random.choice(4, 1)
            intervene_obs[intervened_node] = 1

            self.state.intervene(intervened_node)
            observed_vals = self.state.sample_all()
            if selected <= 3:
                # inappropriate action for phase
                reward = -10
            else:
                reward = observed_vals[selected_node]
            done = True

        # concatenate node values and one-hot for quiz intervention
        obs = np.concatenate((observed_vals, intervene_obs))
        # move along in the episode phases
        self.state.step_phase()
        return obs, reward, done, info

    def reset(self):
        self.state.reset()
        observed_vals = self.state.sample_all()
        intervene_obs = np.zeros(4)
        obs = np.concatenate((observed_vals, intervene_obs))
        return obs

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=94566):
        np.random.seed(seed)


class EnvState(object):
    def __init__(self):
        """Create an object which holds the state of a CBNEnv"""
        self.info_phase = True
        self.info_steps = 0
        self.graph = CausalGraph()

    def step_phase(self):
        if not self.info_phase:
            self.info_phase = True
        else:
            self.info_steps += 1
            if self.info_steps == 4:
                self.info_steps = 0
                self.info_phase = False

    def intervene(self, node_idx):
        intervene_val = 5 if self.info_phase else -5
        self.graph.intervene(node_idx, intervene_val)

    def sample_all(self):
        return self.graph.sample_all()

    def get_value(self, node_idx):
        return self.graph.get_value(node_idx)

    def reset(self):
        self.info_phase = True
        self.info_steps = 0
        self.graph = CausalGraph()
