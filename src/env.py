from collections import defaultdict
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv

from src.causal import CausalGraph


def one_hot(length, idx):
    one_hot = np.zeros(length)
    one_hot[idx] = 1
    return one_hot


class CBNEnv(Env):
    def __init__(self):
        """Create a stable_baselines-compatible environment to train policies on"""
        self.action_space = Discrete(8)
        self.observation_space = Box(-5, 5, (17,))
        self.state = EnvState()

        self.logger = None
        self.log_data = defaultdict(int)

    @classmethod
    def create(cls, n_env):
        return DummyVecEnv([lambda: cls() for _ in range(n_env)])

    def step(self, action):
        selected_node = action % 4
        info = dict()
        intervene_obs = np.zeros(4)
        if self.state.info_phase:
            if action > 3:
                # inappropriate action for phase
                reward = -10.
                self.log_data['wrong_phase_info'] += 1
            else:
                reward = 0.
                self.log_data['right_phase_info'] += 1
                self.state.intervene(selected_node, 5)
            observed_vals = self.state.sample_all()

            if self.state.info_steps == 3:
                # prep for quiz phase
                # create one-hot observation for node intervened on
                intervened_node = np.random.randint(0, 4)
                intervene_obs = one_hot(4, intervened_node)
                self.state.intervene(intervened_node, -5)
            done = False
        else:
            observed_vals = self.state.sample_all()
            if action <= 3:
                # inappropriate action for phase
                reward = -10.
                self.log_data['wrong_phase_quiz'] += 1
            else:
                reward = observed_vals[selected_node]
                self.log_data['right_phase_quiz'] += 1
            done = True

        # concatenate all data that goes into an observation
        obs_tuple = (observed_vals, intervene_obs, self.state.prev_action, self.state.prev_reward)
        obs = np.concatenate(obs_tuple)
        # step the environment state
        new_prev_action = one_hot(8, action)
        self.state.step_state(new_prev_action, np.array([reward]))
        return obs, reward, done, info

    def log_callback(self):
        for k, v in self.log_data.items():
            self.logger.logkv(k, v)
        self.log_data = defaultdict(int)

    def reset(self):
        self.state.reset()
        observed_vals = self.state.sample_all()
        intervene_obs = np.zeros(4)
        obs_tuple = (observed_vals, intervene_obs, self.state.prev_action, self.state.prev_reward)
        obs = np.concatenate(obs_tuple)
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
        self.info_phase = None
        self.info_steps = None
        self.prev_action = None
        self.prev_reward = None
        self.graph = None

        self.reset()

    def step_state(self, new_prev_action, new_prev_reward):
        self.prev_action = new_prev_action
        self.prev_reward = new_prev_reward
        if not self.info_phase:
            self.info_steps = 0
            self.info_phase = True
        else:
            self.info_steps += 1
            if self.info_steps == 4:
                self.info_phase = False

    def intervene(self, node_idx, intervene_val):
        self.graph.intervene(node_idx, intervene_val)

    def sample_all(self):
        return self.graph.sample_all()[:-1]

    def get_value(self, node_idx):
        return self.graph.get_value(node_idx)

    def reset(self):
        self.info_phase = True
        self.info_steps = 0
        self.prev_action = np.zeros(8)
        self.prev_reward = np.zeros(1)
        self.graph = CausalGraph()


class DebugEnvState(EnvState):
    def __init__(self):
        super().__init__()
        self.reward_data = None
