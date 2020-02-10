from stable_baselines.common.policies import LstmPolicy
from stable_baselines.a2c import A2C
from stable_baselines.ppo2 import PPO2
from .env import create_env


def train():
    env = create_env(2)
    model = PPO2(LstmPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)