from stable_baselines.common.policies import LstmPolicy
from stable_baselines.a2c import A2C
from stable_baselines.ppo2 import PPO2
import tensorflow as tf

from .env import CBNEnv


def train():
    sess = tf.Session()
    env = CBNEnv.create(2)

    policy = get_policy(sess)

    model = PPO2(LstmPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)


def get_policy(sess):
    pass
