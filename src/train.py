from stable_baselines.common.policies import LstmPolicy
from stable_baselines.a2c import A2C
import tensorflow as tf

from src.env import CBNEnv


def train():
    sess = tf.Session()
    env = CBNEnv.create(2)
    n_steps = 1024 / env.num_envs

    policy = LstmPolicy(sess=sess,
                        ob_space=env.observation_space,
                        ac_space=env.action_space,
                        n_env=env.num_envs,
                        n_steps=n_steps,
                        n_batch=1024,
                        n_lstm=192,
                        layers=[],
                        feature_extraction="mlp")

    model = A2C(policy=policy,
                env=env,
                gamma=0.93,
                n_steps=n_steps,
                learning_rate=9e-6,
                lr_schedule='linear',
                verbose=1)

    model.learn(total_timesteps=int(1e7))


if __name__ == "__main__":
    train()