import os
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.a2c import A2C

from src.env import CBNEnv


def train():
    env = CBNEnv.create(2)
    n_steps = int(1024 / env.num_envs)

    policy_kwargs = dict(
        n_lstm=192,
        layers=[],
        feature_extraction="mlp"
    )

    logdir = '../logs'
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    model = A2C(policy=LstmPolicy,
                env=env,
                gamma=0.93,
                n_steps=n_steps,
                learning_rate=9e-6,
                lr_schedule='linear',
                policy_kwargs=policy_kwargs,
                tensorboard_log=logdir,
                verbose=1,)

    model.learn(total_timesteps=int(1e7))


if __name__ == "__main__":
    train()