import os
from stable_baselines import logger
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.a2c import A2C

from src.env import CBNEnv


def train():
    env = CBNEnv.create(1)
    n_steps = int(1024 / env.num_envs)

    logdir, logger = setup_logger()
    env.set_attr("logger", logger)
    log_callback = lambda locals, globals: env.env_method("log_callback")

    policy_kwargs = dict(
        n_lstm=192,
        layers=[],
        feature_extraction="mlp"
    )

    model = A2C(
        policy=LstmPolicy,
        env=env,
        alpha=0.95,
        gamma=0.93,
        n_steps=1024,
        vf_coef=0.05,
        ent_coef=0.25,
        learning_rate=1e-4,
        max_grad_norm=10000,
        lr_schedule='linear',
        policy_kwargs=policy_kwargs,
        tensorboard_log=logdir,
        verbose=1,
        seed=94566
    )

    model.learn(
        total_timesteps=int(1e7),
        callback=log_callback
    )


def setup_logger():
    logdir = '../logs'
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    logger.configure(folder=logdir, format_strs=['tensorboard', 'stdout'])
    logger_instance = logger.Logger.CURRENT

    return logdir, logger_instance


if __name__ == "__main__":
    train()