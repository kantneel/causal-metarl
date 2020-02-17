import os
from stable_baselines import logger
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.a2c import A2C

from src.env import CBNEnv


def train():
    env = CBNEnv.create(1)
    n_steps = int(1024 / env.num_envs)

    policy_kwargs = dict(
        n_lstm=192,
        layers=[],
        feature_extraction="mlp"
    )

    logdir, logger = setup_logger()
    env.set_attr("logger", logger)
    log_callback = lambda locals, globals: env.env_method("log_callback")

    model = A2C(policy=LstmPolicy,
                env=env,
                gamma=0.93,
                n_steps=n_steps,
                learning_rate=9e-6,
                lr_schedule='linear',
                policy_kwargs=policy_kwargs,
                tensorboard_log=logdir,
                verbose=1,)

    model.learn(total_timesteps=int(1e7),
                callback=log_callback)


def setup_logger():
    logdir = '../logs'
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    logger.configure(folder=logdir, format_strs=['tensorboard', 'stdout'])
    logger_instance = logger.Logger.CURRENT

    return logdir, logger_instance


if __name__ == "__main__":
    train()