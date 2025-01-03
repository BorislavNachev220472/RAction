from env_wrapper import PyBulletEnv
from stable_baselines3 import PPO
import os

env = PyBulletEnv(render=False)
env.reset()


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"logs/ppo")

name = "PPO"
id = "PPO_3"

models_dir = f"models/{id}"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = PyBulletEnv(render=False)
env.reset()

TIMESTEPS = 10000
i = 0

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"{logdir}/{id}")

while True:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True)
    model.save(f"models/{id}/{TIMESTEPS*(i)}")
    i += 1