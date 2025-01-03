from env_wrapper import PyBulletEnv
from stable_baselines3 import PPO


env = PyBulletEnv()
env.reset()


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"logs/ppo")



while True:
    model.learn(total_timesteps=10000, reset_num_timesteps=False, progress_bar=True)
    model.save("models/ppo")

for _ in range(1000000):
    action = env.action_space.sample()  # Sample random action
    obs, reward, done, _ = env.step(action)
