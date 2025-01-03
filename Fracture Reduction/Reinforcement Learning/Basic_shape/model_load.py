from env_wrapper import PyBulletEnv
from stable_baselines3 import PPO
import numpy as np
import random
env = PyBulletEnv(render=True)

env.reset()


model_path = "models/PPO_3/110000.zip"
model = PPO.load(model_path, env=env)


observation, info = env.reset()

while True:
    action, _ = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    #print(f"Reward: {reward}")
    if terminated == True:
        done = True
        print("Finished!")
        observation, info = env.reset()
        #print(f"Finished in {info[1]} steps")
        #print(f"Euc distance: {info[0]}")
        #print(f"Goal position: {observation[3:]} End position: {observation[:3]}")
    if truncated == True:
        done = True
        print("Sorry, couldn't reach that.")
        #print(f"Eucl distance: {info[0]}")
        #print(f"Goal position: {observation[3:]} End position: {observation[:3]}")
        env.reset()



env.close()
