import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation 

class PyBulletEnv(gym.Env):
    def __init__(self, render):
        super(PyBulletEnv, self).__init__()

        self.sim = Simulation(render=render, vertices_logging=False)

        self.max_steps = 2048
        
        self.action_space = spaces.Box(low=np.array([-10, -10, -10, -5, -5, -5]), high=np.array([10, 10, 10, 5, 5, 5]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
    
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.observation = self.sim.reset()
        self.observation = np.array(self.observation, dtype=np.float32).flatten()

        self.difficulty = self.observation[-1]
        print(f"Difficulty {self.difficulty}")
        self.observation = np.delete(self.observation, -1)
        self.observation = np.delete(self.observation, -1)
        self.resets = self.observation[-1]
        print(f"Resets {self.resets}")
        self.observation = np.delete(self.observation, -1)
        print(self.observation)

        self.steps = 0

        self.reward = 0
        return self.observation, {}


    def step(self, action):
        self.observation = self.sim.run(action)


        #distance, self.observation = self.sim.calculate_distance()

        
        volume_difference = self.sim.calculate_reward()

        self.observation = np.array(self.observation, dtype=np.float32).flatten()



        volume_difference_new = abs(volume_difference)


        if volume_difference < 15:
            self.reward = ((5 /  volume_difference_new - 1) / 2) * self.difficulty
        else:
            self.reward = (volume_difference_new*-1 / 10) * self.difficulty

        if (volume_difference < 0.01):
            self.reward += 20000 * self.difficulty
            done = True
        else:
            done = False

        if self.steps > self.max_steps:
            self.reward -= 800 * self.difficulty
            truncated = True
        else:
            truncated = False
        
        self.steps += 1




        return self.observation, self.reward, done, truncated, {}