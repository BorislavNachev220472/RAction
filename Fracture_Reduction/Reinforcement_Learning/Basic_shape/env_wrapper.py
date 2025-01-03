import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import time
import trimesh
import scipy.ndimage
class PyBulletEnv(gym.Env):
    def __init__(self, render):
        super(PyBulletEnv, self).__init__()

        self.sim = Simulation(render=render)

        self.max_steps = 2000

        self.action_space = spaces.Box(low=np.array([-3, -3, -3, -1, -1, -1]), high=np.array([3, 3, 3, 1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)


        self.robot_cordinates, _ = self.sim.reset()

        self.observation = np.array(self.robot_cordinates, dtype=np.float32)

        self.observation = np.array(self.observation, dtype=np.float32)

        self.steps = 0
        return self.observation, {}


    def step(self, action):
        _ = self.sim.run(action)


        #distance, self.observation = self.sim.calculate_distance()

        
        volume_difference, self.observation = self.sim.calculate_reward()
        self.reward = (10 /  volume_difference - 1) / 2

        self.observation = np.array(self.observation, dtype=np.float32)

        if volume_difference < 0.1:
            self.reward += 20000
            done = True
        else:
            done = False

        if self.steps > self.max_steps:
            self.reward -= 900
            truncated = True
        else:
            truncated = False
        
        self.steps += 1

        return self.observation, self.reward, done, truncated, {}