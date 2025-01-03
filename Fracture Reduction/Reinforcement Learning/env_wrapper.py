import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class PyBulletEnv(gym.Env):
    def __init__(self):
        super(PyBulletEnv, self).__init__()

        self.sim = Simulation(render=False)

        self.max_steps = 1000

        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.goal_position = np.random.uniform(-1, 1, 3)
        self.observation = self.sim.reset()

        self.observation = np.append(self.observation[0], self.goal_position)
        self.observation = np.array(self.observation, dtype=np.float32)

        self.steps =0 

        return self.observation, {}

    def step(self, action):
        self.current_position = self.sim.run(action)

        self.observation = np.append(self.current_position, self.goal_position)
        self.observation = np.array(self.observation, dtype=np.float32)

        distance = np.linalg.norm(self.current_position - self.goal_position)
        self.reward = -distance
        
        
        if distance < 0.1:
            self.reward += 100
            done = True

        else:
            done = False

        if self.steps > self.max_steps:
            truncated = True
            self.reward -= 15
        else:
            truncated = False

        self.steps += 1
        return self.observation, self.reward, done, truncated, {}



