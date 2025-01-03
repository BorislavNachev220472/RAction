
from gymnasium import spaces
import numpy as np
from custom_enviroment.Simulation import Simulation 
from pettingzoo import ParallelEnv
from copy import copy
import functools

class CustomEnviroment(ParallelEnv):
    metadata = {
        "name": "CustomEnviroment_v0"
    }

    def __init__(self, render):
        
        
        self.sim = Simulation(render=render)
        self.render_mode = render
        self.possible_agents = self.sim.agent_names

        self.agents = copy(self.possible_agents)

        """
        self.observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Box(
                low=np.array([-10, -10, -10, -5, -5, -5]),
                high=np.array([10, 10, 10, 5, 5, 5]),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        """

        self.max_steps = 2048
        self.timestep = None
        self.rewards = None

    def reset(self, seed=None, options=None):
        self.possible_agents = self.sim.agent_names
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        # Reset the environment and get the initial observations
        observations = self.sim.reset()
        

        
        infos = {agent: {} for agent in self.agents}
        

        observations = {agent: np.array(obs, dtype=np.float32) for agent, obs in observations.items()}

        
        return observations, infos

    
    def step(self, actions):
        self.timestep += 1


        observations = self.sim.run(actions)
        
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        total_distance = []
        for agent in self.agents:
            vertice_distance = self.sim.get_vertice_distance(agent)
            total_distance.append(vertice_distance)

            

            if vertice_distance < 10:
                reward = (5 /  vertice_distance - 1) / 2
            else:
                reward = -1
                
            rewards[agent] = reward

        total_average_distance = np.mean(total_distance)

        if self.timestep >= self.max_steps:
            terminations = {agent: True for agent in self.agents}
            truncations = {agent: True for agent in self.agents}
            self.agents = []

        if total_average_distance < 0.5:
            terminations = {agent: True for agent in self.agents}
            remaining_steps = self.max_steps - self.timestep
            rewards = {agent: remaining_steps*15 for agent in self.agents}
            self.agents = []
            self.sim.reset()
        
        self.rewards = rewards

        observations = {agent: np.array(obs, dtype=np.float32) for agent, obs in observations.items()}
        
        return observations, rewards, terminations, truncations, infos
    
    def last(self):
        return self.rewards

    def render(self, mode="human"):
        pass


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent=None):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent=None):
        return spaces.Box(low=np.array([-10, -10, -10, -5, -5, -5]), high=np.array([10, 10, 10, 5, 5, 5]), dtype=np.float32)

