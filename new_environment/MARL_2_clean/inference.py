from custom_enviroment.env import CustomEnviroment
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import os 
import numpy as np
from gymnasium import spaces

from ray.rllib.algorithms.ppo import PPO

def env_creator():
    env = CustomEnviroment(render=True)
    env._agent_ids = ['agent_' + str(i) for i in range(3)]
    print("AGENT IDS")
    print(env._agent_ids)
    return env

ray.init()

env_name = "CustomEnv"
env = env_creator()
register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator()))


checkpoint_path = "2400300"

PPOagent = PPO.from_checkpoint(checkpoint_path)



reward_sum = 0
frame_list = []
i = 0

observations, _ = env.reset()
print(type(observations), observations)  # Inspect observations

reward_sum = 0
i = 0

while True:
    actions = {}
    for agent_id, obs in observations.items():
        # Use the shared policy for all agents
        actions[agent_id] = PPOagent.compute_single_action(obs, policy_id="shared_policy")
    
    observations, rewards, terminations, truncations, infos= env.step(actions)
    reward_sum += sum(rewards.values())  # Sum rewards across all agents
    i += 1

    # Check if all agents are done
    if all(terminations.values()):  # Stop when all agents are done
        print("Break!")
        observations, _ = env.reset()
    if all(truncations.values()):
        print("Truncation!")
        observations, _ = env.reset()