from custom_enviroment.env import CustomEnviroment
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import os 
import numpy as np
from clearml import Task, Logger
from gymnasium import spaces

task = Task.init(
    project_name="MARL",
    task_name="MARL_RLib_PettingZoo",
    tags=["RL", "PPO", "PyBulletEnv"]
)

def env_creator():
    env = CustomEnviroment(render=False)
    env._agent_ids = ['agent_' + str(i) for i in range(3)]
    print(env._agent_ids)
    return env

ray.init()

env_name = "CustomEnv"

register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator()))

test_env = env_creator()
obs_space = test_env.observation_space()
act_space = test_env.action_space()

config = (
    PPOConfig()
    .environment(env=env_name, disable_env_checking=True)
    .rollouts(num_rollout_workers=1, rollout_fragment_length=30)
    .training(train_batch_size=512)
    .multi_agent(
            policies={
                "shared_policy": (None, obs_space, act_space, {}),
                },
                policy_mapping_fn=(lambda agent_id, *args, **kwargs: "shared_policy"),

        )
)

algo = config.build()

i = 0
while True:
    results = algo.train()

    if i % 5 == 0:

        timesteps_total = results.get("timesteps_total", 0)
        rewards = results.get("hist_stats", {}).get("episode_reward", [])
        mean_reward = np.mean(rewards) if rewards else 0


        Logger.current_logger().report_scalar(
            title="Timesteps",
            series="Total Timesteps",
            value=timesteps_total,
            iteration=timesteps_total,
        )
    checkpoint_path = algo.save(f"checkpoints/{timesteps_total}")