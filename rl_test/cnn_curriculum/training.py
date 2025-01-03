import os
import torch
import argparse
from stable_baselines3 import PPO
from env_wrapper import PyBulletEnv

# Argument parsing for hyperparameters


if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
    device = "cuda"
else:
    print("CUDA is not available. Training on CPU.")
    device = "cpu"


train_locally = False


if not train_locally:

    from clearml import Task
    from stable_baselines3.common.callbacks import BaseCallback
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.00003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--total_timesteps", type=int, default=20000000)
parser.add_argument("--run_name", type=str, default="RUN_7_cnn_curriculum_update")
parser.add_argument("--model_observation", type=str, default="Model observation space is the xyz coordinates of the agents center of volume, the orientation of the agent, xyz coordinates of base center of volume, orientation of the base and the volume difference.")
parser.add_argument("--model_output", type=str, default="Linear movement of the agent in xyz directions. Angular movement of the agent in xyz directions. Angular movement between -0.01 and 0.01.")
parser.add_argument("--reward_desc", type=str, default="Volume difference between combined volume of agent and base and the volume of the ground truth.")
args = parser.parse_args()

if not train_locally:
    task = Task.init(
        project_name="Bone Runs",
        task_name=args.run_name,
        tags=["RL", "PPO", "PyBulletEnv"]
    )

    task.connect({
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "n_epochs": args.n_epochs,
        "model_observation": args.model_observation,
        "model_output": args.model_output,
        "reward_desc": args.reward_desc
    })

env = PyBulletEnv(render=False)
env.reset()


if not train_locally:
    class ClearMLCallback(BaseCallback):
        def __init__(self, model_save_path, save_freq, verbose=0):
            super(ClearMLCallback, self).__init__(verbose)
            self._clearml_logger = task.get_logger()
            self.model_save_path = model_save_path
            self.save_freq = save_freq
            self.previous_r = 0
            self.previous_t = 0
            os.makedirs(model_save_path, exist_ok=True)

        def _on_step(self) -> bool:

            if len(self.model.ep_info_buffer) != 0:
                if self.previous_r != model.ep_info_buffer[-1]['r'] and self.previous_t != model.ep_info_buffer[-1]['t']:
                    self.previous_r = model.ep_info_buffer[-1]['r']
                    self.previous_t = model.ep_info_buffer[-1]['t']
                    last_episode = self.model.ep_info_buffer[-1]
                    ep_rew_length = last_episode['l']  
                    ep_rew_mean = last_episode['r']
                    self._clearml_logger.report_scalar("Avg Reward", "Reward", value = ep_rew_mean, iteration=model.num_timesteps)
                    self._clearml_logger.report_scalar("Avg Reward Length", "Steps", value = ep_rew_length, iteration=model.num_timesteps)

            return True

        def __call__(self, _locals, _globals):
            step_count = _locals['self'].num_timesteps
            if step_count % self.save_freq == 0:
                model_file = os.path.join(self.model_save_path, f"model_{step_count}.zip")
                _locals['self'].save(model_file)
                task.upload_artifact(f"model_{step_count}", model_file)
            return True

    clearml_callback = ClearMLCallback(model_save_path=f"models/{task.id}", save_freq=50000)
else:
    clearml_callback = None

load_pre_trained = True
if load_pre_trained:
    model = PPO.load('models/235520.zip', env=env, device=device)
else:
    print("No pre-trained model found. Creating a new model.")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        device=device,
        verbose=1
    )

TIMESTEPS = 10000
os.makedirs("models", exist_ok=True)
while True:
    model.learn(total_timesteps=TIMESTEPS, callback=clearml_callback, reset_num_timesteps=False, progress_bar=True)
    model.save(f"models/{model.num_timesteps}")