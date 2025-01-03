from env_wrapper import PyBulletEnv
from stable_baselines3 import PPO
import numpy as np
import random
import trimesh
import ast
env = PyBulletEnv(render=True)


model_path = "models/256000"
model = PPO.load(model_path, env=env)


observation, info = env.reset()
print(f"Observation: {observation}")
input("Press Enter to continue...")
avr_reward = []
while True:


    action = ""#input("Action:")
    if not action == "":
        if action == "reset":
            observation, info = env.reset()
            print(f"Observation: {observation}")
            continue
        lst = ast.literal_eval(action)

        # Convert the list to a NumPy array
        action = np.array(lst)
    else:
        action, _ = model.predict(observation)

         

    observation, reward, terminated, truncated, info = env.step(action)
    avr_reward.append(reward)

    print(f"Reward: {reward}")
    #print(f"Volume difference: {info}")
    print(f"Observation: {observation}")
    #input("Press Enter to continue...")
    if terminated == True:
        done = True
        print("Finished!")
        input("Press Enter to continue...")
        result = env.sim.save_results()
        result.export('combined_result.stl')
        observation, info = env.reset()
        
        #print(f"Finished in {info[1]} steps")
        #print(f"Euc distance: {info[0]}")
        #print(f"Goal position: {observation[3:]} End position: {observation[:3]}")
    if truncated == True:
        done = True
        print("average reward: ", sum(avr_reward)/len(avr_reward))
        print("Reward:", sum(avr_reward))
        print("Sorry, couldn't reach that.")
        result = env.sim.save_results()
        result.export('combined_result.stl')
        avr_reward = []
        #print(f"Eucl distance: {info[0]}")
        #print(f"Goal position: {observation[3:]} End position: {observation[:3]}")
        env.reset()



env.close()
