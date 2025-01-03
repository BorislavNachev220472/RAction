import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np

class PyBulletEnv(gym.Env):
    def __init__(self):
        super(PyBulletEnv, self).__init__()
        self.physicsClient = p.connect(p.GUI)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)  # [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)  # [x, y, z]

        # Connect to PyBullet
        p.connect(p.DIRECT)  # Use p.GUI if you want to see the simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        

        self.baseplaneId = p.loadURDF("plane.urdf")


        # Load plane
        #self.plane_id = p.loadURDF("plane.urdf")

        # Load the .obj file using a custom visual shape
        self.visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                  fileName='shifted_shaft.obj',
                                                  meshScale=[1, 1, 1])  # Adjust meshScale if needed

        self.collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                        fileName='shifted_shaft.obj',
                                                        meshScale=[1, 1, 1])
        
        # Create a multibody with a non-zero mass
        self.object_id = p.createMultiBody(baseMass=1,  # Set a non-zero mass for movement
                                           baseCollisionShapeIndex=self.collision_shape_id,
                                           baseVisualShapeIndex=self.visual_shape_id,
                                           basePosition=[0, 0, 0])  # Initial position

        # Target coordinate
        self.target_position = np.array([5.0, 0.0, 0.0])
        
        # Reset environment
        self.reset()

    def reset(self):
        # Reset the object's position
        p.resetBasePositionAndOrientation(self.object_id, [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
        self.current_position = np.array([0.0, 0.0, 0.0])
        return self.current_position

    def step(self, action):
        # Apply the action to the object
        linear_velocity = action[:3]
        angular_velocity = action[3:]
        p.resetBaseVelocity(self.object_id, linearVelocity=linear_velocity, angularVelocity=[0,0,0])
        
        # Step the simulation
        p.stepSimulation()
        
        # Get the current position of the object
        position, _ = p.getBasePositionAndOrientation(self.object_id)
        self.current_position = np.array(position)
        print(self.current_position)
        
        # Compute the reward (negative distance to target)
        distance = np.linalg.norm(self.current_position - self.target_position)
        reward = -distance
        
        # Check if done (you can define your own condition for the episode to end)
        done = distance < 0.1
        
        return self.current_position, reward, done, {}

    def render(self, mode='human'):
        env.render(mode)
        pass

    def close(self):
        # Disconnect from PyBullet
        p.disconnect()

# Example usage:
env = PyBulletEnv()
obs = env.reset()
for _ in range(1000000):
    action = env.action_space.sample()  # Sample random action
    obs, reward, done, _ = env.step(action)

env.close()

