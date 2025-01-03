import pybullet as p
import time
import pybullet_data
import math
import numpy as np


class Simulation():
    def __init__(self, render=True):
        if render:
            mode = p.GUI
        else:
            mode = p.DIRECT
        
        self.physicsClient = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        ## Add camera settings here

        self.baseplaneId = p.loadURDF("plane.urdf")

        self.create_robot()

    
    def create_robot(self):

        self.visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                            fileName='Data/Test1/frag_05.obj',
                                            meshScale=[1, 1, 1])  # Adjust meshScale if needed

        self.collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                        fileName='Data/Test1/frag_05.obj',
                                                        meshScale=[1, 1, 1])
        
        # Create a multibody with a non-zero mass
        self.object_id = p.createMultiBody(baseMass=1,  # Set a non-zero mass for movement
                                           baseCollisionShapeIndex=self.collision_shape_id,
                                           baseVisualShapeIndex=self.visual_shape_id,
                                           basePosition=[0, 0, 0])  # Initial position
        

    def reset(self):
        p.resetBasePositionAndOrientation(self.object_id, [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
        self.current_position = np.array([0.0, 0.0, 0.0])
        return self.current_position, {}
    

    def run(self, action):
        linear_velocity = action[:3]
        #angular_velocity = action[0][3:]
        p.resetBaseVelocity(self.object_id, linearVelocity=linear_velocity, angularVelocity=[0,0,0])
        p.stepSimulation()
        position, _ = p.getBasePositionAndOrientation(self.object_id)
        self.current_position = np.array(position)
        return self.current_position
    
    def render(self, mode='human'):
        pass

    def close(self):
        # Disconnect from PyBullet
        p.disconnect()
