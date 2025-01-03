import pybullet as p
import pybullet_data
import time
import numpy as np
# Connect to PyBullet
p.connect(p.GUI)

# Set the search path to PyBullet's data folder
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the plane URDF
plane_id = p.loadURDF("plane.urdf")

# Load the .obj file using a custom visual shape
visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                      fileName='shifted_shaft.obj',
                                      meshScale=[1, 1, 1])  # Adjust meshScale if needed

collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                            fileName='shifted_shaft.obj',
                                            meshScale=[1, 1, 1])


body_ids = []
for i in range(5):
    body_id = p.createMultiBody(baseMass=0,  # Set a non-zero mass for movement
                            baseCollisionShapeIndex=collision_shape_id,
                            baseVisualShapeIndex=visual_shape_id,
                            basePosition=[0, 0, 0])  # Initial position
    
    body_ids.append(body_id)
    


for body_id in body_ids:
    p.resetBaseVelocity(body_id, linearVelocity=[np.random.uniform(0, 10), np.random.uniform(0, 10), np.random.uniform(0, 10)])


# Run simulation
for i in range(10000):
    p.stepSimulation()
    time.sleep(1. / 240.)

p.disconnect()
