import pybullet as p
import pybullet_data
import time

# Connect to PyBullet
p.connect(p.GUI)

# Set the search path to PyBullet's data folder
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the plane URDF
plane_id = p.loadURDF("plane.urdf")

# Load the .obj file using a custom visual shape
visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                      fileName='shaft.obj',
                                      meshScale=[1, 1, 1])  # Adjust meshScale if needed

collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                            fileName='shaft.obj',
                                            meshScale=[1, 1, 1])

# Create a multibody using the visual and collision shapes
body_id = p.createMultiBody(baseMass=0,  # Set baseMass=0 for a static object
                            baseCollisionShapeIndex=collision_shape_id,
                            baseVisualShapeIndex=visual_shape_id,
                            basePosition=[0, 0, 1])  # Adjust position

# Run simulation
for i in range(10000):
    p.stepSimulation()
    time.sleep(1. / 240.)

p.disconnect()
