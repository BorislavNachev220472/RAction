import pybullet as p
import time
import pybullet_data
import math
import numpy as np
import trimesh


class Simulation():
    def __init__(self, render=True):
        if render:
            mode = p.GUI
        else:
            mode = p.DIRECT
        
        self.physicsClient = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        ## Add camera settings here

        #self.baseplaneId = p.loadURDF("plane.urdf")
        self.robot_file_path = 'Data/cube_robot_new.obj'
        self.base_file_path = 'Data/cube_base_new.obj'
        self.ground_truth_file_path = 'Data/cube_ground_truth_new.obj'
        self.trimesh_object_id = None

        self.create_base_piece()

        self.create_robot()


    def create_base_piece(self):
        self.base_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                fileName=self.base_file_path,
                                                meshScale=[1, 1, 1],
                                                rgbaColor=[0, 0, 0, 0.5])
        
        self.base_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName=self.base_file_path,
                                                meshScale=[1, 1, 1])
        
        self.base_object_id = p.createMultiBody(baseMass=0,
                                            baseCollisionShapeIndex=self.base_collision_shape_id,
                                            baseVisualShapeIndex=self.base_visual_shape_id,
                                            basePosition=[0,0,0])
        """

        self.base_visual_shape_id1 = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                fileName=self.ground_truth_file_path,
                                                meshScale=[1, 1, 1])
        
        self.base_collision_shape_id1 = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName=self.ground_truth_file_path,
                                                meshScale=[1, 1, 1])
        
        self.base_object_id1 = p.createMultiBody(baseMass=0,
                                            baseCollisionShapeIndex=self.base_collision_shape_id1,
                                            baseVisualShapeIndex=self.base_visual_shape_id1,
                                            basePosition=[0,0,2])

        """


    
    def create_robot(self):

        self.visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                            fileName=self.robot_file_path,
                                            meshScale=[1, 1, 1],
                                            rgbaColor=[0, 0, 0, 0.5])  # Adjust meshScale if needed

        self.collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                        fileName=self.robot_file_path,
                                                        meshScale=[1, 1, 1])
    

        #self.robot_cordinates = np.random.uniform(-5, 5, 3)  # Adjust the position of the robot
        robot_cord_x = np.random.uniform(-2, 2)
        robot_cord_y = np.random.uniform(-2, 0)
        robot_cord_z = np.random.uniform(-2, 2)

        self.robot_cordinates = [robot_cord_x, robot_cord_y, robot_cord_z]



        # Create a multibody with a non-zero mass
        self.object_id = p.createMultiBody(baseMass=1,  # Set a non-zero mass for movement
                                        baseCollisionShapeIndex=self.collision_shape_id,
                                        baseVisualShapeIndex=self.visual_shape_id,
                                        basePosition=self.robot_cordinates)

        return self.robot_cordinates
        


    

    def reset(self):
        robot_cord_x = np.random.uniform(-2, 2)
        robot_cord_y = np.random.uniform(-2, 0)
        robot_cord_z = np.random.uniform(-2, 2)
        self.robot_cordinates = [robot_cord_x, robot_cord_y, robot_cord_z]

        p.resetBasePositionAndOrientation(self.object_id, self.robot_cordinates, p.getQuaternionFromEuler([0, 0, 0]))

        self.previous_cord = self.robot_cordinates
        self.previous_orientation = (0, 0, 0, 1)

        self.robot_mesh, self.base_mesh, self.ground_truth_mesh = self.load_in_trimeshes()

        #self.set_robot_mesh_cords()
        self.dot_object_id3 = None
        self.dot_object_id2 = None
        self.dot_object_id4 = None
        self.dot_object_id5 = None
        self.robot_mesh = self.set_mesh_cords(self.robot_mesh, self.robot_cordinates)
        self.base_mesh = self.set_mesh_cords(self.base_mesh, self.base_mesh.centroid*-1)
        self.base_mesh = self.set_mesh_cords(self.base_mesh, [0.5,0,0])
        self.ground_truth_mesh = self.set_mesh_cords(self.ground_truth_mesh, self.ground_truth_mesh.centroid*-1)
        self.ground_truth_mesh = self.set_mesh_cords(self.ground_truth_mesh, [0,0,0])

        self.base_center_of_mass = self.find_center_of_mass(self.base_mesh)
        self.ground_truth_center_of_mess = self.find_center_of_mass(self.ground_truth_mesh)
        self.robot_mesh_center_of_mass = self.find_center_of_mass(self.robot_mesh)

        return self.robot_cordinates, {}
    

    def run(self, action):
        linear_velocity = action[:3]
        angular_velocity = action[3:]

        
        p.resetBaseVelocity(self.object_id, linearVelocity=linear_velocity, angularVelocity=angular_velocity)
        p.stepSimulation()
        position, current_orientation = p.getBasePositionAndOrientation(self.object_id)

        orientation_difference = p.getDifferenceQuaternion(self.previous_orientation, current_orientation)
        self.current_position = np.array(position)

        distance_from_origin = np.array(self.previous_cord) * -1
        self.adjust_robot_mesh_cords(distance_from_origin)

        self.rotate_mesh_with_quaternion(orientation_difference)

        self.adjust_robot_mesh_cords(self.current_position)


        self.previous_cord = self.current_position
        self.previous_orientation = current_orientation


        return self.current_position


    def get_combined_volume(self):
        # Step 3: Create a convex hull or a simple connector structure to connect them
        # You can create a convex hull around both meshes, which will effectively fill the space between them

        combined_points = np.vstack([self.robot_mesh.vertices, self.base_mesh.vertices])
        convex_hull = trimesh.convex.convex_hull(combined_points)

        # Step 4: Combine the original meshes and the connecting hull (or just the two meshes directly)
        combined_mesh = trimesh.util.concatenate([self.robot_mesh, self.base_mesh])

        # Optional: If you want to add the convex hull to fill the gap
        combined_mesh_with_hull = trimesh.util.concatenate([combined_mesh, convex_hull])


        combined_mesh_with_hull.fix_normals()

        return combined_mesh_with_hull.volume
    



    def get_vertices(self):
        return self.robot_mesh.vertices, self.base_mesh.vertices

    def rotate_mesh_with_quaternion(self, quaternion):
        # Convert quaternion to rotation matrix
        q = np.array(quaternion)
        q = q / np.linalg.norm(q)  
        x, y, z, w = q

        rotation_matrix = np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)]
        ])

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix

        # Apply the rotation to the mesh
        self.robot_mesh.apply_transform(transformation_matrix)


    def calculate_distance(self):        
        self.robot_mesh_center_of_mass = self.find_center_of_mass(self.robot_mesh)
        #print("robot cords center of mass:", self.robot_mesh_center_of_mass)
        #print("ground truth cords center of mass:", self.ground_truth_center_of_mess)
        self.middle_distance_point = (self.robot_mesh_center_of_mass - self.ground_truth_center_of_mess) / 2
        distance = np.linalg.norm(self.middle_distance_point - self.ground_truth_center_of_mess)
        return distance, self.robot_mesh_center_of_mass
    

    def calculate_volume_difference(self):
        combined_volume = self.get_combined_volume()
        ground_truth_volume = self.ground_truth_mesh.volume

        return combined_volume - ground_truth_volume
    
    def calculate_reward(self):
       # distance, self.robot_mesh_center_of_mass = self.calculate_distance()
        volume_difference = self.calculate_volume_difference()
        reward = abs(volume_difference)
        self.robot_mesh_center_of_mass = self.find_center_of_mass(self.robot_mesh)
        return reward, self.robot_mesh_center_of_mass


    def load_in_trimeshes(self):
        self.robot_mesh = trimesh.load(self.robot_file_path)
        self.base_mesh = trimesh.load(self.base_file_path)
        self.ground_truth_mesh = trimesh.load(self.ground_truth_file_path)
        return self.robot_mesh, self.base_mesh, self.ground_truth_mesh

    def set_robot_mesh_cords(self):
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, 3] = self.robot_cordinates
        #print("trans matrix:", transformation_matrix[:3, 3])
        self.robot_mesh.apply_transform(transformation_matrix)
        #print(f"Robot mesh cords set {self.robot_cordinates}")


    def set_mesh_cords(self, mesh, cords):


        transformation_matrix = np.eye(4)
        transformation_matrix[:3, 3] = cords
        mesh.apply_transform(transformation_matrix)
        return mesh

    def adjust_robot_mesh_cords(self, action):
        self.robot_mesh.apply_translation(action)
    
    
    def find_center_of_mass(self, mesh):
        avr_x = mesh.vertices[:,0].mean()
        avr_y = mesh.vertices[:,1].mean()
        avr_z = mesh.vertices[:,2].mean()
        return np.array([avr_x, avr_y, avr_z])


    def render(self, mode='human'):
        pass

    def close(self):
        # Disconnect from PyBullet
        p.disconnect()
