import pybullet as p
import pybullet_data
import numpy as np
import trimesh
import math
from scipy.spatial.transform import Rotation as R
from trimesh.collision import CollisionManager
import os


class Simulation():
    def __init__(self, render=True, vertices_logging=False):
        if render:
            mode = p.GUI
            self.physicsClient = p.connect(mode)
        else:
            mode = p.DIRECT
            self.physicsClient = p.connect(mode)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.vertices_logging = False
        self.resets = 0

        data_folder = "Data/"
        objects = os.listdir(data_folder)
        amount_of_objects = len(objects)

        self.objects = objects
        self.amount_of_objects = amount_of_objects
        self.data_folder = data_folder

        self.agent_object_id = None
        self.base_object_id = None

        new_data_folder = "Data_new/"
        self.difficulty = 0

        for object in objects:
            for object_piece in os.listdir(f"{data_folder}{object}/"):
                if object_piece.endswith(".obj"):
                    os.makedirs(f"{new_data_folder}{object}", exist_ok=True)
                    new_file = f"{new_data_folder}{object}/{object_piece}"
                    file = f"{data_folder}{object}/{object_piece}"
                    self.center_of_volume_piece(file, new_file)

        data_folder = new_data_folder
        objects = os.listdir(data_folder)
        amount_of_objects = len(objects)

        self.objects = objects
        self.amount_of_objects = amount_of_objects
        self.data_folder = data_folder

        if self.vertices_logging:
            self.agent_dot = None
            self.base_dot = None
            self.ground_truth_dot = None
            self.ground_truth_agent_dot = None

            self.agent_visual_shape = self.create_dot_visual_shape("agent")
            self.base_visual_shape = self.create_dot_visual_shape("base")
            self.ground_truth_visual_shape = self.create_dot_visual_shape("ground_truth")
            self.ground_truth_agent_shape = self.create_dot_visual_shape("ground_truth_agent")

            





    def reset(self):
        
        self.agent_file_path, self.base_file_path, self.ground_truth_file_path, self.offset = self.get_random_object()
        self.resets += 1
        if self.resets <= 500:
            self.difficulty = self.resets / 500
        else:
            # Logarithmic growth with a cap of 2.5
            cap = 5  # Maximum difficulty value
            base_difficulty = 1  # Difficulty at 500 resets
            growth_rate = 0.005  # Adjust to control how quickly it approaches the cap
            self.difficulty = min(base_difficulty + growth_rate * math.log(self.resets - 99), cap)

        # Load agent
        self.agent_cordinates, self.agent_euler = self.get_random_cord_and_euler("agent")
        #self.agent_euler = [0,0,0]
        agent_orentation = p.getQuaternionFromEuler(self.agent_euler)
        if self.agent_object_id is not None:
            self.delete_object_pybullet(self.agent_object_id)
        self.agent_object_id = self.create_object_pybullet(type="agent")
        p.resetBasePositionAndOrientation(self.agent_object_id, self.agent_cordinates, agent_orentation)


        self.agent_mesh = self.create_mesh_trimesh(self.agent_file_path)
        self.agent_mesh = self.apply_rotation_mesh(self.agent_mesh, [0,0,0], agent_orentation, self.agent_cordinates)


        # Load base
        self.base_cordinates, self.base_euler = self.get_random_cord_and_euler("base")
        #self.base_euler = [0,0,0]
        base_orentation = p.getQuaternionFromEuler(self.base_euler)
        if self.base_object_id is not None:
            self.delete_object_pybullet(self.base_object_id)
        self.base_object_id = self.create_object_pybullet(type="base")
        p.resetBasePositionAndOrientation(self.base_object_id, self.base_cordinates, base_orentation)


        self.base_mesh = self.create_mesh_trimesh(self.base_file_path)
        self.base_mesh = self.apply_rotation_mesh(self.base_mesh, [0,0,0], base_orentation, self.base_cordinates)

        
        self.agent_ground_truth = self.create_mesh_trimesh(self.agent_file_path)
        self.agent_ground_truth = self.apply_rotation_mesh(self.agent_ground_truth, [0,0,0], base_orentation, np.array(self.base_cordinates) + self.offset)


        # load ground truth
        self.ground_truth_mesh = self.create_mesh_trimesh(self.ground_truth_file_path)


        self.ground_truth_volume = self.ground_truth_mesh.volume
        volume_sum = self.agent_mesh.volume + self.base_mesh.volume
        self.ground_truth_difference = self.ground_truth_volume - volume_sum

        p.stepSimulation()

        if self.vertices_logging:
            if self.base_dot is not None:
                self.remove_dots(self.base_dot)
            self.base_dot = self.create_dots("base")
            if self.ground_truth_dot is not None:
                self.remove_dots(self.ground_truth_dot)
            self.ground_truth_dot = self.create_dots("ground_truth")
            if self.ground_truth_agent_dot is not None:
                self.remove_dots(self.ground_truth_agent_dot)
            self.ground_truth_agent_dot = self.create_dots("ground_truth_agent")

        return self.agent_cordinates, self.agent_euler, self.base_cordinates, self.base_euler, [self.resets, 80085, self.difficulty]

    def run(self, action):
        
        linear_movement = action[:3]
        angular_movement= action[3:]

        previous_agent_cordinates = self.agent_cordinates
        previous_agent_euler = self.agent_euler
        previous_agent_orentation = p.getQuaternionFromEuler(previous_agent_euler)



        p.resetBaseVelocity(self.agent_object_id, linearVelocity=linear_movement, angularVelocity=angular_movement)
        p.stepSimulation()

        agent_position, agent_orentation = p.getBasePositionAndOrientation(self.agent_object_id)

        orentation_difference = p.getDifferenceQuaternion(previous_agent_orentation, agent_orentation)

        self.agent_mesh = self.apply_rotation_mesh(self.agent_mesh, previous_agent_cordinates, orentation_difference, agent_position)

        new_agent_euler = p.getEulerFromQuaternion(agent_orentation)

        self.agent_cordinates = agent_position
        self.agent_euler = new_agent_euler

        if self.vertices_logging:
            if self.agent_dot is not None:
                self.remove_dots(self.agent_dot)
            self.agent_dot = self.create_dots("agent")
        
        return self.agent_cordinates, self.agent_euler, self.base_cordinates, self.base_euler

    

    def center_of_volume_piece(self, file, new_file):
        x_total = 0
        y_total = 0
        z_total = 0
        num_vertices = 0

        with open(file, 'r') as file2:
            for line in file2:
                if line.startswith('v '):  # Vertex line
                    parts = line.split()
                    x, y, z = map(float, parts[1:4])
                    x_total += x
                    y_total += y
                    z_total += z
                    num_vertices += 1

        if num_vertices == 0:
            raise ValueError("No vertex data found in the .obj file.")

        x_center = x_total / num_vertices
        y_center = y_total / num_vertices
        z_center = z_total / num_vertices

        center_of_volume = (x_center, y_center, z_center)


        i = 0
        center_of_volume = list(center_of_volume)
        for x in center_of_volume:
            center_of_volume[i] = x * -1
            i += 1
        center_of_volume = tuple(center_of_volume)


        x_shift, y_shift, z_shift = center_of_volume


        with open(file, 'r') as file:
            lines = file.readlines()

        with open(new_file, 'w') as file:
            for line in lines:
                if line.startswith('v '):  # Vertex line
                    parts = line.split()
                    x, y, z = map(float, parts[1:4])
                    x += x_shift
                    y += y_shift
                    z += z_shift
                    file.write(f'v {x} {y} {z}\n')
                else:
                    file.write(line)  # Write non-vertex lines unchanged

    def get_random_cord_and_euler(self, type):
        """
        type: str
            "agent" or "base"
        """
        if type == "agent":
            x = np.random.uniform(-1 , 1)
        elif type == "base":
            x = np.random.uniform(1, 2)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(0, 1)
        cordinates = [x, y, z]
        min_angle = -15 * np.pi / 180 * self.difficulty
        max_angle = 15 * np.pi / 180 * self.difficulty
        
        roll = np.random.uniform(min_angle, max_angle)
        pitch = np.random.uniform(min_angle, max_angle)
        yaw = np.random.uniform(min_angle, max_angle)
        
        euler = [roll, pitch, yaw]
        return cordinates, euler
    
    def delete_object_pybullet(self, object_id):
        p.removeBody(object_id)
    
    def get_random_object(self):
        random_number = 0
        object_path = f"{self.data_folder}{self.objects[random_number]}/"
        for object_parts in os.listdir(object_path):
            if object_parts.endswith("agent.stl"):
                agent_file_path = f"{object_path}{object_parts}"
            elif object_parts.endswith("base.stl"):
                base_file_path = f"{object_path}{object_parts}"
            elif object_parts.endswith("ground_truth.stl"):
                ground_truth_file_path = f"{object_path}{object_parts}"
        if random_number == 0:
            # Cube
            offset = [0.10367, 0.041137, -0.014883]
        elif random_number == 1:
            # Cylinder
            offset = [-0.37000001, -0.30700006999999996, -0.04999997999999997]
        elif random_number == 2:
            #Sphere
            offset = [-0.7099999000000001, -0.15000001000000002, 0.04999998]

        
        return agent_file_path, base_file_path, ground_truth_file_path, np.array(offset)

    def create_object_pybullet(self, type):
        if type == "agent":
            object_file_path = self.agent_file_path
        elif type == "base":
            object_file_path = self.base_file_path

        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName=object_file_path,
                                    meshScale=[1, 1, 1])

        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                        fileName=object_file_path,
                                                        meshScale=[1, 1, 1])
        object_id = p.createMultiBody(baseMass=0,  # Set a non-zero mass for movement
                                        baseCollisionShapeIndex=collision_shape_id,
                                        baseVisualShapeIndex=visual_shape_id,
                                        basePosition=[0, 0, 0])
        return object_id

    def create_mesh_trimesh(self, file_path):
        mesh = trimesh.load_mesh(file_path)
        return mesh
    
    def apply_rotation_mesh(self, mesh, cordinates, orentation, new_cordinates=None):
        if new_cordinates is None:
            new_cordinates = cordinates
        mesh.apply_translation(np.array(cordinates)*-1)
        mesh = self.rotate_mesh_with_quaternion(orentation, mesh)
        mesh.apply_translation(np.array(new_cordinates))
        return mesh


    def rotate_mesh_with_quaternion(self, quaternion, mesh):
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

        mesh.apply_transform(transformation_matrix)
        return mesh
    

    def remove_dots(self, dot_ids):
        for dot_id in dot_ids:
            p.removeBody(dot_id)

    def create_dot_visual_shape(self, type):
        
        if type == "agent":
            rgba = [1, 0, 0, 1]
        elif type == "base":
            rgba = [0, 1, 0, 1]
        elif type == "ground_truth":
            rgba = [0, 0, 1, 1]
        elif type == "ground_truth_agent":
            rgba = [1, 1, 0, 1]
        
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.05,  
            rgbaColor=rgba
        )
        return visual_shape

    def create_dots(self, type):
        if type == "agent":
            visual_shape = self.agent_visual_shape
            mesh = self.agent_mesh
        elif type == "base":
            visual_shape = self.base_visual_shape
            mesh = self.base_mesh
        elif type == "ground_truth":
            visual_shape = self.ground_truth_visual_shape
            mesh = self.ground_truth_mesh
        elif type == "ground_truth_agent":
            visual_shape = self.ground_truth_agent_shape
            mesh = self.agent_ground_truth
        dot_ids = []
        for vertice in mesh.vertices:
            dot_id = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=-1,
                                        baseVisualShapeIndex=visual_shape,
                                        basePosition=vertice)
            dot_ids.append(dot_id)
        
        return dot_ids
            
    def get_combined_volume(self):
        combined_points = np.vstack((self.agent_mesh.vertices, self.base_mesh.vertices))

        convex_hull = trimesh.convex.convex_hull(combined_points)

        approximate_volume = convex_hull.volume

        return(approximate_volume)

    def calculate_volume_difference(self):
        combined_volume = self.get_combined_volume()
        volume_difference = combined_volume - self.ground_truth_volume
        new_volume_difference = volume_difference + self.ground_truth_difference
        volume_difference_percentage = (new_volume_difference / self.ground_truth_volume) * 100

        return new_volume_difference ,volume_difference_percentage
    
    def calculate_vertice_distance(self):
        distances = []
        for x in range(len(self.agent_mesh.vertices)):
            distance = np.linalg.norm(self.agent_mesh.vertices[x] - self.agent_ground_truth.vertices[x])
            distances.append(distance)
        distances = np.mean(distances)
            
        return distances

    def calculate_reward(self):
        #volume_difference = self.calculate_volume_difference()
        vertice_distance = self.calculate_vertice_distance()
        return vertice_distance

    def check_collision(self):
        """
        contact_points = p.getContactPoints(bodyA=self.object_id, bodyB=self.base_object_id)
        print(f"Contact points: {contact_points}")
        if contact_points == ():
            return False    
        else:
            return True 
        """
        manager = CollisionManager()

        manager.add_object("agent", self.agent_mesh)
        manager.add_object("base", self.base_mesh)
        collision = manager.in_collision_internal()
        return collision


    def save_results(self):
        mesh1 = self.agent_mesh
        mesh2 = self.base_mesh
        result = trimesh.util.concatenate([mesh1,mesh2])
        return result
    def render(self, mode='human'):
        pass

    def close(self):
        # Disconnect from PyBullet
        p.disconnect()
