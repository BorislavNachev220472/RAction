import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
import os
import cv2
import random
import numpy as np
from caseloader import CaseLoader

class Simulation:
    def __init__(self, render):
        if render:
            mode = p.GUI
            self.physicsClient = p.connect(mode)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(rgbBackground=[255, 255, 255])
        else:
            mode = p.DIRECT
            self.physicsClient = p.connect(mode)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())


        data_folder = "Data/"
        new_data_folder = "Data_new/"

        objects = os.listdir(data_folder)
        amount_of_objects = len(objects)

        """
        for object in objects:
            for object_piece in os.listdir(f"{data_folder}{object}/"):
                if object_piece.endswith(".stl"):
                    os.makedirs(f"{new_data_folder}{object}", exist_ok=True)
                    new_file = f"{new_data_folder}{object}/{object_piece}"
                    file = f"{data_folder}{object}/{object_piece}"
                    self.center_of_volume_piece(file, new_file)
        """

        data_folder = new_data_folder
        objects = os.listdir(data_folder)
        amount_of_objects = len(objects)


        self.objects = objects
        self.amount_of_objects = amount_of_objects
        self.data_folder = data_folder
        self.agent_object_ids = None
        self.base_object_id = None
        self.agent_ground_truth_object_id = None
        self.binary_threshold_image = False
        self.width = 256
        self.height = 256
        self.number_of_agents = 3
        self.agents = None
        self.init = True
        self.log_vertices = False

        self.agent_names = [f"agent_{i}" for i in range(self.number_of_agents)]



    def reset(self):
        random_object = random.choice(self.objects)
        base_object_file_path = f"{self.data_folder}{random_object}/base.stl"

        if not self.agents == None:
            p.removeBody(self.base_object_id)
        base_object = self.load_in_object_pybullet(base_object_file_path, 0)
        base_cordinates, base_euler = self.get_random_cord_and_euler()
        base_orentation = p.getQuaternionFromEuler(base_euler)
        p.resetBasePositionAndOrientation(base_object, base_cordinates, p.getQuaternionFromEuler(base_euler))


        if not self.agents == None:
            for agent in self.agents:
                agent_object_id = agent["agent_id"]
                p.removeBody(agent_object_id)
        self.agents = []

        if self.log_vertices:
            if not self.init:
                for dot_id in self.dot_ids:
                    p.removeBody(dot_id)

        self.dot_ids = []
        for i in range(self.number_of_agents):
            agent_file_path = f"{self.data_folder}{random_object}/agent_{i}.stl"
            agent = self.load_in_object_pybullet(agent_file_path, 1)


            _, vertices = p.getMeshData(agent, linkIndex=-1, flags=p.MESH_DATA_SIMULATION_MESH)
            p.resetBasePositionAndOrientation(agent, base_cordinates, base_orentation)
            current_cordinate, current_orentation = p.getBasePositionAndOrientation(agent)
            ground_truth_vertices = self.transform_to_world(vertices, current_cordinate, current_orentation)


            cordinates, euler = self.get_random_cord_and_euler()
            orentation = p.getQuaternionFromEuler(euler)
            p.resetBasePositionAndOrientation(agent, cordinates, orentation)
            self.agents.append({"agent": f"agent_{i}",
                            "agent_id": agent,
                            "cordinates": cordinates,
                            "euler": euler,
                            "true_vertices": ground_truth_vertices,
                            "vertices": vertices,
                            "base_cordinates": base_cordinates,
                            "base_euler": base_euler})
                            
            if self.log_vertices:
                self.create_vertice_dot(i, ground_truth_vertices)

        p.stepSimulation()
        self.base_object_id = base_object

        self.init = False
        observations = {
            x['agent']: tuple(x['cordinates']) + tuple(x['euler']) + tuple(x['base_cordinates']) + tuple(x['base_euler'])
            for x in self.agents
        }

        return observations


    def run(self, actions):

        for key in actions.keys():
            action = actions[key]
            agent_data = next((x for x in self.agents if x['agent'] == key), None)
            agent_object_id = agent_data["agent_id"]
            linear_velocity = action[0:3]
            angular_velocity = action[3:6]
            p.resetBaseVelocity(agent_object_id, linearVelocity=linear_velocity, angularVelocity=angular_velocity)

        p.stepSimulation()

        for i in range(self.number_of_agents):
            agent = self.agents[i]
            agent_object_id = agent["agent_id"]
            cordinates, orentation = p.getBasePositionAndOrientation(agent_object_id)
            euler = p.getEulerFromQuaternion(orentation)

            agent["cordinates"] = cordinates
            agent["euler"] = euler

        #print(self.agents)

        observations = {
            x['agent']: tuple(x['cordinates']) + tuple(x['euler']) + tuple(x['base_cordinates']) + tuple(x['base_euler'])
            for x in self.agents
        }

        return observations

    def calculate_vertice_distance_pybullet(self, agent_vertices, agent_position, agent_orentation, true_vertices):

        world_vertices = self.transform_to_world(agent_vertices, agent_position, agent_orentation)

        distances = []
        for x in range(len(world_vertices)):
            distances.append(np.linalg.norm(world_vertices[x] - true_vertices[x]))
        distances = np.mean(distances)
        return distances
    
    def get_vertice_distance(self, agent):
        agent_data = next((x for x in self.agents if x['agent'] == agent), None)
        true_vertices = agent_data["true_vertices"]
        agent_position = agent_data["cordinates"]
        agent_euler = agent_data["euler"]
        agent_vertices = agent_data["vertices"]
        agent_orentation = p.getQuaternionFromEuler(agent_euler)

        distances = self.calculate_vertice_distance_pybullet(agent_vertices, agent_position, agent_orentation, true_vertices)

        return distances

    def create_vertice_dot(self, i, cordinates):
            if i == 0:
                rgba = [0, 1, 0, 1]
            elif i == 1:
                rgba = [0, 0, 1, 1]
            elif i == 2:
                rgba = [1, 0, 0, 1]
            elif i == 3:
                rgba = [1, 1, 0, 1]
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.05,  
                rgbaColor=rgba
            )


            for vertice in cordinates:
                dot_id = p.createMultiBody(baseMass=0,
                                            baseCollisionShapeIndex=-1,
                                            baseVisualShapeIndex=visual_shape,
                                            basePosition=vertice)
                self.dot_ids.append(dot_id)

    def transform_to_world(self, vertices, position, orientation):
        # Convert orientation (quaternion) to rotation matrix
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        world_vertices = []

        for vertex in vertices:
            # Apply rotation and translation
            local_vertex = np.array(vertex)
            world_vertex = np.dot(rotation_matrix, local_vertex) + np.array(position)
            world_vertices.append(world_vertex)

        return world_vertices


    def get_random_cord_and_euler(self):
        x1, y1, z1 = random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(0, 2)
        roll, pitch, yaw = random.uniform(-3.14, 3.14), random.uniform(-3.14, 3.14), random.uniform(-3.14, 3.14)
        cordinates = [x1, y1, z1]
        euler = [roll, pitch, yaw]
        return cordinates, euler

    def load_in_object_pybullet(self, file_path, baseMass, rgbaColor=None):

        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                        fileName=file_path,
                                        meshScale=[1, 1, 1],
                                        rgbaColor=rgbaColor)

        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                        fileName=file_path,
                                                        meshScale=[1, 1, 1])
        object_id = p.createMultiBody(baseMass=baseMass, 
                                        baseCollisionShapeIndex=collision_shape_id,
                                        baseVisualShapeIndex=visual_shape_id,
                                        basePosition=[0, 0, 0])
        return object_id

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