import trimesh

# Raw data location

filename = "frag_05.stl"

raw_data_location = "Data/raw/"
obj_data_location = "Data/obj/"


# Load the .stl file
mesh = trimesh.load_mesh(f'{raw_data_location}{filename}')

# Export the mesh as .obj
mesh.export(f'{obj_data_location}{filename[:-4]}.obj')

print(f'{filename} has been converted to .obj format')