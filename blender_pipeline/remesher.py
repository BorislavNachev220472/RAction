import bpy
import math
import bmesh
import os
import numpy as np

input_path = 'blender_pipeline/input'
output_path = 'blender_pipeline/output'

for filename in os.listdir(input_path):
    if ".DS_Store" not in filename:
        # Clear existing objects in the scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        file_path = os.path.join(input_path, filename)
        print(f"Processing: {file_path}")

        fragment_name = file_path.split('/')[-1].split('.')[0]
        fragment_dupe_name = fragment_name + ".001"
        direct_output_path = os.path.join(output_path, fragment_name)
        os.makedirs(direct_output_path, exist_ok=True)

        gt_path = os.path.join(direct_output_path, f"{fragment_name}_ground_truth.stl")
        agent_path = os.path.join(direct_output_path, f"{fragment_name}_agent.stl")
        base_path = os.path.join(direct_output_path, f"{fragment_name}_base.stl")
        distance_path = os.path.join(direct_output_path, f"{fragment_name}_distances.npy")

        # Import the mesh
        bpy.ops.wm.stl_import(filepath=file_path, forward_axis='NEGATIVE_Y', up_axis='Z')
        current_object = bpy.context.active_object
        if not current_object:
            print(f"Error: Failed to load object from {file_path}")
            continue

        # Set origin to center of volume
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='BOUNDS')
        center_of_volume = np.array([current_object.location.x, current_object.location.y, current_object.location.z])

        # Reset object location and rotation
        #current_object.location = (0, 0, 0)
        #current_object.rotation_euler[0] += math.radians(180)
        bpy.ops.object.transform_apply()

        # Apply remesh modifier
        remesh_modifier = current_object.modifiers.new(name='Remesh', type='REMESH')
        remesh_modifier.mode = 'VOXEL'
        remesh_modifier.voxel_size = 0.6
        remesh_modifier.adaptivity = 1
        bpy.ops.object.modifier_apply(modifier=remesh_modifier.name)

        # Scale object and apply transformations
        bpy.ops.object.transform_apply()

        # Calculate random planes centered at the center of volume
        plane_co_rand = tuple(center_of_volume)
        plane_no_rand = tuple(np.random.uniform(-0.5, 0.5, 3))

        # Recalculate normals
        mesh_data = current_object.data
        current_bmesh = bmesh.new()
        current_bmesh.from_mesh(mesh_data)
        bmesh.ops.recalc_face_normals(current_bmesh, faces=current_bmesh.faces)
        current_bmesh.to_mesh(mesh_data)
        current_bmesh.free()

        # Export ground truth STL
        bpy.ops.wm.stl_export(
            filepath=gt_path,
            export_selected_objects=True,
            apply_modifiers=True,
            up_axis='Z',
            forward_axis='Y',
            global_scale=1
        )

        # Duplicate object
        bpy.ops.object.duplicate(linked=False)
        duplicated_object = bpy.context.active_object

        # Bisect for base
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.bisect(
            plane_co=plane_co_rand,
            plane_no=plane_no_rand,
            use_fill=True,
            clear_inner=True
        )
        bpy.ops.object.mode_set(mode='OBJECT')

        # Export base STL
        bpy.ops.wm.stl_export(
            filepath=base_path,
            export_selected_objects=True,
            apply_modifiers=True,
            up_axis='Z',
            forward_axis='Y',
            global_scale=1
        )

        # Hide original object
        original_object = bpy.data.objects.get(fragment_name)
        if original_object:
            original_object.hide_viewport = True
            original_object.hide_render = True

        # Restore duplicate and prepare for agent STL
        bpy.context.view_layer.objects.active = duplicated_object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.bisect(
            plane_co=plane_co_rand,
            plane_no=plane_no_rand,
            use_fill=True,
            clear_outer=True
        )
        bpy.ops.object.mode_set(mode='OBJECT')

        # Export agent STL
        bpy.ops.wm.stl_export(
            filepath=agent_path,
            export_selected_objects=True,
            apply_modifiers=True,
            up_axis='Z',
            forward_axis='Y',
            global_scale=1
        )

        # Cleanup objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
