import bpy
import math
import bmesh
import os
import numpy as np
import time

def duplicate_object(frag_name):
    bpy.ops.object.duplicate(linked=False)
    duplicated_obj = bpy.context.object
    duplicated_obj.select_set(True)
    duplicated_obj.name = frag_name

def select_by_name(object_name):
    obj = bpy.data.objects.get(object_name)
    if obj:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        print(f"Object '{object_name}' selected.")
    else:
        print(f"Object '{object_name}' not found.")

def quarter_cut(obj, qrt_nr, obj_name, output_path, fragment_name, plane_co_rand, plane_no_rand, plane_co_rand2, plane_no_rand2):
    quarter = obj.copy()
    quarter.data = obj.data.copy()
    quarter.name = obj_name
    bpy.context.collection.objects.link(quarter)
    bpy.context.view_layer.objects.active = quarter

    if qrt_nr == 1:
        clear_inner_cond1 = True
        clear_outer_cond1 = False
        clear_inner_cond2 = True
        clear_outer_cond2 = False
    if qrt_nr == 2:
        clear_inner_cond1 = True
        clear_outer_cond1 = False
        clear_inner_cond2 = False
        clear_outer_cond2 = True
    if qrt_nr == 3:
        clear_inner_cond1 = False
        clear_outer_cond1 = True
        clear_inner_cond2 = True
        clear_outer_cond2 = False
    if qrt_nr == 4:
        clear_inner_cond1 = False
        clear_outer_cond1 = True
        clear_inner_cond2 = False
        clear_outer_cond2 = True

    if bpy.context.object.mode != 'EDIT':
        bpy.ops.object.mode_set(mode='EDIT')

    time.sleep(0.1)
    bm = bmesh.from_edit_mesh(quarter.data)

    # Ensure the operation is completed before proceeding
    result = bmesh.ops.bisect_plane(bm, geom=bm.faces[:] + bm.edges[:] + bm.verts[:],
                                    plane_co=plane_co_rand, plane_no=plane_no_rand,
                                    clear_inner=clear_inner_cond1, clear_outer=clear_outer_cond1)

    if 'geom_cut' in result:
        bmesh.ops.holes_fill(bm, edges=[ele for ele in result['geom_cut'] if isinstance(ele, bmesh.types.BMEdge)])

    result = bmesh.ops.bisect_plane(bm, geom=bm.faces[:] + bm.edges[:] + bm.verts[:],
                                    plane_co=plane_co_rand2, plane_no=plane_no_rand2,
                                    clear_inner=clear_inner_cond2, clear_outer=clear_outer_cond2)

    if 'geom_cut' in result:
        bmesh.ops.holes_fill(bm, edges=[ele for ele in result['geom_cut'] if isinstance(ele, bmesh.types.BMEdge)])

    bmesh.update_edit_mesh(quarter.data)

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(False)
    bpy.context.view_layer.objects.active = obj

    time.sleep(0.2)
    select_by_name(obj_name)
    time.sleep(0.2)

    bpy.ops.wm.stl_export(filepath=f"{output_path}/{fragment_name}/{obj_name}.stl",
                          export_selected_objects=True, apply_modifiers=True,
                          up_axis='Z', forward_axis='NEGATIVE_Z', global_scale=1)
    obj.select_set(False)

def import_and_process_mesh(file_path, fragment_name, output_path):
    bpy.ops.wm.stl_import(filepath=file_path, forward_axis='NEGATIVE_Y', up_axis='Z')

    current_object = bpy.context.active_object
    if not current_object:
        raise Exception(f"Failed to load object from {file_path}")

    # Change the origin to the center of the volume
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='BOUNDS')

    current_object.location = (0, 0, 0)
    current_object.rotation_euler[0] += math.radians(180)
    bpy.ops.object.transform_apply()

    remesh_modifier = current_object.modifiers.new(name='Remesh', type='REMESH')
    remesh_modifier.mode = 'VOXEL'
    remesh_modifier.voxel_size = 0.6
    remesh_modifier.adaptivity = 1

    bpy.ops.object.modifier_apply(modifier=remesh_modifier.name)

    bpy.ops.object.select_all(action='SELECT')
    current_object.scale = (0.1, 0.1, 0.1)
    bpy.ops.object.transform_apply()

    os.makedirs(f"{output_path}/{fragment_name}", exist_ok=True)
    gt_path = f"{output_path}/{fragment_name}/ground_truth.stl"

    plane_co_rand = tuple(np.random.uniform(-0.5, 0.5, 3))
    plane_no_rand = tuple(np.random.uniform(-0.5, 0.5, 3))
    plane_co_rand2 = tuple(np.random.uniform(-0.5, 0.5, 3))
    plane_no_rand2 = tuple(np.random.uniform(-0.5, 0.5, 3))

    # Export ground truth
    bpy.ops.wm.stl_export(filepath=gt_path, export_selected_objects=True,
                          apply_modifiers=True, up_axis='Z', forward_axis='NEGATIVE_Z', global_scale=1)

    # Now perform the quarter cuts
    quarter_cut(current_object, 1, 'agent1', output_path, fragment_name, plane_co_rand, plane_no_rand, plane_co_rand2, plane_no_rand2)
    quarter_cut(current_object, 2, 'agent2', output_path, fragment_name, plane_co_rand, plane_no_rand, plane_co_rand2, plane_no_rand2)
    quarter_cut(current_object, 3, 'agent3', output_path, fragment_name, plane_co_rand, plane_no_rand, plane_co_rand2, plane_no_rand2)
    quarter_cut(current_object, 4, 'agent4', output_path, fragment_name, plane_co_rand, plane_no_rand, plane_co_rand2, plane_no_rand2)

    try:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
    except Exception as e:
        print(f"Error while deleting objects: {str(e)}")

def main():
    input_path = 'blender_pipeline/input'
    output_path = 'blender_pipeline/output'

    for filename in os.listdir(input_path):
        if ".DS_Store" not in filename:
            file_path = os.path.join(input_path, filename)
            fragment_name = file_path.split('/')[-1].split('.')[0]
            print(fragment_name)
            import_and_process_mesh(file_path, fragment_name, output_path)

main()
