import bpy
import math
import random
from mathutils import Matrix, Vector

def get_BoundBox(object_name):
    """
    returns the corners of the bounding box of an object in world coordinates
    #  ________ 
    # |\       |\
    # |_\______|_\
    # \ |      \ |
    #  \|_______\|
    # 
    """
    
    ob = bpy.context.scene.objects[object_name]
    bbox_corners = [ob.matrix_world @ Vector(corner) for corner in ob.bound_box]
 
    return bbox_corners
 
 
 
def check_Collision(box1, box2):
    """
    Check Collision of 2 Bounding Boxes
    box1 & box2 muss Liste mit Vector sein,
    welche die Eckpunkte der Bounding Box
    enthält
    #  ________ 
    # |\       |\
    # |_\______|_\
    # \ |      \ |
    #  \|_______\|
    # 
    #
    """
    print('\nKollisionscheck mit:')
 
    x_max = max([e[0] for e in box1])
    x_min = min([e[0] for e in box1])
    y_max = max([e[1] for e in box1])
    y_min = min([e[1] for e in box1])
    z_max = max([e[2] for e in box1])
    z_min = min([e[2] for e in box1])
    print('Box1 min %.2f, %.2f, %.2f' % (x_min, y_min, z_min))
    print('Box1 max %.2f, %.2f, %.2f' % (x_max, y_max, z_max))    
     
    x_max2 = max([e[0] for e in box2])
    x_min2 = min([e[0] for e in box2])
    y_max2 = max([e[1] for e in box2])
    y_min2 = min([e[1] for e in box2])
    z_max2 = max([e[2] for e in box2])
    z_min2 = min([e[2] for e in box2])
    print('Box2 min %.2f, %.2f, %.2f' % (x_min2, y_min2, z_min2))
    print('Box2 max %.2f, %.2f, %.2f' % (x_max2, y_max2, z_max2))        
     
     
    isColliding = ((x_max >= x_min2 and x_max <= x_max2) \
                    or (x_min <= x_max2 and x_min >= x_min2)) \
                    and ((y_max >= y_min2 and y_max <= y_max2) \
                    or (y_min <= y_max2 and y_min >= y_min2)) \
                    and ((z_max >= z_min2 and z_max <= z_max2) \
                    or (z_min <= z_max2 and z_min >= z_min2))
 
    if isColliding:
        print('Kollision!')
         
    return isColliding
 
# MAIN
# Check Collision of Objects named 'Cube1' and 'Cube2' in Scene
#
def create_cut_object():
    print("Start")
    # Clear all existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Add a cube to the scene
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    cube1 = bpy.context.object

    # Duplicate the cube to get a second half
    cube2 = cube1.copy()
    cube2.data = cube1.data.copy()
    bpy.context.collection.objects.link(cube2)

    # Add a plane to act as the cutting object
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    cutter = bpy.context.object

    # Subdivide the plane to create more geometry
    bpy.ops.object.mode_set(mode='EDIT')  # Switch to edit mode
    bpy.ops.mesh.subdivide(number_cuts=10)  # Subdivide the plane into a grid
    bpy.ops.object.mode_set(mode='OBJECT')  # Switch back to object mode

    # Generate random displacements for each vertex
    for vert in cutter.data.vertices:
        # Generate a random displacement value for each vertex's z-axis
        random_displacement = random.uniform(-0.1, 0.1)  # Adjust range as necessary
        vert.co.z += random_displacement  # Apply the random displacement to the z-coordinate
        
    # Rotate the plane so it cuts the cube in half
    cutter.rotation_euler = (1.5708, 0, 0)  # Rotate 90 degrees on the X-axis


    # Boolean modifier for the first half (DIFFERENCE)
    bool_modifier_1 = cube1.modifiers.new(name="Boolean_Diff", type='BOOLEAN')
    bool_modifier_1.operation = 'DIFFERENCE'
    bool_modifier_1.object = cutter

    # Apply the boolean modifiers
    bpy.context.view_layer.objects.active = cube1
    bpy.ops.object.modifier_apply(modifier=bool_modifier_1.name)


     # Switch to Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')

    # Recalculate normals (outward by default)
    bpy.ops.mesh.normals_make_consistent(inside=False)

    # Switch back to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')


    # Boolean modifier for the second half (DIFFERENCE)
    bool_modifier_2 = cube2.modifiers.new(name="Boolean_Diff", type='BOOLEAN')
    bool_modifier_2.operation = 'DIFFERENCE'
    bool_modifier_2.object = cutter

    bpy.context.view_layer.objects.active = cube2
    bpy.ops.object.modifier_apply(modifier=bool_modifier_2.name)


    # Remove the cutter object (optional, to keep the scene clean)
    bpy.data.objects.remove(cutter, do_unlink=True)





    # Adjust the locations of the cubes to separate them
    cube2.location.y += 0.1
    cube1.location.y -= 0.1
    if check_Collision(get_BoundBox('Cube'),get_BoundBox('Cube.001')):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)




create_cut_object()
        
