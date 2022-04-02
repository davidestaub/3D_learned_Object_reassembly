import bpy
import os

PATH = os.path.abspath("C:\\Users\\mathi\\OneDrive\\Studium\\ETH\\MA2\\3D Vision\\Project\\Old Project\\object_fracturing\\data\\")

# deselect all objects
bpy.ops.object.select_all(action='DESELECT')    

# loop through all the objects in the scene
scene = bpy.context.scene
for ob in scene.objects:
    # make the current object active and select it
    scene.objects.active = ob
    ob.select = True

    # make sure that we only export meshes
    if ob.type == 'MESH':
        # export the currently selected object to its own file based on its name
        bpy.ops.export_scene.obj(
                filepath=os.path.join(PATH, ob.name + '.obj'),
                use_selection=True,
                )
    # deselect the object and move on to another if any more are left
    ob.select = False