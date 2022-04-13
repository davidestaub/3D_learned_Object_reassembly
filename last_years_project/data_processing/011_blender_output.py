import bpy
import os

folder = "\\home\\mathias\\blender_exports\\cylinder\\"

# deselect all objects
bpy.ops.object.select_all(action='DESELECT')    

# loop through all the objects in the scene
scene = bpy.context.scene
for ob in scene.objects:
    # make the current object active and select it
    bpy.context.view_layer.objects.active = ob
    bpy.context.active_object.select_set(state=True)

    # make sure that we only export meshes
    if ob.type == 'MESH':
        print("Mesh of", ob.name, "found! Exporting...")
        # export the currently selected object to its own file based on its name
        bpy.ops.export_scene.obj(
                filepath= folder + ob.name + ".obj",
                use_selection=True,
                )
    # deselect the object and move on to another if any more are left
    bpy.context.active_object.select_set(state=False)

print("Export finished")
