import os
from random import random
import bpy
import bmesh
from bpy.types import Operator
from bpy.props import FloatVectorProperty, IntVectorProperty
from bpy_extras.object_utils import AddObjectHelper, object_data_add

from mathutils import Matrix
bpy.app.debug = True
bpyscene = bpy.context.scene


# loop through all the objects in the scene
for ob in bpyscene.objects:
    if ob.type == 'MESH':
        # make the current object active and select it
        bpyscene.objects.active = ob
        ob.select = True

ob = bpy.context.active_object

# change to edit mode
if bpy.ops.object.mode_set.poll():
    bpy.ops.object.mode_set(mode='EDIT')
print("Active object = ",ob.name)

me = ob.data
bm = bmesh.new()
bm.from_mesh(me)

# subdivide
bmesh.ops.subdivide_edges(bm,
                        edges=bm.edges,
                        cuts=40,
                        use_grid_fill=True,
                        )

# Write back to the mesh
bpy.ops.mesh.select_all(action='DESELECT')
bm.select_flush(True)

bpy.ops.object.mode_set(mode='OBJECT') # if bmesh.from_edit_mesh() --> mode == EDIT - ValueError: to_mesh(): Mesh 'Cube' is in editmode 
bm.to_mesh(me) #If mode ==Object  -> ReferenceError: BMesh data of type BMesh has been removed
bm.free() 
ob.update_from_editmode()