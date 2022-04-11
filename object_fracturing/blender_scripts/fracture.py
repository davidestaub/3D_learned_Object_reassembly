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

filename = 'bottle_1'

ob = bpy.context.active_object

# variables
count = 20

# modifier = object.modifiers.new(name="Fracture", frac_algorithm='BOOLEAN_FRACTAL')
bpy.ops.object.modifier_add(type='FRACTURE')
md = ob.modifiers["Fracture"]
md.fracture_mode = 'PREFRACTURED'
md.frac_algorithm = 'BOOLEAN_FRACTAL'
md.fractal_cuts = 2
md.fractal_iterations = 4
md.shard_count = count
md.point_seed = random()  # random seed

bpy.ops.object.fracture_refresh(reset=True)
bpy.ops.object.rigidbody_convert_to_objects()
