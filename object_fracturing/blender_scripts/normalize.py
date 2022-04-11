import bpy
from mathutils import Vector

obj = bpy.context.object

#Eventually apply transforms (comment if unwanted)
bpy.ops.object.transform_apply( rotation = True, scale = True )

minX = min( [vertex.co[0] for vertex in obj.data.vertices] )
minY = min( [vertex.co[1] for vertex in obj.data.vertices] )
minZ = min( [vertex.co[2] for vertex in obj.data.vertices] )

vMin = Vector( [minX, minY, minZ] )

maxDim = max(obj.dimensions)

if maxDim != 0:
    for v in obj.data.vertices:
        v.co -= vMin #Set all coordinates start from (0, 0, 0)
        v.co /= maxDim #Set all coordinates between 0 and 1
else:
    for v in obj.data.vertices:
        v.co -= vMin
        
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
obj.location[0] = 0 # x
obj.location[1] = 0 # y