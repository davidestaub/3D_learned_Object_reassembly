----------Data--------------------
The objects are stored in the data folder in a coresponding subfolder.
In these subfolder you have the original output of the Blender fracturing process.
There are shards and the "whole" object consisting of all the fractures in their original place.

Each object folder gets a cleaned subfolder once the clear_data script is run.
This script searches for if there are actually multiple fragments in each shard.
It also deletes small objects with less than 100 vertices.
The result is stored in the cleaned folder as obj and npy files.

----------Scripts-----------------
clear_data:
deletes very small fragments and makes sure that each obj file only
contains one connected piece. It also deletes innecessary files.

visualize_data:
visualizes the output of blender with compas.

visualize_cleaned_data:
visualizes the output of the cleaned data where we have normally
more fragments.

mesh_explosion:
this expands the object such that each fragment can be viewed independently

mesh_explosion_cleaned:
this expands the object such that each fragment can be viewed independently
but works in the cleaned subfolder

subdivide:
divides all the vertices in half until they are less in size than twice the size of
the biggest vertice (why? I don't know yet...)
