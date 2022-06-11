import itertools
import numpy as np
from numpy.linalg import lstsq
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
from compas.datastructures import mesh_split_face

def dot_product(vector_1, vector_2):
    """return the dot product of two vectors"""
    return sum((a*b) for a, b in zip(vector_1, vector_2))


def length(vector):
    """returns length of vector by calculating sqrt(vec²)"""
    return sqrt(dot_product(vector, vector))


def polyfit3d(x, y, z, order=3):
    """finding the least square solution to fit a polynome
    of certain degree to 3 vectors in space"""
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    m, _, _, _ = lstsq(G, z)
    return m


# Copied from compas and modified to also take care of 5 sidede meshes.
def mesh_faces_to_triangles(mesh):
    """Convert all quadrilateral faces of a mesh to triangles by adding a diagonal edge.
    mesh : :class:`~compas.datastructures.Mesh` A mesh data structure.
    The mesh is modified in place.
    """

    def cut_off_traingle(fkey):
        attr = mesh.face_attributes(fkey)
        attr.custom_only = True
        vertices = mesh.face_vertices(fkey)
        # We skip degenerate faces because compas can't even handle deleting them.
        if len(vertices) >= 4 and len(vertices) == len(set(vertices)):
            a = vertices[0]
            c = vertices[2]
            t1, t2 = mesh_split_face(mesh, fkey, a, c)  # Cut off abc triangle.
            mesh.face_attributes(t1, attr.keys(), attr.values())
            mesh.face_attributes(t2, attr.keys(), attr.values())
            if fkey in mesh.facedata:
                del mesh.facedata[fkey]
            cut_off_traingle(t2)  # t2 still can have more than 3 vertices.

    for fkey in list(mesh.faces()):
        cut_off_traingle(fkey)