import os
import open3d
import compas
from tools import *
import numpy as np

from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from fractured_object_o3d import FracturedObject
from compas_vis import compas_show
from utils import get_viewer_data
from compas_view2.app import App
import open3d as o3d
import time



here = os.path.dirname(os.path.abspath(__file__))
path = "../object_fracturing/data/"


def animate(source,target):
    gray = [0.5,0.5,0.5]
    green = [0.0,1.0,0.0]
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    source_mesh = []
    target_mesh = []

    radii = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(0, len(source)):
        rec_mesh_source = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(source[i], o3d.utility.DoubleVector(radii))
        rec_mesh_target = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(target[i], o3d.utility.DoubleVector(radii))
        source_mesh.append(rec_mesh_source)
        target_mesh.append(rec_mesh_target)

    for i in range(0,len(source)):
        source[i].paint_uniform_color(gray)
        vis.add_geometry(source[i])
        vis.add_geometry(source_mesh[i])
        vis.poll_events()
        vis.update_renderer()

    time.sleep(5)

    for i in range(0,len(source)):

        source[i].paint_uniform_color(green)
        vis.update_geometry(source[i])
        vis.poll_events()
        vis.update_renderer()
        time.sleep(2)

        target[i].paint_uniform_color(green)
        vis.add_geometry(target[i])
        vis.add_geometry(target_mesh[i])
        vis.poll_events()
        vis.update_renderer()

        time.sleep(2)

        source[i].paint_uniform_color(gray)
        vis.update_geometry(source[i])
        vis.poll_events()
        vis.update_renderer()

        target[i].paint_uniform_color(gray)
        vis.update_geometry(target[i])
        vis.poll_events()
        vis.update_renderer()

        time.sleep(2)



    vis.run()




if __name__ == "__main__":

    bottle = FracturedObject(name="101620_12_seed_0")
    bottle.load_object(path)
    bottle.create_random_pose()
    bottle.create_random_pose_for_o3d()
    bottle.apply_random_transf_to_o3d()


    animate(bottle.fragments_orig_o3d,bottle.fragments_o3d)










