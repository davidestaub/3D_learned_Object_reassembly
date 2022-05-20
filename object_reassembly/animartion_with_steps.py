import os
import open3d
import compas
from tools import *
import numpy as np

from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from fractured_object_open3d import FracturedObject
from compas_vis import compas_show
from utils import get_viewer_data
from compas_view2.app import App
import open3d as o3d
import time
import os
import glob

here = os.path.dirname(os.path.abspath(__file__))
path = "../object_fracturing/data/"


def animate_with_steps(source, target, keep_running,rotations,translation_steps):
    counter = 0

    gray = [0.5, 0.5, 0.5]
    light_gray = [0.9, 0.9, 0.9]
    green = [0.0, 1.0, 0.0]
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    source_mesh = []
    target_mesh = []

    radii = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(0, len(source)):
        #source[i].translate([100, 0, 0])
        rec_mesh_source = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(source[i],
                                                                                          o3d.utility.DoubleVector(
                                                                                              radii))
        rec_mesh_target = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(target[i],
                                                                                          o3d.utility.DoubleVector(
                                                                                              radii))
        source_mesh.append(rec_mesh_source)
        target_mesh.append(rec_mesh_target)

    for i in range(0, len(source)):
        source[i].paint_uniform_color(gray)
        vis.add_geometry(source[i])
        vis.add_geometry(source_mesh[i])

    for i in range(0, len(target)):
        target[i].paint_uniform_color(light_gray)
        vis.add_geometry(target[i])
        vis.add_geometry(target_mesh[i])
    vis.poll_events()
    vis.update_renderer()

    time.sleep(2)
    vis.capture_screen_image("images/im_{}.jpg".format(counter))
    counter += 1

    for i in range(0, len(source)):
        vis.capture_screen_image("images/im_{}.jpg".format(counter))
        counter += 1

        source[i].paint_uniform_color(green)
        vis.update_geometry(source[i])
        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image("images/im_{}.jpg".format(counter))
        counter += 1

        time.sleep(1)


        for j in range(0,100):
            source[i].translate(translation_steps[i])
            vis.update_geometry(source[i])
            vis.poll_events()
            vis.update_renderer()

            vis.capture_screen_image("images/im_{}.jpg".format(counter))
            counter += 1
        vis.capture_screen_image("images/im_{}.jpg".format(counter))
        counter += 1


        source[i].rotate(rotations[i])
        vis.update_geometry(source[i])
        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image("images/im_{}.jpg".format(counter))
        counter += 1

        time.sleep(1)


        source[i].paint_uniform_color(gray)
        vis.update_geometry(source[i])
        vis.poll_events()
        vis.update_renderer()



        #target[i].paint_uniform_color(green)
        #vis.add_geometry(target[i])
        #vis.add_geometry(target_mesh[i])
        #vis.poll_events()
        #vis.update_renderer()

        #vis.capture_screen_image("images/im_{}.jpg".format(counter))
        #counter += 1

        #time.sleep(1)
#
        #source[i].paint_uniform_color(light_gray)
        #vis.update_geometry(source[i])
        #vis.poll_events()
        #vis.update_renderer()

        #vis.capture_screen_image("images/im_{}.jpg".format(counter))
        #counter += 1

        #target[i].paint_uniform_color(gray)
        #vis.update_geometry(target[i])
        #vis.poll_events()
        #vis.update_renderer()
        #time.sleep(1)

        #vis.capture_screen_image("images/im_{}.jpg".format(counter))
        #counter += 1

    if keep_running:
        vis.run()


if __name__ == "__main__":

    if not os.path.exists("images"):
        os.mkdir("images")
    else:
        files = glob.glob('images/*')
        for f in files:
            os.remove(f)

    # Create an object
    bottle = FracturedObject(name="101620_12_seed_0")
    bottle.load_object(path)

    # Create a random transformation (later this will be the input transformation obtained trough superglue - matching)
    bottle.create_random_pose_for_o3d()

    bottle.apply_random_transf_to_o3d()

    rotations = bottle.get_rotation()

    translations_steps = bottle.get_translation_step()



    # Animate the transformation
    keep_running = False
    animate_with_steps(source=bottle.fragments_orig_o3d,target=bottle.fragments_o3d, keep_running=keep_running,rotations=rotations,translation_steps=translations_steps)

    os.system("ffmpeg -f image2 -r 1/0.3 -i ./images/im_%01d.jpg -vcodec mpeg4 -y ./result.mp4")






