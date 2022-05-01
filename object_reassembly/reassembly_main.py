import os
import open3d
import compas
from tools import *
import numpy as np

from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from fractured_object import FracturedObject
from compas_vis import show

here = os.path.dirname(os.path.abspath(__file__))
path = "../data/"


def get_keypoint_assignment(keypoints1, keypoints2, threshold=0.001):
    dists = cdist(keypoints1, keypoints2)
    close_enough_mask = np.min(dists, axis=0) < threshold
    closest = np.argmin(dists, axis=0)

    keypoint_assignment = np.zeros((keypoints1.shape[0], keypoints2.shape[0]))
    keypoint_assignment[closest[close_enough_mask], close_enough_mask] = 1

    return keypoint_assignment

def get_viewer_data(fragments = None, keypoints = None):
    data = {}
    if fragments:
        data["fragments"] = fragments

    if keypoints:
        data["keypoints"] = keypoints
    return data


def main():
    bottle = FracturedObject(name="bottle_10_seed_1")
    bottle.load_object(path)
    bottle.load_gt(path)

    bottle.create_random_pose()
    bottle.apply_random_transf()

    data = get_viewer_data(keypoints=list(bottle.kpts.values()), fragments=list(bottle.fragments_meshes.values()))
    show(data)

    matches = get_keypoint_assignment(bottle.keypoints[0], bottle.keypoints[5])

    dummy_matches0 = np.where(matches == 1)[0]
    dummy_matches1 = np.where(matches == 1)[1]

    bottle.load_kpt_matches(dummy_matches0, dummy_matches1, 0, 5)
    bottle.find_transformations_first3kpts()

    # matched = bool(0)
    # for i in range(10):
    #     matched = 0
    #     ii = i
    #     while not matched:
    #         if bottle.fragment_matches[i][ii]:
    #             obj = viewer.add(bottle.fragments[i])
    #             R,t,T = bottle.transformations[(i, ii)]
    #             m = 0
    #             @viewer.on(interval=1000, frames=2)
    #             def move(f=100):
    #                 # obj.rotation = R
    #                 obj.translation = [f, 0, 0]
    #                 obj.update()
    #                 viewer.show()
    #             # Mesh.transform(bottle.fragments[i], T)
    #             # obj = viewer.add(bottle.fragments[i])
    #             matched = 1
    #         ii += 1
    #         if ii >= len(bottle.fragments):
    #             break

    # viewer.run()


if __name__ == "__main__":
    main()
