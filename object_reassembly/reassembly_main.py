import os
import open3d
import compas
from tools import *
import numpy as np

from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from fractured_object import FracturedObject
from compas_vis import compas_show
from utils import get_viewer_data

here = os.path.dirname(os.path.abspath(__file__))
path = "../data/"

if __name__ == "__main__":

    bottle = FracturedObject(name="bottle_10_seed_1")
    bottle.load_object(path)
    bottle.load_gt(path)
    # bottle.gt_from_closest()
    bottle.create_random_pose()
    bottle.apply_random_transf()

    data = {"keypoints": bottle.kpts,
            "fragments": bottle.fragments}

    # compas_show(data)

    bottle.matching()

    # for key in bottle.transf.keys():
    #     bottle.apply_transf(key[0], key[1])

    A = 10
    b = [1, 6, 11]

    for B in b:

        bottle.apply_transf(A, B)

        # for A in range(12):
        #
        data = {"keypoints": {1: bottle.kpts[A],
                              # 2: bottle.kpts_orig[A],
                              # 3: bottle.kpts_orig[B],
                              4: bottle.kpts[B]
                              },
                "fragments": {1: bottle.fragments[A],
                              # 2: bottle.fragments_orig[A],
                              # 3: bottle.fragments_orig[B],
                              4: bottle.fragments[B]
                              }}
        compas_show(data)

    data = {"keypoints": bottle.kpts,
            "fragments": bottle.fragments}

    compas_show(data)

    print("Hello")

    #
    # data = get_viewer_data(keypoints=list(bottle.kpts.values()), fragments=list(bottle.fragments_meshes.values()))
    # compas_show(data)
