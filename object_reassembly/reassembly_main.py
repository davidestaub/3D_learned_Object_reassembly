import os

import compas
import numpy as np
import open3d
from compas.geometry import Line
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from compas_vis import compas_show
from fractured_object import FracturedObject
from tools import *
from utils import get_viewer_data

here = os.path.dirname(os.path.abspath(__file__))
path = "data_from_pred/"

if __name__ == "__main__":

    # bottle = FracturedObject(name="bottle_10_seed_1")
    bottle = FracturedObject(name="cube_10_seed_0")
    bottle.load_object(path)
    bottle.load_gt(path)
    # bottle.gt_from_closest()
    bottle.create_random_pose()
    bottle.apply_random_transf()

    visualize = False
    print(bottle.kpt_matches_gt)

    for key in bottle.kpt_matches_gt:
        A = key[0]
        B = key[1]
        matches_AB = bottle.kpt_matches_gt[(A, B)]
        if matches_AB:
            A_indices = matches_AB[0]
            B_indices = matches_AB[1]
            if len(A_indices) != len(B_indices):
                print("unequal lenght")
                exit()
            line_list = []
            for i in range(0, len(A_indices)):
                line = Line(bottle.kpts[A][A_indices[i]], bottle.kpts[B][B_indices[i]])
                line_list.append(line)

            data = {"keypoints": {1: bottle.kpts[A],
                                  # 2: bottle.kpts_orig[A],
                                  # 3: bottle.kpts_orig[B],
                                  4: bottle.kpts[B]
                                  },
                    "fragments": {1: bottle.fragments[A],
                                  # 2: bottle.fragments_orig[A],
                                  # 3: bottle.fragments_orig[B],
                                  4: bottle.fragments[B]
                                  },
                    "lines": line_list}
            if visualize:
                compas_show(data)
            bottle.find_transformations()
            bottle.apply_transf(A, B)

            matches_AB = bottle.kpt_matches_gt[(A, B)]
            A_indices = matches_AB[0]
            B_indices = matches_AB[1]
            if len(A_indices) != len(B_indices):
                print("unequal lenght")
                exit()

            line_list = []
            for i in range(0, len(A_indices)):
                line = Line(bottle.kpts[A][A_indices[i]], bottle.kpts[B][B_indices[i]])
                line_list.append(line)

            data = {"keypoints": {1: bottle.kpts[A],
                                  # 2: bottle.kpts_orig[A],
                                  # 3: bottle.kpts_orig[B],
                                  4: bottle.kpts[B]
                                  },
                    "fragments": {1: bottle.fragments[A],
                                  # 2: bottle.fragments_orig[A],
                                  # 3: bottle.fragments_orig[B],
                                  4: bottle.fragments[B]
                                  },
                    "lines": line_list}
            if visualize:
                compas_show(data)

    # data = {"keypoints": bottle.kpts,
    # "fragments": bottle.fragments}

    # compas_show(data)

    print("Hello")
    print(bottle.transf)
    bottle.create_inverse_transformations_for_existing_pairs()
    bottle.tripplet_matching(1.0, 10.0)

    #
    # data = get_viewer_data(keypoints=list(bottle.kpts.values()), fragments=list(bottle.fragments_meshes.values()))
    # compas_show(data)
