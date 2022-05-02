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
    bottle.gt_from_closest()

    bottle.matching()

    # bottle.create_random_pose()
    # bottle.apply_random_transf()
    #
    # data = get_viewer_data(keypoints=list(bottle.kpts.values()), fragments=list(bottle.fragments_meshes.values()))
    # compas_show(data)


