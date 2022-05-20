import os

from compas.geometry import Line

from compas_vis import compas_show
from fractured_object import FracturedObject

here = os.path.dirname(os.path.abspath(__file__))
path = "data_from_pred/"

visualize = True
vis_idx = (3, 4)


def full_reassembly(obj):
    obj.create_random_pose()
    obj.apply_random_transf()
    compas_show(obj.kpts, obj.fragments)
    obj.find_transformations()
    obj.create_inverse_transformations_for_existing_pairs()
    obj.tripplet_matching(1.0, 10.0)
    obj.find_final_transforms()

    if visualize:
        compas_show(obj.kpts, obj.fragments)


def pairwise_reassembly(obj):
    obj.create_random_pose()
    obj.apply_random_transf()
    print(obj.kpt_matches_gt)
    for key in obj.kpt_matches_gt:
        A = key[0]
        B = key[1]
        matches_AB = obj.kpt_matches_gt[key]
        if matches_AB:
            A_indices = matches_AB[0]
            B_indices = matches_AB[1]
            assert len(A_indices) == len(B_indices), "unequal length"
            line_list = []
            for i in range(0, len(A_indices)):
                line = Line(obj.kpts[A][A_indices[i]], obj.kpts[B][B_indices[i]])
                line_list.append(line)

            keypoints = {
                A: obj.kpts[A],
                B: obj.kpts[B]
            }
            fragments = {
                A: obj.fragments[A],
                B: obj.fragments[B]
            }
            if visualize or (A, B) == vis_idx or (B, A) == vis_idx:
                compas_show(keypoints, fragments, line_list)
            obj.find_transformations()
            obj.apply_transf(A, B)

            line_list = []
            for i in range(0, len(A_indices)):
                line = Line(obj.kpts[A][A_indices[i]], obj.kpts[B][B_indices[i]])
                line_list.append(line)

            keypoints = {
                A: obj.kpts[A],
                B: obj.kpts[B]
            }
            fragments = {
                A: obj.fragments[A],
                B: obj.fragments[B]
            }
            if visualize or (A, B) == vis_idx or (B, A) == vis_idx:
                compas_show(keypoints, fragments, line_list)

    print("Hello")
    print(obj.transf)


if __name__ == "__main__":
    obj = FracturedObject(name="cube_10_seed_0")
    obj.load_object(path)
    obj.load_gt(path)
    full_reassembly(obj)
    # pairwise_reassembly(obj)
