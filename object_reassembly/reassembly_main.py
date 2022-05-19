import os

from compas.geometry import Line

from compas_vis import compas_show
from fractured_object import FracturedObject

here = os.path.dirname(os.path.abspath(__file__))
path = "data_from_pred/"

visualize = False
vis_idx = (3, 4)
visualize_only_solution = True


def full_reassembly(fractured_object):
    fractured_object.create_random_pose()
    fractured_object.apply_random_transf()
    fractured_object.find_transformations()
    fractured_object.create_inverse_transformations_for_existing_pairs()
    fractured_object.tripplet_matching(1.0, 10.0)
    fractured_object.find_final_transforms()

    if visualize or visualize_only_solution:
        keypoints = {}
        fragments = {}
        for idx in range(len(fractured_object.fragments)):
            keypoints[idx] = fractured_object.kpts[idx]
            fragments[idx] = fractured_object.fragments[idx]
        compas_show({"keypoints": keypoints, "fragments": fragments})


def pairwise_reassembly(fractured_object):
    bottle.create_random_pose()
    bottle.apply_random_transf()
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
            if visualize or (A, B) == vis_idx or (B, A) == vis_idx:
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
            if visualize or (A, B) == vis_idx or (B, A) == vis_idx:
                compas_show(data)

    print("Hello")
    print(bottle.transf)


if __name__ == "__main__":
    # bottle = FracturedObject(name="bottle_10_seed_1")
    bottle = FracturedObject(name="cube_10_seed_3")
    bottle.load_object(path)
    bottle.load_gt(path)
    # bottle.gt_from_closest()
    full_reassembly(bottle)
    # pairwise_reassembly(bottle)




    #
    # data = get_viewer_data(keypoints=list(bottle.kpts.values()), fragments=list(bottle.fragments_meshes.values()))
    # compas_show(data)
