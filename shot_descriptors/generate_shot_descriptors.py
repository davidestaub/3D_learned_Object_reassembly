import argparse

import numpy as np

import pyshot


# keypoints.shape: [NUM_KEYPOINTS, 6], 0:3 is location, [3:] is normals
def generate_shot_descriptors(keypoint_array, args):
    vertices = keypoint_array[:, :3]
    normals = keypoint_array[:, 3:]

    local_rf_radius = args.radius if args.local_rf_radius is None else args.local_rf_radius
    descriptors = pyshot.get_descriptors(vertices, normals,
                                         radius=args.radius,
                                         local_rf_radius=local_rf_radius,
                                         min_neighbors=args.min_neighbors,
                                         n_bins=args.n_bins,
                                         double_volumes_sectors=args.double_volumes_sectors,
                                         use_interpolation=args.use_interpolation,
                                         use_normalization=args.use_normalization,
                                         )

    print(descriptors)
    return descriptors


def main():
    parser = argparse.ArgumentParser("example_pyshot", description=__doc__)
    parser.add_argument("--radius", type=float, default=100)
    parser.add_argument("--local_rf_radius", type=float, default=None)
    parser.add_argument("--min_neighbors", type=int, default=4)
    parser.add_argument("--n_bins", type=int, default=20)
    parser.add_argument("--double_volumes_sectors", action='store_true')
    parser.add_argument("--use_interpolation", action='store_true')
    parser.add_argument("--use_normalization", action='store_true')
    args = parser.parse_args()

    arr = np.random.random((10, 6))
    generate_shot_descriptors(arr, args)


if __name__ == '__main__':
    main()
