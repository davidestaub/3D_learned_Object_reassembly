import os
from typing import List

import numpy as np
from scipy.spatial.distance import cdist


def get_fragment_matchings(fragments: List[np.array], folder_path: str):
    object_name = os.path.basename(folder_path)
    match_path = os.path.join(folder_path, 'processed', 'matching')
    os.makedirs(match_path, exist_ok=True)

    matching_matrix_path = os.path.join(
        match_path, f'{object_name}_matching_matrix.npy')

    # If matching is calculated already, use it.
    if os.path.exists(matching_matrix_path):
        matching_matrix = np.load(matching_matrix_path)
        return matching_matrix

    # Otherwise compute and save matchings.
    num_parts = len(fragments)
    matching_matrix = np.zeros((num_parts, num_parts))
    for i in range(num_parts):
        for j in range(i):
            # Search for corresponding points in two parts (distance below a treshold).
            matches = np.sum(cdist(fragments[i][:, :3], fragments[j][:, :3]) < 1e-3)
            # If there are more than 100 matches, the parts are considered neighbours.
            if matches > 600:
                print(f"Matched fragment {i} and {j}!")
                matching_matrix[i, j] = matching_matrix[j, i] = 1

    np.save(matching_matrix_path, matching_matrix)
    return matching_matrix


def get_keypoint_assignment(keypoints1, keypoints2, threshold=3e-2):
    dists = cdist(keypoints1, keypoints2)
    close_enough_mask = np.min(dists, axis=0) < threshold
    closest = np.argmin(dists, axis=0)

    keypoint_assignment = np.zeros((keypoints1.shape[0], keypoints2.shape[0]))
    keypoint_assignment[closest[close_enough_mask], close_enough_mask] = 1

    return keypoint_assignment
