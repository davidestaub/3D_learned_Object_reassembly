import glob

import matplotlib.pyplot as plt
import numpy as np

DATA_FOLDER = '../datasets'
VISUALIZE = True
# Put the contents of /252-0579-00L 3DVision_project/keypoints/keypoints_final from
# https://polybox.ethz.ch/index.php/s/AkdL2sbNfKPhxwa
# in ../datasets/keypoints
# NAMES = list(map(lambda i: f'cube_20_seed_{i}', range(1, 15)))
NAMES = ['cube_20_seed_3']
THRESHOLD = 0.05


def pairwise_distances(a, b):
    N = a.shape[0]
    M = b.shape[0]
    a = np.repeat(a, M, axis=0)
    b = np.tile(b, (N, 1))
    dist = np.sqrt(np.sum((a - b) ** 2, axis=-1))
    dist = dist.reshape(N, M)
    return dist


def main():
    for name in NAMES:
        # 1vN had highest scores according to last year's report.
        keypoint_folder_glob = f'{DATA_FOLDER}/keypoints/{name}_1vN/*.npy'
        num_parts = len(glob.glob(keypoint_folder_glob))
        fragments = []
        for i in range(num_parts):
            f = np.load(f'{DATA_FOLDER}/keypoints/{name}_1vN/{i}.npy')
            fragments.append(f)
        print(f'Loaded {len(fragments)} fragments.')

        matching_matrix_path = f'{DATA_FOLDER}/fragment_matchings/{name}.npy'
        matching_matrix = np.load(matching_matrix_path)
        assert num_parts == matching_matrix.shape[0] == matching_matrix.shape[
            1], f"something is wrong: {num_parts} parts loaded vs matrich matrix for {matching_matrix.shape[0]} parts."
        for (a_idx, b_idx), is_match in np.ndenumerate(matching_matrix):
            if is_match:
                # Fourth coord is some other parameters outputted by the network (certainty?)
                a = fragments[a_idx][:, :3]
                b = fragments[b_idx][:, :3]
                # Calculate the distances.
                distance_matrix = pairwise_distances(a, b)  # Euclidean distance.
                print(distance_matrix)
                print(f'Number of close points: {(distance_matrix < THRESHOLD).sum()}')
                print(f'Closest: {distance_matrix.min()}')
                # Visualize.
                if VISUALIZE:
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    ax.scatter(a[:, 0], a[:, 2], a[:, 1], s=1, label=a_idx)
                    ax.scatter(b[:, 0], b[:, 2], b[:, 1], s=1, label=b_idx)
                    for (i, j), dist in np.ndenumerate(distance_matrix):
                        if dist < THRESHOLD:
                            p1 = a[i]
                            p2 = b[j]
                            ax.plot3D([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p1[1]])

                    legend = plt.legend(bbox_to_anchor=(0, 1), loc="upper left", bbox_transform=fig.transFigure)
                    for handle in legend.legendHandles:
                        handle.set_sizes([50.0])
                    plt.show()
                    plt.close(fig)


if __name__ == '__main__':
    main()
