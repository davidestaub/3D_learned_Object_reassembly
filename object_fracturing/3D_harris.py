from multiprocessing import Pool, cpu_count
import transformation
import neighborhoords
import shutil
import itertools
import numpy as np
from numpy.linalg import lstsq
from sklearn.decomposition import PCA
from math import sqrt
import os
import warnings
warnings.filterwarnings("ignore")

here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_PATH = os.path.join(here, 'data')


def dot_product(vector_1, vector_2):
    return sum((a*b) for a, b in zip(vector_1, vector_2))


def length(vector):
    return sqrt(dot_product(vector, vector))


def polyfit3d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    m, _, _, _ = lstsq(G, z)
    return m


def find_keypoints(fragment_path, file, kpts_path):
    # parameters
    n_neighbours = 3
    delta = 0.025
    k = 0.04
    fraction = 0.1

    # Load point cloud
    fragment = np.load(os.path.join(fragment_path, file))
    points = fragment[:, :3]

    # subsample for big pointclouds
    if len(points) > 5000:
        samp_idx = np.random.choice(len(points), 5000, replace=False)
        points = points[samp_idx]

    # initialisation of the solution
    labels_fraction = np.zeros(len(points))
    resp = np.zeros(len(points))

    # compute neighborhood
    #neighborhood = neighborhoords.k_ring_delaunay(points, n_neighbours)
    neighborhood = neighborhoords.k_ring_delaunay_adaptive(points, delta)

    points_centred, _ = transformation.centering_centroid(points)

    # best fitting point, this was calculated in the loop before, maybe that's not necessary?!
    pca = PCA(n_components=3)  # Principal Component Analysis
    points_pca = pca.fit_transform(np.transpose(points_centred))
    eigenvalues, eigenvectors = np.linalg.eigh(points_pca)

     # rotate the cloud
    for i in range(points.shape[0]):
        points[i, :] = np.dot(np.transpose(eigenvectors), points[i, :])

    for i in neighborhood.keys():

        # restrict to XY plane and translate
        points_2D = points[:, :2]-points[i, :2]

        # fit a quadratic surface
        m = polyfit3d(points_2D[:, 0], points_2D[:, 1], points[:, 2], order=2)
        m = m.reshape((3, 3))

        # Compute the derivative
        fx2  = m[1, 0]*m[1, 0] + 2*m[2, 0]*m[2, 0] + 2*m[1, 1]*m[1, 1]  # A
        fy2  = m[1, 0]*m[1, 0] + 2*m[1, 1]*m[1, 1] + 2*m[0, 2]*m[0, 2]  # B
        fxfy = m[1, 0]*m[0, 1] + 2*m[2, 0]*m[1, 1] + 2*m[1, 1]*m[0, 2]  # C

        # Compute response
        resp[i] = fx2*fy2 - fxfy*fxfy - k*(fx2 + fy2)*(fx2 + fy2)
    
    # rotate back
    for i in range(points.shape[0]):
        points[i, :] = np.dot(eigenvectors, points[i, :])

    # Select interest points
    # search for local maxima
    candidate = [[i, resp[i]] for i in neighborhood.keys() if resp[i] >= np.max(resp[neighborhood[i]])]

    # sort by decreasing order
    candidate = np.array(candidate.sort(reverse=True, key=lambda x: x[1]))

    # Method 1 : fraction
    interest_points = np.array(candidate[:int(fraction*len(points)), 0], dtype=np.int)
    labels_fraction[interest_points] = 1

    # Save the result
    filename = file.split('cleaned')[0] + "kpts_"+file.split('.')[-2] + ".npy"
    labels = labels_fraction.astype('bool')
    output = points[labels]
    np.save(os.path.join(kpts_path, filename), output)

    log = f'{filename} : {len(output)}\n'
    return log


def handle_folder(folder):
    # initiate the paths and a log
    log = []
    fragment_path = os.path.join(DATA_PATH, folder, 'cleaned')
    kpts_path = os.path.join(DATA_PATH, folder, 'keypoints_harris')

    # clean keypoints
    if os.path.exists(kpts_path):
        shutil.rmtree(kpts_path)
    os.makedirs(kpts_path)

    # generate keypoints for each fragment
    for file in os.listdir(fragment_path):
        if file.endswith('.npy'):
            result = find_keypoints(fragment_path, file, kpts_path)
            log.append(result)

    folder_path = os.path.join(DATA_PATH, folder)
    log_path = os.path.join(folder_path, 'log.txt')
    with open(log_path, "a") as text_file:
        text_file.write(''.join(log))

    print("Done with folder: ", folder)


def main():
    folders = os.listdir(DATA_PATH)
    with Pool(cpu_count()) as p:
        p.map(handle_folder, folders)

if __name__ == '__main__':
    main()
