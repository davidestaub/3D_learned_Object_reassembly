import numpy as np
from scipy import spatial
import os
from multiprocessing import Pool
import time 
import open3d as o3d
import shutil
from joblib import Parallel, delayed

here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_PATH = os.path.join(here, 'data') 

def compute_SD(neighbourhood, points, normals, p_idx, r):
    p_i = points[p_idx]
    n_p_i = normals[p_idx]
    p_i_bar = np.mean(points[neighbourhood], axis=0)
    v = p_i - p_i_bar
    SD = np.dot(v, n_p_i)
    return SD

# Assembling the above


def get_SD_for_point_cloud(point_cloud, normals, r, threshold):
    n_points = len(point_cloud)
    tree = spatial.KDTree(point_cloud)
    # Compute SD
    SD = np.zeros((n_points))
    neighbourhoods = tree.query_ball_point(point_cloud, r, workers=-1)

    for i in range(n_points):
        neighbourhood = np.asarray(neighbourhoods[i])
        SD[i] = compute_SD(neighbourhood, point_cloud, normals, i, r)
    return SD


def handle_folder(folder):
    log = []

    fragment_path = os.path.join(DATA_PATH, folder, 'cleaned')
    kpts_path = os.path.join(DATA_PATH, folder, 'keypoints')

    if os.path.exists(kpts_path):
        shutil.rmtree(kpts_path)
    os.makedirs(kpts_path)

    #kpts_path_paper = os.path.join(data_path, folder, 'paper_keypoints')
    #kpts_path_iss = os.path.join(data_path, folder, 'iss_keypoints')

    # generate keypoints for each fragment
    for file in os.listdir(fragment_path):
        if file.endswith('.npy'):
            fragment = np.load(os.path.join(fragment_path, file))
            point_cloud = fragment[:, :3]
            normals = fragment[:, 3:]
            
            filename = file.split('cleaned')[0] + "kpts_"+file.split('.')[-2] + ".npy"
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            output = o3d.geometry.keypoint.compute_iss_keypoints(pcd, min_neighbors=5)
            output = [i for i in output.points]
            np.save(os.path.join(kpts_path, filename), output)

            '''
            if len(point_cloud) > 10000:
                r = 0.1
            else:
                r = 0.09
            threshold = 0.011
            # get sd
            SD = get_SD_for_point_cloud(
                point_cloud, normals, r=r, threshold=threshold)

            indices_to_keep = np.argsort(np.abs(SD))[-100:]
            keypoints = point_cloud[indices_to_keep]
            kp_normals = normals[indices_to_keep]
            output = np.concatenate((keypoints, kp_normals), axis=1)
            print("NB Keypoints: ", len(output))
            np.save(os.path.join(kpts_path_paper, filename), output)
            '''

            folder_path = os.path.join(DATA_PATH, folder)
            log_path = os.path.join(folder_path, 'log.txt')
            log.append(f'{filename} : {len(output)}\n')

            with open(log_path, "a") as text_file:
                text_file.write(''.join(log))
            
    print("Done with folder: ", folder)


def main():

    start = time.time()    
    folders = os.listdir(DATA_PATH)

    Parallel(n_jobs=8)(delayed(handle_folder)(folder) for folder in folders)

    print("Duration: ", time.time()- start,"s")

if __name__ == '__main__':
    main()
