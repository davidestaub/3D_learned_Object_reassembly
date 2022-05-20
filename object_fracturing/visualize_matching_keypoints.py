from glob import glob
import os
import open3d as o3d
import numpy as np
from scipy.sparse import load_npz
# chose a data folder
folder = "data"
here = os.path.dirname(os.path.abspath(__file__))
data_list = os.listdir(os.path.join(here, folder))


print("id  name")
for idx, val in enumerate(data_list):
    print(idx, " ", val)
idx = int(input("Enter the index of the subfolder in data where the shards are located:\n"))
subfolder = data_list[idx]
ROOT = os.path.join(here, folder, subfolder)

# get the other folders
CLEANED = os.path.join(ROOT, 'cleaned')
KPTS_IN = os.path.join(ROOT,'processed','keypoints')
MATCHINGS = os.path.join(ROOT,'processed','matching')

kpts_mode = 'SD'

# chose a fragments
data_list = [file for file in os.listdir(CLEANED) if file.endswith('pcd')]
print("id  name")
for idx, val in enumerate(data_list):
    if val.endswith('.pcd'):
        print(idx, " ", val)
idx = int(input("Enter the index of the subfolder in data where the shards are located:\n"))
file = data_list[idx]
idx = int(file.split('cleaned.')[1].split('.')[0])
print(file, idx)
# extract the corresponding filename
kpts_file_sticky = f'keypoints_hybrid.{idx}.npy'
kpts_sticky = np.load(os.path.join(KPTS_IN, kpts_file_sticky))[:,:3]

# find the matching fragments
match_gt = glob(os.path.join(MATCHINGS,f'*{idx}*.npz'))
match_matrix = np.load(glob(os.path.join(MATCHINGS,f'*.npy'))[0])
matches = [i for i in range(match_matrix.shape[0]) if match_matrix[idx,i] == 1]
pcd_matches = []
for i in matches:
    kpts = np.load(os.path.join(KPTS_IN, f'keypoints_hybrid.{i}.npy'))[:,:3]
    pcd_matches.append(kpts)

# load the masks for ground trugh
mask_paths = []
for match in matches:
    files = []
    files.append(glob(os.path.join(MATCHINGS, f'*{idx}_{match}*.npz')))
    files.append(glob(os.path.join(MATCHINGS, f'*{match}_{idx}*.npz')))
    mask_paths.append([i[0] for i in files if i])

close_kps = []
for j in range(len(mask_paths)):
    mask = np.array(load_npz(mask_paths[j][0]).toarray(), dtype=np.int)
    kpts = np.array(pcd_matches[j])
    close_kps.append(kpts[mask])

pcd_kpts = o3d.geometry.PointCloud()
pcd_kpts.points= o3d.utility.Vector3dVector(kpts_sticky)
print(pcd_kpts)
pcd_kpts.paint_uniform_color([1,0,0])


fragment_pcd = o3d.io.read_point_cloud(os.path.join(CLEANED, file))
print(fragment_pcd)
fragment_pcd.scale(0.99, fragment_pcd.get_center())
fragment_pcd.paint_uniform_color([0.1, 0.1, 0.1])
sphere = o3d.geometry.TriangleMesh.create_sphere(2e-2)
sphere.paint_uniform_color([1,0,0])

# make same center
fragment_pcd.translate([0,0,0], relative=False)

o3d.visualization.draw_geometries([fragment_pcd, pcd_kpts, sphere])
