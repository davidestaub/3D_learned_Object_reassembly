from glob import glob
import os
from turtle import window_height, window_width
import open3d as o3d
import numpy as np

# chose a data folder
here = os.path.dirname(os.path.abspath(__file__))
data_list = os.listdir(os.path.join(here, 'data'))


print("id  name")
for idx, val in enumerate(data_list):
    print(idx, " ", val)
idx = int(input("Enter the index of the subfolder in data where the shards are located:\n"))
subfolder = data_list[idx]
ROOT = os.path.join(here, 'data', subfolder)

# get the other folders
CLEANED = os.path.join(ROOT, 'cleaned')
KPTS_IN = os.path.join(ROOT,'processed','keypoints')

kpts_mode = 'SD'

# chose a fragments
data_list = os.listdir(CLEANED)
print("id  name")
for idx, val in enumerate(data_list):
    if val.endswith('.pcd'):
        print(idx, " ", val)
idx = int(input("Enter the index of the subfolder in data where the shards are located:\n"))
file = data_list[idx]
idx = int(file.split('cleaned.')[1].split('.')[0])
print(file, idx)
# extract the corresponding filename
kpts_file_sticky = f'keypoints_sticky.{idx}.npy'
kpts_sticky = np.load(os.path.join(KPTS_IN, kpts_file_sticky))[:,:3]
pcd_kpts = o3d.geometry.PointCloud()
pcd_kpts.points= o3d.utility.Vector3dVector(kpts_sticky)
print(pcd_kpts)
pcd_kpts.paint_uniform_color([1,0,0])


fragment_pcd = o3d.io.read_point_cloud(os.path.join(CLEANED, file))
print(fragment_pcd)
fragment_pcd.scale(0.99, fragment_pcd.get_center())
fragment_pcd.paint_uniform_color([0.1, 0.1, 0.1])

# make same center
fragment_pcd.translate([0,0,0], relative=False)

o3d.visualization.draw_geometries([fragment_pcd, pcd_kpts])
