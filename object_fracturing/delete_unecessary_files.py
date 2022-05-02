import os
from joblib import Parallel, delayed
import compas.geometry as cg
here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
dataroot = os.path.join(here, 'data')
dataroot = os.path.abspath('D:\\Studium Daten\\MA2\\3D Vision\\cleaned_cluster\\data_full')
dashed_line = "----------------------------------------------------------------\n"


def handle_folder(object_folder, dataroot):

    # remove shards
    folder_path = os.path.join(dataroot, object_folder)
    shard_list = [name for name in os.listdir(folder_path) if 'shard' in name]
    for shard in shard_list:
        os.remove(os.path.join(folder_path, shard))
    
    # remove npy and obj in cleaned
    folder_path = os.path.join(dataroot, object_folder, 'cleaned')
    if not os.path.exists(folder_path):
        return
    npy_list = [name for name in os.listdir(folder_path) if name.endswith('.npy') or name.endswith('.obj')]
    for file in npy_list:
        os.remove(os.path.join(folder_path, file))
        
    print(f'Processed folder {object_folder}')

Parallel(n_jobs=6)(delayed(handle_folder)(folder, dataroot) for folder in os.listdir(dataroot))
