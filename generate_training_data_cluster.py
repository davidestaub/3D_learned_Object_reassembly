import argparse
import os
import subprocess
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generates keypoints and descriptors")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    args.data_dir = os.path.abspath('/cluster/project/infk/courses/252-0579-00L/group19/3D_learned_Object_reassembly/object_fracturing/data')

    print(f'Data dir: {args.data_dir}')
    # set to the part which should be generatred
    object_folders = os.listdir(args.data_dir)
    for folder in object_folders:
        folder = os.path.join(args.data_dir, folder)
        subprocess.run(
            [
                "bsub",
                "-n",
                "4",
                "-R",
                "rusage[mem=8G]",
                "python",
                "process_folder_cluster.py",
                "--path",
                folder,
            ]
        )
