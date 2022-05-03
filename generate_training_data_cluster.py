import argparse
import os
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generates keypoints and descriptors")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    args.data_dir = os.path.abspath('/cluster/project/infk/courses/252-0579-00L/group19/blender_fracture_modifier/script-output')

    print(f'Data dir: {args.data_dir}')
    # set to the part which should be generatred
    object_folders = os.listdir(args.data_dir)[:1]

    for folder in object_folders:
        folder = os.path.join(args.data_dir, folder)
        subprocess.run(
            [
                "bsub",
                "-R",
                "rusage[mem=2000]",
                "python",
                "process_folder_cluster.py",
                "--path",
                folder,
            ]
        )
