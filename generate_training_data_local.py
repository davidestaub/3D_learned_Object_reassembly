import argparse
import os
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser("generate_iss_keypoints_and_shot_descriptors")

    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    args.data_dir = os.path.join('.', 'object_fracturing', 'data')

    print(f'Data dir: {args.data_dir}')
    # set to the part which should be generatred
    object_folders = os.listdir(args.data_dir)

    for folder in object_folders:
        folder = os.path.join(args.data_dir, folder)
        subprocess.check_call(
            [
                "python",
                "process_folder_cluster.py",
                "--path",
                folder,
                "--descriptor_method",
                "fpfh",
                "--keypoint_method",
                "hybrid"
            ]
        )
