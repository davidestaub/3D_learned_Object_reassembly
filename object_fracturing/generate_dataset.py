import sys
import inspect
import os
import subprocess
import argparse 

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


def main():
    parser = argparse.ArgumentParser(description="Generates an example cube dataset.")
    parser.add_argument("--num_cubes", type=str, help='Number of cubes to generate')
    parser.add_argument("--num_frags", type=str, help='Number of fragments to generate')
    args = parser.parse_args()
    num_cubes = args.num_cubes
    count = args.num_frags

    subprocess.run(
    [
        os.path.join(currentdir, 'blender_fracture_modifier', 'blender'),
        "--background",
        "--python",
        os.path.join(currentdir, 'blender_scripts', 'generate_cubes.py'),
        "--",
        str(num_cubes),
        str(count)
    ]
    )


if __name__ == '__main__':
    main()
