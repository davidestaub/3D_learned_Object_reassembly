import bpy
import numpy as np
from scipy.spatial.transform import Rotation

from fractured_object import FracturedObject


def install_dependencies():
    import subprocess
    import sys
    import os
    from pathlib import Path

    py_exec = str(sys.executable)
    # Get lib directory
    lib = os.path.join(Path(py_exec).parent.parent, "lib")
    # Ensure pip is installed
    subprocess.call([py_exec, "-m", "ensurepip", "--user"])

    # Update pip (not mandatory)
    subprocess.call([py_exec, "-m", "pip", "install", "--upgrade", "pip"])

    # Install packages
    subprocess.call([py_exec, "-m", "pip", "install", f"--target={str(lib)}", "scipy"])


def load_obj(path):
    bpy.ops.import_scene.obj(filepath=path)


def get_random_rotation() -> np.array:
    theta_x = np.random.uniform(0, 2 * np.pi)
    theta_y = np.random.uniform(0, 2 * np.pi)
    theta_z = np.random.uniform(0, 2 * np.pi)

    r = Rotation.from_euler('xyz', [theta_x, theta_y, theta_z])
    return r.as_matrix()


def animate_rotation(R: np.array):
    r = Rotation.from_matrix(R)
    r_euler = r.as_euler('xyz')

    obj = bpy.data.objects['bottle_0390_shard']
    obj.location = 0, 0, 0
    obj.keyframe_insert(data_path="location", frame=100)


def rotate_obj(R: np.array):
    r = Rotation.from_matrix(R)
    r_euler = r.as_euler('xyz')

    obj = bpy.data.objects['bottle_0390_shard']
    obj.rotation_euler = r_euler


def main():
    path = "/Users/Kasia/PycharmProjects/3D_learned_Object_reassembly/data/bottle_10_seed_1/bottle_10_seed_1_shard.000.obj"

    # load_obj(path)
    R = get_random_rotation()
    animate_rotation(R)


if __name__ == "__main__":
    main()



