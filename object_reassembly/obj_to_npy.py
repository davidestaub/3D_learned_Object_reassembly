import numpy as np
import os
import pywavefront as pw

path = "/Users/Kasia/PycharmProjects/3D_learned_Object_reassembly/object_reassembly/cube_10_seed_0/cleaned/"

if __name__ == "__main__":

    for file in os.listdir(path):
        if file.endswith(".obj"):
            data = pw.Wavefront(path+file)
            np.save(data)