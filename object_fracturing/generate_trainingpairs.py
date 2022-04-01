import time
from tools import *
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# standard deviation of random displacement
sigma = 0.001

ROOT = select_folder()
CLEANED = os.path.join(ROOT, 'cleaned\\')
SUBDV = os.path.join(ROOT, 'subdv\\')
here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
print(here)
out_folder = os.path.join(here,'training_data\\')

object_name = ROOT.split('\\')[-1]

n = 0
m = 0
p = 0

files = [ob for ob in os.listdir(SUBDV) if ob.endswith(".npy")]

names1_1vN = []
names2_1vN = []
names_FN = []

rng = np.random.default_rng()

start = time.time()

fragment = []
# load the fragments
for file in files:
    fragment.append(np.load(SUBDV+file))

num_fragments = len(fragment)

for i in range(num_fragments):
    print()
    names2_temp = []
    fragment_set = np.zeros((1, 6))

    for j in range(num_fragments):
        printProgressBar(j+1, num_fragments,prefix="Comparing Parts against fragment"+str(i))
        if i == j:
            continue
        # search for corresponding points in two parts (distance below a treshold)
        matches = np.count_nonzero(cdist(fragment[i][:, :3], fragment[j][:, :3]) < 1e-3)
        #print(f'Fragments {i} and {j} have {matches} matches')
        # if there are more than 100 matches, the parts are considered neighbours
        if matches > 100:
            #print('Its a match!')
            names2_temp.append(f'{object_name}_{j}.npy')
            # all matching parts are concatenated together in one file, as if they where one point cloud
            fragment_set = np.concatenate((fragment_set, fragment[j]), axis=0)

    # delete row of zeros at the beginning
    fragment_set = np.delete(fragment_set, 0, axis=0)

    # only generate a trainingpair, if neighbouring part was found
    if fragment_set.shape[0] != 0:
        # generate list of used parts
        names1_1vN.append(f'{object_name}_{i}.npy')
        names2_1vN.append(names2_temp)

        # add random displacement of points with a gaussian profile
        noise_i = rng.normal(0, sigma, fragment[i].shape)
        noise_set = rng.normal(0, sigma, fragment_set.shape)

        # save the two parts of the pairs as seperate .npy files
        np.save(f'{out_folder}/connected_1vN/fragments_1/{m}.npy', fragment[i] + noise_i)
        np.save(f'{out_folder}/connected_1vN/fragments_2/{m}.npy', fragment_set + noise_set)
        m += 1

# save list of used parts to file
pd.DataFrame(names1_1vN).to_csv(f'{out_folder}/connected_1vN/fragments_1.csv', index=False, header=False)
pd.DataFrame(names2_1vN).to_csv(f'{out_folder}/connected_1vN/fragments_2.csv', index=False, header=False)
#pd.DataFrame(names_FN).to_csv(f'{out_folder}/flipped_normals/fragments.csv', index=False, header=False)

print(f'Elapsed time: {time.time() - start:0.4f}')
