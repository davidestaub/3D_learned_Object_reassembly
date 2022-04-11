from joblib import Parallel, delayed
from tools import *
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


# standard deviation of random displacement
sigma = 0.001

# get root path
here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
out_folder = os.path.join(here, 'training_data\\')
data_folder = os.path.join(here, 'data\\')

# mode (connected_1vN or connected_1v1)
mode = 'connected_1vN'

names1_1vN = []
names2_1vN = []


def handle_folder(folder):
    if folder == 'keypoints':
        return
    global index
    global names1_1vN
    global names2_1vN

    # change to the subdivided folder
    folder = os.path.join(folder, 'subdv\\')

    files = [ob for ob in os.listdir(folder) if ob.endswith(".npy")]

    rng = np.random.default_rng()
    fragment = []
    fragment_names = []
    # load the fragments
    for file in files:
        fragment.append(np.load(folder+file))
        fragment_names.append(file)

    num_fragments = len(fragment)

    for i, name in enumerate(fragment_names):
        print()
        names2_temp = []
        fragment_set = np.zeros((1, 6))

        for j in range(num_fragments):
            printProgressBar(j+1, num_fragments,
                             prefix="Comparing Parts against fragment"+str(i))
            if i == j:
                continue
            # search for corresponding points in two parts (distance below a treshold)
            matches = np.count_nonzero(
                cdist(fragment[i][:, :3], fragment[j][:, :3]) < 1e-3)
            #print(f'Fragments {i} and {j} have {matches} matches')
            # if there are more than 100 matches, the parts are considered neighbours
            if matches > 100:
                #print('Its a match!')
                names2_temp.append(fragment_names[j])
                # all matching parts are concatenated together in one file, as if they where one point cloud
                fragment_set = np.concatenate(
                    (fragment_set, fragment[j]), axis=0)

        # delete row of zeros at the beginning
        fragment_set = np.delete(fragment_set, 0, axis=0)

        # only generate a trainingpair, if neighbouring part was found
        if fragment_set.shape[0] != 0:
            # generate list of used parts
            names1_1vN.append(fragment_names[i])
            names2_1vN.append(names2_temp)

            # add random displacement of points with a gaussian profile
            noise_i = rng.normal(0, sigma, fragment[i].shape)
            noise_set = rng.normal(0, sigma, fragment_set.shape)

            # save the two parts of the pairs as seperate .npy files
            prefix = name.split('_subdv')[0]
            number = name.split('.')[-2]
            name_1 = '_'.join([prefix, 'part', number])
            name_2 = '_'.join([prefix, 'counterpart', number])
            np.save(f'{out_folder}/{mode}/fragments_1/{name_1}.npy',
                    fragment[i] + noise_i)
            np.save(f'{out_folder}/{mode}/fragments_2/{name_2}.npy',
                    fragment_set + noise_set)


Parallel(n_jobs=8)(delayed(handle_folder)(os.path.join(data_folder, object))
                   for object in os.listdir(data_folder))

# save list of used parts to file
pd.DataFrame(names1_1vN).to_csv(
    f'{out_folder}/{mode}/fragments_1.csv', index=False, header=False)
pd.DataFrame(names2_1vN).to_csv(
    f'{out_folder}/{mode}/fragments_2.csv', index=False, header=False)
