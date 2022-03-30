import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# standard deviation of random displacement
sigma = 0.001
# folder, where the pointcloud data is
data_folder = './data'
# folder, where to save the training data
out_folder = './training_data'

n = 0
m = 0
p = 0

names = []
# uncomment for whole trainset
# for i in range(1,6):
#     if i == 2 or i==3: continue
#     names.append(f'cat_seed_{i}')
# for i in range (4,15):
#     names.append(f'cube_20_seed_{i}')
for i in range(2, 10):
    names.append(f'cylinder_20_seed_{i}')
# uncomment for whole testset
# names=['cat_seed_0', 'cube_20', 'cube_20_seed_1', 'cube_20_seed_2', 'cube_20_seed_3', 'cylinder_20_seed_1']
# names=['cube_20_seed_4']

names1_1v1 = []
names2_1v1 = []
names1_1vN = []
names2_1vN = []
names_FN = []
rng = np.random.default_rng()

start = time.time()

for name in names:
    print(f'\nProcessing {name}...')
    fragment = []
    keypoints = []

    # Load keypoints and pointcloud of each fragment
    for part in range(1000):
        try:
            fragment.append(np.load(f'{data_folder}/{name}/{name}_{part}.npy'))
        except:
            print(f'{part} parts loaded.\n')
            break

        # normalizing  roughly to [-1,1]
        if name[0:3] == 'cat':
            fragment[part][:, :3] /= 450

        print(fragment[part].shape)
        if part == 999:
            print('WARNING: part limit reached, only loaded the first 1000 parts.')
            part += 1

    # scatter plot of pointcloud and keypoints
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # for i in range(part):
    #     ax.scatter(fragment[i][:,0], fragment[i][:,2], fragment[i][:,1], color=cm.Set1(i/part) , s=1)#, fragment[:,3], fragment[:,4], fragment[:,5])
    # plt.show()

    # save fragments with connected surfaces 1vN
    print('\nSaving connected 1vN\n')
    for i in range(part):
        names2_temp = []
        fragment_set = np.zeros((1, 6))
        for j in range(part):
            if i == j: continue

            # search for corresponding points in two parts (distance below a treshold)
            matches = np.count_nonzero(cdist(fragment[i][:, :3], fragment[j][:, :3]) < 1e-3)
            print(f'Fragments {i} and {j} have {matches} matches')

            # if there are more than 100 matches, the parts are considered neighbours
            if matches > 100:
                print('Its a match!')
                names2_temp.append(f'{name}/{name}_{j}.npy')
                # all matching parts are concatenated together in one file, as if they where one point cloud
                fragment_set = np.concatenate((fragment_set, fragment[j]), axis=0)

        # delete row of zeros at the beginning
        fragment_set = np.delete(fragment_set, 0, axis=0)

        # only generate a trainingpair, if neighbouring part was found
        if fragment_set.shape[0] != 0:
            # generate list of used parts
            names1_1vN.append(f'{name}/{name}_{i}.npy')
            names2_1vN.append(names2_temp)

            # add random displacement of points with a gaussian profile
            noise_i = rng.normal(0, sigma, fragment[i].shape)
            noise_set = rng.normal(0, sigma, fragment_set.shape)

            # save the two parts of the pairs as seperate .npy files
            np.save(f'{out_folder}/connected_1vN/fragments_1/{m}.npy', fragment[i] + noise_i)
            np.save(f'{out_folder}/connected_1vN/fragments_2/{m}.npy', fragment_set + noise_set)
            m += 1

    # save pairs with flipped surface normals
    print('\nSaving flipped normals\n')
    for i in range(part):
        # copy fragment, change the sign of the surface normals
        fragment_neg = np.copy(fragment[i])
        fragment_neg[:, 3:6] = -fragment_neg[:, 3:6]

        # generate list of used parts
        names_FN.append(f'{name}/{name}_{i}.npy')

        # add random displacement of points with a gaussian profile
        noise_i = rng.normal(0, sigma, fragment[i].shape)
        noise_neg = rng.normal(0, sigma, fragment_neg.shape)

        # save the two parts of the pairs as seperate .npy files
        np.save(f'{out_folder}/flipped_normals/fragments_1/{p}.npy', fragment[i] + noise_i)
        np.save(f'{out_folder}/flipped_normals/fragments_2/{p}.npy', fragment_neg + noise_neg)
        p += 1

    # save fragments with connected surfaces 1v1
    # print('\nSaving connected 1v1\n')
    # for i in range(part):
    #     for j in range(i+1,part):
    #         # search for corresponding points in two parts (distance below a treshold)
    #         matches = np.count_nonzero(cdist(fragment[i][:,:3],fragment[j][:,:3])<1e-3)
    #         print(f'Fragments {i} and {j} have {matches} matches')

    #         # if there are more than 100 matches, the parts are considered neighbours
    #         if  matches > 100:
    #             print('Its a match!')

    #             # generate list of used parts
    #             names1_1v1.append(f'{name}/{name}_{i}.npy')
    #             names2_1v1.append(f'{name}/{name}_{j}.npy')

    #             # add random displacement of points with a gaussian profile
    #             noise_i=rng.normal(0,sigma,fragment[i].shape)
    #             noise_j=rng.normal(0,sigma,fragment[j].shape)

    #             # save the two parts of the pairs as seperate .npy files
    #             np.save(f'{out_folder}/connected_1v1/fragments_1/{n}.npy', fragment[i]+noise_i)
    #             np.save(f'{out_folder}/connected_1v1/fragments_2/{n}.npy', fragment[j]+noise_j)
    #             n += 1

# save list of used parts to file
# pd.DataFrame(names1_1v1).to_csv(f'{out_folder}/connected_1v1/fragments_1.csv', index=False, header=False)
# pd.DataFrame(names2_1v1).to_csv(f'{out_folder}/connected_1v1/fragments_2.csv', index=False, header=False)
pd.DataFrame(names1_1vN).to_csv(f'{out_folder}/connected_1vN/fragments_1.csv', index=False, header=False)
pd.DataFrame(names2_1vN).to_csv(f'{out_folder}/connected_1vN/fragments_2.csv', index=False, header=False)
pd.DataFrame(names_FN).to_csv(f'{out_folder}/flipped_normals/fragments.csv', index=False, header=False)

print(f'Elapsed time: {time.time() - start:0.4f}')
