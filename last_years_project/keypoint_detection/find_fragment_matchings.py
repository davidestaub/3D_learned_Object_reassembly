import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from multiprocessing import Pool
VISUALIZE = False
# Root data folder. (such that DATA_FOLDER/pointclouds/{name} contains pointcloud npys.
DATA_FOLDER = '../data'


def find_matching_matrix(name):
    print(f'\nProcessing {name}...')
    fragments = []

    # Load the pointcloud of each fragment
    num_parts = 0
    while True:
        path = f'{DATA_FOLDER}/pointclouds/{name}/{name}_{num_parts}.npy'
        try:
            fragments.append(np.load(path))
        except:
            print(f'Couldn\'t find {path}')
            print(f'{num_parts} parts loaded.\n')
            break

        # normalizing  roughly to [-1,1]
        # if name[0:3] == 'cat':
        #     fragment[part][:, :3] /= 450

        print(fragments[num_parts].shape)
        if num_parts == 999:
            print('WARNING: part limit reached, only loaded the first 1000 parts.')
        num_parts += 1

    # Scatter plot of pointcloud.
    if VISUALIZE:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(num_parts):
            ax.scatter(fragments[i][:, 0], fragments[i][:, 2], fragments[i][:, 1], s=1, label=i)
        legend = plt.legend(bbox_to_anchor=(0, 1), loc="upper left", bbox_transform=fig.transFigure)
        for handle in legend.legendHandles:
            handle.set_sizes([50.0])
        plt.show()
        plt.close(fig)

    matching_matrix = np.zeros((num_parts, num_parts))
    for i in range(num_parts):
        for j in range(i):
            # search for corresponding points in two parts (distance below a treshold)
            matches = np.sum(cdist(fragments[i][:, :3], fragments[j][:, :3]) < 1e-3)
            # print(f'Fragments {i} and {j} have {matches} matches')

            # if there are more than 100 matches, the parts are considered neighbours
            if matches > 100:
                print(f'{name}: {i} and {j} match!')
                matching_matrix[i, j] = matching_matrix[j, i] = 1

    print(matching_matrix)
    print(f'saving to {DATA_FOLDER}/fragment_matchings/{name}.npy')
    np.save(f'{DATA_FOLDER}/fragment_matchings/{name}.npy', matching_matrix)


def main():
    names = []
    # uncomment for whole trainset
    # for i in range(1,6):
    #     if i == 2 or i==3: continue
    #     names.append(f'cat_seed_{i}')
    for i in range(1, 15):
        names.append(f'cube_20_seed_{i}')
    # for i in range(2, 10):
    #     names.append(f'cylinder_20_seed_{i}')
    # uncomment for whole testset
    # names=['cat_seed_0', 'cube_20', 'cube_20_seed_1', 'cube_20_seed_2', 'cube_20_seed_3', 'cylinder_20_seed_1']
    # names = ['cube_20_seed_4']
    # names = ['cube_6']

    with Pool(4) as p:
        p.map(find_matching_matrix, names)


if __name__ == '__main__':
    main()
