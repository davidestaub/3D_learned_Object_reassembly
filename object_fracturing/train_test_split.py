import os
from compas.datastructures import Mesh
from tools import *
from compas.geometry import Pointcloud
import numpy as np
from compas_view2.app import App

# ==============================================================================
# File
# ==============================================================================

here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
train_folder = os.path.join(here,'training_data','train','connected_1vN')
test_folder = os.path.join(here,'training_data','test','connected_1vN')


# generate folders
if not os.path.exists(train_folder):
    os.chdir(here + "\\training_data\\")
    try :
        os.mkdir('train')
    except:
        os.chdir('train')
    os.mkdir('connected_1vN')
if not os.path.exists(test_folder):
    os.chdir(here + "\\training_data\\")
    try:
        os.mkdir('test')
    except:
        os.chdir('test')
    os.mkdir('connected_1vN')

# scan through the data and find their filenames and the size
names = os.listdir(os.path.join(here, 'training_data','connected_1vN','fragments_1'))
n_files = len(names)
print("Detected ", n_files, " files!")

# get a split number
split = int(input("Enter a percentage of the data split as integer: "))

if split > 100 or split < 0:
    exit("Wrong input")
train_size = int(n_files / 100 * split)

# sample a training set
train_names = np.random.choice(names, train_size)
test_names = []
# generate test names
for name in names:
    if name not in train_names:
        test_names.append(name)

for i in range(train_size):
    name = str(i) + ".npy"
    frag_1_file = os.path.join(here, 'training_data','connected_1vN','fragments_1', name)
    frag_2_file = os.path.join(here, 'training_data','connected_1vN','fragments_2', name)
    os.replace(frag_1_file, os.path.join(train_folder,'fragments_1',name))
    os.replace(frag_2_file, os.path.join(train_folder,'fragments_2', name))

# move remaining files
for i in np.arange(start=train_size,stop=n_files):
    name = str(i) + ".npy"
    frag_1_file = os.path.join(here, 'training_data','connected_1vN','fragments_1', name)
    frag_2_file = os.path.join(here, 'training_data','connected_1vN','fragments_2', name)
    os.replace(frag_1_file, os.path.join(test_folder,'fragments_1',name))
    os.replace(frag_2_file, os.path.join(test_folder,'fragments_2', name))

# rename the files in test:
frag_1 = os.path.join(test_folder, 'fragments_1')
for i, filename in enumerate(os.listdir(frag_1)):
    name = str(i) + ".npy"
    os.rename(os.path.join(frag_1, filename), os.path.join(frag_1, name))
frag_2 = os.path.join(test_folder, 'fragments_2')
for i, filename in enumerate(os.listdir(frag_2)):
    name = str(i) + ".npy"
    os.rename(os.path.join(frag_2, filename), os.path.join(frag_2, name))

