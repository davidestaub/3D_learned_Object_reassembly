import os
from compas.datastructures import Mesh
from tools import *
from compas.geometry import Pointcloud
import numpy as np
from compas_view2.app import App
import shutil 


here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
train_folder = os.path.join(here, 'training_data', 'train', 'connected_1vN')
test_folder = os.path.join(here, 'training_data', 'test', 'connected_1vN')
frag_1_path = os.path.join(here, 'training_data','connected_1vN', 'fragments_1')
frag_2_path = os.path.join(here, 'training_data','connected_1vN', 'fragments_2')

# generate folders
if not os.path.exists(train_folder):
    os.chdir(here + "\\training_data\\")
    try:
        os.mkdir('train')
    except:
        pass
    os.chdir('train')
    os.mkdir('connected_1vN')
    os.chdir('connected_1vN')
    os.mkdir('fragments_1')
    os.mkdir('fragments_2')
if not os.path.exists(test_folder):
    os.chdir(here + "\\training_data\\")
    try:
        os.mkdir('test')
    except:
        pass
    os.chdir('test')
    os.mkdir('connected_1vN')
    os.chdir('connected_1vN')
    os.mkdir('fragments_1')
    os.mkdir('fragments_2')

# scan through the data and find their filenames and the size
names = os.listdir(frag_1_path)
n_files = len(names)
print("Detected ", n_files, " files!")

# get a split number
split = int(input("Enter a percentage of the data split as integer: "))

if split > 100 or split < 0:
    exit("Wrong input")
train_size = int(n_files / 100 * split)

# sample a training set
train_names_frag1 = np.random.choice(names, train_size)
test_names_frag1 = []

# generate test names
for name in names:
    if name not in train_names_frag1:
        test_names_frag1.append(name)

# generate counterpart names
train_names_frag2 = []
test_names_frag2 = []
for item in train_names_frag1:
    train_names_frag2.append(item.split('part')[0]+"counterpart_"+item.split('_')[-1])
for item in test_names_frag1:
    test_names_frag2.append(item.split('part')[0]+"counterpart_"+item.split('_')[-1])

# copy the files
for src in train_names_frag1:
    shutil.copy(os.path.join(frag_1_path,src), os.path.join(train_folder,'fragments_1'))

for src in test_names_frag1:
    shutil.copy(os.path.join(frag_1_path,src), os.path.join(test_folder,'fragments_1'))

for src in train_names_frag2:
    shutil.copy(os.path.join(frag_2_path,src), os.path.join(train_folder,'fragments_2'))

for src in test_names_frag2:
    shutil.copy(os.path.join(frag_2_path,src), os.path.join(test_folder,'fragments_2'))