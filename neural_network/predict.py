import argparse
import os
import shutil
import tkinter
from tkinter import filedialog

import numpy as np
import torch
from tqdm import tqdm

from neural_network.dataset import DatasetPredict
from neural_network.model import build_model, batch_to_device
from neural_network.utils import conf as conf


def create_output_folders(folder_root):
    """Creates necessary folders for the model prediction"""
    for folder in os.listdir(folder_root):
        shutil.rmtree(os.path.join(folder_root, folder, 'predictions'), ignore_errors=True)
        os.makedirs(os.path.join(folder_root, folder, 'predictions'))


def predict(weights_path, folder_path, single_object=False):
    # Set single object to true if `folder_path` points directly to the object directory. Set it to False if it's a
    # folder containing multiple object folders.
    config_all = {**conf.model_conf, **conf.data_conf, **conf.train_conf}
    print(conf.train_conf)
    model = build_model(weights_path, config_all)
    dataset = DatasetPredict(folder_path, config_all, single_object)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    if single_object:
        shutil.rmtree(os.path.join(folder_path, 'predictions'), ignore_errors=True)
        os.makedirs(os.path.join(folder_path, 'predictions'))
    else:
        create_output_folders(folder_path)

    # predict each possible pair
    for item in tqdm(dataset, desc="Predict"):
        with torch.no_grad():
            item = batch_to_device(item, device, non_blocking=True)
            pred = model(item)

        name = item['pair_name']
        basename = '_'.join(name.split('_')[:-2])
        name_pair = '_'.join(['prediction'] + name.split('_')[-2:])
        if single_object:
            basepath = os.path.join(folder_path, 'predictions')
        else:
            basepath = os.path.join(folder_path , basename, 'predictions')
        m0 = pred["matches0"].cpu().squeeze()
        m1 = pred["matches1"].cpu().squeeze()
        print(f"Num matches 0: {(m0 != -1).sum()}")
        print(f"Num matches 1: {(m0 != -1).sum()}")

        np.save(os.path.join(basepath, f'{name_pair}_m0.npy'), m0)
        np.save(os.path.join(basepath, f'{name_pair}_m1.npy'), m1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', default=os.path.join('weights', 'model_weights.pth'))
    parser.add_argument('--data_dir')
    args = parser.parse_intermixed_args()

    if not args.data_dir:
        root = tkinter.Tk()
        root.withdraw()
        root = filedialog.askdirectory(parent=root, initialdir=os.getcwd(),
                                       title='Please select the parent directory of the fractured object folders')
    else:
        root = args.data_dir
    # create the necessary config and create the model and dataset

    predict(args.weights_path, root)
