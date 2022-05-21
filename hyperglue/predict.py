import argparse, os
from tqdm import tqdm
from hyperglue_cluster import build_model, batch_to_device
from dataset import DatasetPredict
from utils import conf
import numpy as np
import tkinter
from tkinter import filedialog
import os
import torch


def create_output_folders(folder_root):
    """Creates necessary folders for the model prediction"""
    for folder in os.listdir(folder_root):
        if "prediction" in folder:
            continue
        os.makedirs(os.path.join(folder_root, folder, 'predictions'), exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', default='weights\LW677.pth')
    args = parser.parse_intermixed_args()

    root = tkinter.Tk()
    root.withdraw()
    root = filedialog.askdirectory(parent=root, initialdir=os.getcwd(),
                                   title='Please select the parent directory of the fractured object folders')

    # create the necessary config and create the model and dataset
    config_all = {**conf.model_conf, **conf.data_conf, **conf.train_conf}
    model = build_model(args.weights_path, config_all)
    dataset = DatasetPredict(root, config_all)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    create_output_folders(root)

    # predict each possible pair
    for item in tqdm(dataset, desc="Predict"):
        with torch.no_grad():
            item = batch_to_device(item, device, non_blocking=True)
            pred = model(item)

        name = item['pair_name']
        basename = '_'.join(name.split('_')[:-2])
        name_pair = '_'.join(['prediction'] + name.split('_')[-2:])
        m0 = pred["matches0"].cpu()
        m1 = pred["matches1"].cpu()
        np.save(os.path.join(root, basename,'predictions', f'{name_pair}_m0.npy'), m0)
        np.save(os.path.join(root, basename,'predictions', f'{name_pair}_m1.npy'), m1)
