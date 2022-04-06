import os
import numpy as np

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def select_folder(mode='objects'):
    here = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    if mode == 'objects':
        data_list = os.listdir(os.path.join(here, 'data'))
        print("id  name")
        for idx, val in enumerate(data_list):
            print(idx," ", val)

        idx = int(input("Enter the index of the subfolder in data where the shards are located:\n"))
        subfolder = data_list[idx]
        print("Opening folder:", subfolder)
        folder = os.path.join(here, 'data', subfolder)
    if mode == 'keypoints':
        folder = os.path.join(here, 'data', 'keypoints')
    return folder

def print_minimal_num_vertices():
     here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
     data = os.path.join(here,'data')

     min = 50000

     for folder in os.listdir(data):
        if folder == 'keypoints':
            continue
        log = os.path.join(data, folder, 'log.txt')
        with open(log) as f:
            lines = f.readlines()

        min_numbs = []
        for line in lines:
            if line.find('Minimal number of vertices after subdv: ') == 0:
                min_numbs.append(line.split(': ')[1].split('\n')[0])


        for item in min_numbs:
            if int(item) < min:
                min = int(item)
                min_folder = folder

    
     print("Minimum vertices in the dataset are ", min, " located in the folder: ", min_folder)