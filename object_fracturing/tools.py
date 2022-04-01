import os

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

def select_folder():
    here = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    data_list = os.listdir(os.path.join(here, 'data'))
    print("id  name")
    for idx, val in enumerate(data_list):
        print(idx," ", val)

    idx = int(input("Enter the index of the subfolder in data where the shards are located:\n"))
    subfolder = data_list[idx]
    print("Opening folder:", subfolder)
    folder = os.path.join(here, 'data', subfolder)
    return folder
    