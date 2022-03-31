import numpy as np
import pptk
import os
from tools import *

PATH, _ = select_folder()
print(1, "cleaned")
print(2, "subdivided")
choice = int(input("Which version do you want to plot?"))
if choice not in [1,2,3]:
    exit("You failed to type an integer between 1 and 2...\n\n\n")

if choice == 1:
    FOLDER = os.path.join(PATH, 'cleaned')
if choice == 2:
    FOLDER = os.path.join(PATH, 'subdv')

if not os.path.exists(FOLDER):
    exit("There is no matching folder")

files = [ob for ob in os.listdir(FOLDER) if ob.endswith(".npy")]

for filename in files:
    