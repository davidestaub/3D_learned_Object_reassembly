import open3d as o3d
import numpy as np
from tools import *
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = select_folder()
CLEANED = os.path.join(ROOT, 'cleaned\\')
filename = os.path.join(CLEANED, 'cylinder_6_seed_1_4.obj')

V, F = [], []
with open(filename) as f:
   for line in f.readlines():
       if line.startswith('#'):
           continue
       values = line.split()
       if not values:
           continue
       if values[0] == 'v':
           V.append([float(x) for x in values[1:4]])
       elif values[0] == 'f':
           F.append([int(x) for x in values[1:4]])
V, F = np.array(V), np.array(F)-1
T = V[F][...,:2]

fig = plt.figure(figsize=(6,6))
ax = fig.add_axes([0,0,1,1], xlim=[-1,+1], ylim=[-1,+1],
                  aspect=1, frameon=False)
collection = PolyCollection(T, closed=True, linewidth=0.1,
                            facecolor="blue", edgecolor="None")
plt.legend('Test')
ax.add_collection(collection)
plt.show()