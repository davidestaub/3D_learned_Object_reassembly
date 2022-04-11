import os 
import sys

# change the system path to USIP root to load the config
dir_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir_path, 'USIP')
sys.path.insert(0,path)

from modelnet import config

# find the best model
best_loss = 100
best_name = ''

for filename in os.listdir(config.model_save_path):
    if filename.endswith(".pth") and len(filename.split('_')) > 3:
        model_loss = float(filename.split('_')[2])
        if model_loss < best_loss:
            best_loss = model_loss
            best_name = filename

if best_loss ==  100:
    print("Error in finding the best model")
    exit(1)

print("Best model loss is:", best_loss)


# delete previous best model
os.chdir(config.model_save_path)
if os.path.exists('best.pth'):
    os.remove('best.pth')

# rename the file to a simpler name
os.rename(best_name, 'best.pth')


# delete all models except the best one
for filename in os.listdir(config.model_save_path):
    if filename.endswith(".pth"):
        if filename == 'best.pth':
            continue 
        else:
            path = os.path.join(config.model_save_path, filename)
            os.remove(path)