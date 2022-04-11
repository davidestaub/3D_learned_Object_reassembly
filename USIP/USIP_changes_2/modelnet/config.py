import os 
# mode up to USIP data folder
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
os.chdir('..')
os.chdir('..')

# update dir path to parent directory where
# all of the usip files are located
dir_path = os.getcwd()
print("Initialized root path to:\n", dir_path)
# define the different folder paths
usip_root = os.path.join(dir_path, 'USIP')
dataroot = os.path.join(dir_path,'data')
model_save_path = os.path.join(dir_path, 'trained_models')
pretrained_model_path = os.path.join(dir_path,'pretrained_usip.pth')
keypoints_path = os.path.join(dir_path,'keypoints')

# number of epochs the model trains for
epochs = 1000

# batch size for unet and loaders 
# since our pointclouds have different sizes, better use 1
batch_size = 1

# start learning rate (there is decay also)
learning_rate = 0.001

# the number of pointclouds totally used (npy files)
# this does not have to match the actual number --- currently investigating
number_of_pointclouds = 1000

# this parameter is to control the distance of the keypoints to the actual 
# pointcloud. The bigger the value, the close it should be.
# note that this also affects training
# should be in range 0.5-5 (default 0.5)
kp_alpha = 5

# the node number (M) is used in the FPN, it samples M nodes from the pointcloud
# processes them and should then estimate M proposal keypoints and their saliency
node_num = 128

# this value corresponds to the K value in the FPN. Increasing K values cause degenracies
# in the feature propsal network
knn_k = 8