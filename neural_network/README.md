# Neural Network
This folder deals with the neural network setup, training and prediction.
### Configuration
You can find the configuration file in ````utils/config````.
It provides a lot of settings for the architecture as well the data to use.
Please make sure that you set the keypoints and descriptors right! They should be the same as used
in the processing step.
### WANDB
You can monitor the training with [WANDB](https://wandb.ai/). Provide your login credentials in the config file
to use it.

## Training
Train the network on the default cube dataset, located in ```object_fracturing/data```
````
python3 train.py
````
or add a custom path with the ````--path```` argument. The final weights are stored in the ```weights```
folder as ````model_weights.pth````.
## Inference
For inference, the previously trained weights or provided pretrained weights, saved as ```weights/model_weights_best.pth``` can be used.
To predict keypoint matches, using a match_threshold of 0.9 use the command
````
python3 predict.py --weights_path weights/model_weights.pth --sensitivity 0.9
````
You will be asked to give the parent directory of your dataset.

The pairwise predictions for each fragment pair of each object are found in a ``predictions`` subfolder.
A prediction file can look as follows:

``prediction_4_6_m0.npy``

The first number from the left indicates the 0th fragment number, the following number indicates the 1st fragment number.
The m0 indicates that there are matches from the 0th fragment to the 1st fragment, m1 would be from fragment 1 to fragment 0. This file therefore contains
a matching matrix from fragment 4 to fragment 6.

