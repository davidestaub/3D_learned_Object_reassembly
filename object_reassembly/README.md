# 3D Object Reassembly
This folder deals only with the reasembly part of the problem. It is assumed that network predictions or ground truth matches are given


## Quickstart
1. Run the file ```reassembly_main.py``` 
2. When prompted, ```select example_data/cube_10_seed_0``` as the data folder
3. A window appears showcasing the pre-reassembled object
4. Close the window
5. A new window appeares showcasing the reassembled cube

### Ground truth or predictions
One can choose between using the ground truth matches or predictions from the neural network. 
- To use ground truth matches set ````obj.load_matches(use_ground_truth=True)```` in the ```reassembly_main.py file```
- To use network predictions set ````obj.load_matches(use_ground_truth=False)```` in the ```reassembly_main.py file```

### Provide your own data
An example folder is already given under: ````example_data/cube_10_seed_0````. If you want to use your own data, provide a folder with the same structure and content as the example folder, namely:


- ````\predictions````:
This folder is neccessary when using network predictions, if you only intend on using ground truth data then you do not need this folder.
The "predictions" folder should be named as such, and contain all the fragment matching files. The files should be named as follows:
  - `````prediction_$FRAGMENT NR 1$_$FRAGMENT NR 2$_m0.npy````` if the file contains the matches from fragment 1 to fragment 2.
  - `````prediction_$FRAGMENT NR 1$_$FRAGMENT NR 2$_m1.npy````` if the file contains the matches from fragment 2 to fragment 1.
####TODO describe what exactly should be inside predictions_0_1_m0.npy
- ````\processed````:
This folder is neccessary both for using ground truth data and network predictions.
It should contain the following 3 subfolders:
  - ````\keypoint_descriptos````: A folder containig the descriptors for each fragment, saved as:
  `````keypoint_descriptors_hybrid_pillar.$FRAGMENT NR$.npy`````
  - ````\keypoints````: #TODO
  - ````\matching````: #TODO, I am a bit unsure here, so one should put the matching_matrix when using gt and what should be in here exactly when using network predictions?




