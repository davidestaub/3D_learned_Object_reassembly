# 3D Object Reassembly
Group 19: Mathias Vogel, Julia Bazinska, Katarzyna Krasnopolska, Davide Staub

## Quickstart

```
pip install -r requirements.txt
cd full_pipeline
python3 reassemble_object.py --object_dir ./example_data/cube_10_seed_0
```

You should see first a visualization of a scrambled cube and then a reassembled one.

## Project structure
- evaluation – contains a keypoint visualizer as well as 
- full_pipeline – reassemble an object from its shards, including all necessary intermediate steps
- keypoints_and_descriptors – calculations of keypoints and features
- neural_network – build and train a neural network and use it for predictions
- object_fracturing – fracture objects with blender and preprocess the objects for the neural network
- object_reassembly – load preprocessed object fragments and reassemble either using ground truth matches or predictions from 
the network

