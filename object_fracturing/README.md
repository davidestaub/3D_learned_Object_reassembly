# Object fracturing
This folder deals with the dataset generation. For simplicity just cubes
are generated and fractured with the blender 2.79 fracture modifier.
## Installation
Install blender 2.79 for your platform.
````
cd object_fracturing
python3 download_blender.py --platform [your-platform (linux,mac,windows)]
````
Finally, extract the archive.

## Quickstart
Generate a synthetic cube dataset with 10 cubes and 10
fragments per cube the following way:
````
python3 generate_dataset.py --num_cubes 10 --num_frags 10
````
The generated cubes will appear in the data folder. To clean the folders up
by deleting the material files, splitting fragments in loose parts and deleting small
shards type
```` 
python3 clean_data.py
````
After cleanup, keypoints and features can be extracted.
```` 
python3 process_data.py --keypoint_method hybrid --descriptor_method pillar
````
Now the neural network can be trained on the data. You can find a log for both
the cleaning and processing in each object folder.

#### Remark
There is support for the following keypoints:
- iss ([Intrinsic Shape Signatures](http://www.open3d.org/docs/latest/tutorial/Advanced/iss_keypoint_detector.html))
- SD (Multi-scale symbolic projection distance)
- harris ([3D Harris](https://github.com/rengotj/projet_NPM))
- sticky (Adapted from [StickyPillars](https://arxiv.org/abs/2002.03983))
- hybrid (Mixture between SD and StickyPillars)

Descriptors which can be used:
- fpfh (Fast Point Feature Histograms)
- pillar (Adapted from [StickyPillars](https://arxiv.org/abs/2002.03983))
- fpfh_pillar (mix between two)