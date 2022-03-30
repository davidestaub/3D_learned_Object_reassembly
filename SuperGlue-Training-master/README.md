# SuperGlue Training

The framework is described below.

- To train SuperGlue for SuperPoint on MegaDepth, the configuration is in `d3dv/configs/megadepth_superglue_scratch.yaml`. Assuming that keypoints, scores, and descriptors are exported to `$DATA_PATH/exports/SP-nms3-n2048_megadepth-undist-r1024/`, the command is then:
```
python -m d3dv.train megadepth_SG-SP_n1024 --conf d3dv/configs/megadepth_superglue_scratch.yaml
```
- To train for DISK, with features in `$DATA_PATH/exports/DISK-n2048_megadepth-undist-r1024/`, use:
```
python -m d3dv.train megadepth_SG-DISK_n1024 --conf d3dv/configs/megadepth_superglue_disk.yaml
```

Add the flag `--distributed` to train on multiple GPUs. For 1024 keypoints per image and a batch size of 32 images pairs, with gradient checkpointing, the total memory requirement is around 14GB. It is possible to reduce the batch size to 24 to fit on an 11GB GPU, but the performance might be slightly lower. A good convergence should be achieved in around 150k iterations.

##

`d3dv` is a general codebase designed to train deep learning components for 3D geometry tasks: image matching, pose estimation, or depth prediction. `d3dv` is built on top of a framework whose core principles are:
- modularity: it is easy to add a new dataset or model with custom loss and metrics;
- reusability: components like geometric primitives, training loop, or experiment tools are reused across projects;
- reproducibility: a training run is parametrized by a configuration, which is saved and reused for evaluation;
- simplicity: it has few external dependencies, and can be easily grasped by a new user.

## Installation
`d3dv` requires Python >=3.7 and PyTorch >=1.3. `requirements.txt` lists other dependencies. To use `d3dv` from outside, like a notebook, install it locally with `pip3 install -e .`

You need to rename [`setting.py.example`](d3dv/settings.py.example) to `settings.py` and edit the the paths of two directories:
- `EXPER_PATH` will contain the experiment outputs: configurations, logs, checkpoints, evaluation results
- `DATA_PATH` will contain the datasets and pretrained model weights

## Framework structure
`d3dv` includes of the following components:
- [`datasets/`](d3dv/datasets) contains the dataloaders, all inherited from [`BaseDataset`](d3dv/datasets/base_dataset.py). Each loader is configurable and produces a set of batched data dictionaries.
- [`models/`](d3dv/models) contains the deep networks and learned blocks, all inherited from [`BaseModel`](d3dv/models/base_model.py). Each model is configurable, takes as input data, and outputs predictions. It also exposes its own loss and evaluation metrics.
- [`geometry/`](d3dv/geometry) groups Numpy/PyTorch primitives for 3D vision: [linear algebra](d3dv/geometry/utils.py), [2D](d3dv/geometry/viz_2d.py) or [3D](d3dv/geometry/viz_3d.py) visualization, optimization, etc.
- [`utils/`](d3dv/utils) contains various utilities, for example to [manage experiments](d3dv/utils/experiments.py).

Datasets, models, and training runs are parametrized by [omegaconf](https://github.com/omry/omegaconf) configurations. See examples of training configurations in [`d3dv/configs/`](d3dv/configs/) as `.yaml` files.

## Workflow
<details>
<summary><b>Training:</b></summary><br/>
  
The following command starts a new training run:
```bash
python3 -m d3dv.train experiment_name --conf d3dv/configs/config_name.yaml
```

It creates a new directory `experiment_name/` in `EXPER_PATH` and dumps the configuration, model checkpoints, logs of stdout, and [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) summaries. As an example, train a CNN on MNIST with:
```bash
python3 -m d3dv.train my_mnist_training --conf d3dv/configs/mnist_train.yaml
```
Extra flags can be given:
- `--overfit` loops the training and validation sets on a single batch ([useful to test losses and metrics](http://karpathy.github.io/2019/04/25/recipe/)).
- `--restore` restarts the training from the last checkpoint (last epoch) of the same experiment.
- `--distributed` uses all GPUs available with multiple processes and batch norm synchronization.
- individual configuration entries to overwrite`.yaml`. Examples: `train.lr=0.001` or `data.batch_size=8`.

**Monitoring the training:** Launch a Tensorboard session with `tensorboard --logdir=path/to/EXPER_PATH` to visualize losses and metrics, and compare them across experiments. Press `Ctrl+C` to gracefully interrupt the training.
</details>

<details>
<summary><b>Inference with a trained model:</b></summary><br/>

After training, you can easily load a model to evaluate it:
```python
from d3dv.utils.experiments import load_experiment

test_conf = {}  # will overwrite the training and default configurations
model = load_experiment('name_of_my_experiment', test_conf)
model = model.cuda()  # optionally move the model to GPU
predictions = model(data)  # data is a dictionary of tensors
```

</details>

<details>
<summary><b>Working on the Leonhard cluster:</b></summary><br/>
  
`source env.sh` creates a virtual environment and loads the Python and CUDA modules. Install the remaining requirements with `pip3 install -r requirements.txt` (only once).

Prepend any command with `./launch_cluster.sh [-I]` to launch as an (optionally interactive) GPU job. A few minutes before the time limit, an interrupt signal will be sent to the job, which will gracefully stop the training. As described above, it can later be restarted with the `--restore` flag.
</details>

<details>
<summary><b>Adding new datasets or models:</b></summary><br/>

You simply need to create a new file in [`datasets/`](d3dv/datasets/) or [`models/`](d3dv/models/). This makes it easy to collaborate on the same codebase. Each class should inherit from the base class, declare a `default_conf`, and define some specific methods. Have a look at the base files [`BaseDataset`](d3dv/datasets/base_dataset.py) and [`BaseModel`](d3dv/models/base_model.py) for more details. Please use [PEP 8](https://www.python.org/dev/peps/pep-0008/) and relative imports.
</details>

## Example: feature matching
[`models/two_view_pipeline.py`](d3dv/models/two_view_pipeline.py) implements a typical two-view feature matching pipeline. [This notebook](notebooks/demo_two-view_matching.ipynb) shows an example with [MegaDepth](d3dv/datasets/megadepth.py) data and [SuperPoint](d3dv/models/superpoint.py) local features matched by [Nearest Neighbor](d3dv/models/nearest_neighbor_matcher.py). More can be easily added.

Created and maintained by [Paul-Edouard Sarlin](https://psarlin.com/).
