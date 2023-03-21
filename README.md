# Graph Network Simulator (GNS)

[![DOI](https://zenodo.org/badge/427487727.svg)](https://zenodo.org/badge/latestdoi/427487727)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/geoelements/gns/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/geoelements/gns/tree/main)
[![Docker](https://quay.io/repository/geoelements/gns/status "Docker Repository on Quay")](https://quay.io/repository/geoelements/gns)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/geoelements/gns/main/license.md)

> Krishna Kumar, The University of Texas at Austin.

> Joseph Vantassel, Texas Advanced Computing Center, UT Austin.

## CS224W Project
We use this repo as the starting point to simulate a cloth drop on a ball. With our enhancement, we are able to generating below result: 
![Cloth Rollout](rollout_1.gif)


To see how we enhance the original model, please refer to this link for details: https://medium.com/@bill.s.lin1/graph-network-based-simulator-of-cloth-falling-through-obstacles-cef3e066a41e

-------------------------------

Graph Network-based Simulator (GNS) is a framework for developing generalizable, efficient, and accurate machine learning (ML)-based surrogate models for particulate and fluid systems using Graph Neural Networks (GNNs). GNS code is a viable surrogate for numerical methods such as Material Point Method, Smooth Particle Hydrodynamics and Computational Fluid dynamics. GNS exploits distributed data parallelism to achieve fast multi-GPU training. The GNS code can handle complex boundary conditions and multi-material interactions.

## Run GNS
> Training
```shell
python3 -m gns.train --data_path="<input-training-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>" -ntraining_steps=100
```
> Our cloth setting
```shell
python gns/train.py --data_path data/cloth/ --output_path data/cloth/output/ --model_path data/cloth/model/
```

> Resume training

To resume training specify `model_file` and `train_state_file`:

```shell
python3 -m gns.train --data_path="<input-training-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>"  --model_file="model.pt" --train_state_file="train_state.pt" -ntraining_steps=100
```
> Our cloth setting
```shell
python gns/train.py --data_path data/cloth/ --output_path data/cloth/output/ --model_path data/cloth/model/ --model_file model-30000.pt --train_state_file train_state-30000.pt
```

> Rollout
```shell
python3 -m gns.train --mode="rollout" --data_path="<input-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>" --model_file="model.pt" --train_state_file="train_state.pt"
```
> Our cloth setting
```shell
python gns/train.py --mode rollout --data_path data/cloth/ --output_path data/cloth/output/ --model_path data/cloth/model/ --train_state_file train_state-30000.pt --model_file model-30000.pt
```

> Render
```shell
 python3 -m gns.render_rollout --rollout_path="<path-containing-rollout-file>/rollout_0.pkl" 
```
> Our cloth setting
```shell
python gns/render_rollout.py --rollout_path data/cloth/output/
```

The renderer also writes `.vtu` files to visualize in ParaView.

<!-- ![Sand rollout](figs/rollout_0.gif)
> GNS prediction of Sand rollout after training for 2 million steps. -->

## Datasets

### Original paper dataset

The data loader provided with this PyTorch implementation utilizes the more general `.npz` format. The `.npz` format includes a list of
tuples of arbitrary length where each tuple is for a different training trajectory
and is of the form `(position, particle_type)`. `position` is a 3-D tensor of
shape `(n_time_steps, n_particles, n_dimensions)` and `particle_type` is
a 1-D tensor of shape `(n_particles)`.  

The dataset contains:

* Metadata file with dataset information (sequence length, dimensionality, box bounds, default connectivity radius, statistics for normalization, ...):

* npz containing data for all trajectories (particle types, positions, global context, ...):

We provide the following datasets:
  * `WaterDropSample` (smallest dataset)
  * `Sand`
  * `SandRamps`

Download the dataset from [DesignSafe DataDepot](https://doi.org/10.17603/ds2-0phb-dg64). If you are using this dataset please cite [Vantassel and Kumar., 2022](https://github.com/geoelements/gns#dataset)

### Our cloth setting 
We used the open source [Taichi physics simulator](https://docs.taichi-lang.org/blog/head-first-taichi) to generate the source-of-truth dataset by running 1000 experiments of a piece of cloth falling onto a spherical obstacle. You can find the dataset that we generated [here](https://drive.google.com/drive/folders/1q03NoTLQbFenZIVOehBDadiSx32mbxl6?usp=share_link).


## Building environment on TACC LS6 and Frontera

- to setup a virtualenv

```shell
sh ./build_venv.sh
```

- check tests run sucessfully.
- start your environment

```shell
source start_venv.sh 
```

### Inspiration
PyTorch version of Graph Network Simulator based on [https://arxiv.org/abs/2002.09405](https://arxiv.org/abs/2002.09405) and [https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate).

### Acknowledgement
This code is based upon work supported by the National Science Foundation under Grant OAC-2103937.

### Citation

#### Repo
Kumar, K., & Vantassel, J. (2022). Graph Network Simulator: v1.0.1 (Version v1.0.1) [Computer software]. https://doi.org/10.5281/zenodo.6658322

#### Dataset
Vantassel, Joseph; Kumar, Krishna (2022) “Graph Network Simulator Datasets.” DesignSafe-CI. https://doi.org/10.17603/ds2-0phb-dg64 v1 
