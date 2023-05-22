# Conditional MuseGAN

## Environment setup

```bash
$ cd tp_pt_vae
$ pip install -e .
```

## Folder architecture
coming soon

## Run

In this repo, hyperparameters are managed using [Hydra](https://hydra.cc/docs/intro/).<br>
Hydra provides an easy way to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

### Dataset preparation
coming soon

### Training

```bash
# Train a model customizing the hyperparameters as you like
$ tp_pt_vae-train data=lod_small out_dir=exp/tp_pt_vae
```

### Inference

```bash
# Decode with several F0 scaling factors
$ tp_pt_vae-decode data=lpd_small out_dir=exp/tp_pt_vae checkpoint_steps=400000
```

### Analysis-Synthesis

coming soon

### Monitor training progress

```bash
$ tensorboard --logdir exp
```
