# Disentangled mesh Variational autoencoder

Official Pytorch implementation of the paper [**"Disentangled representations: towards interpretation of sex determination from hip bone"**](https://arxiv.org/pdf/2112.09414.pdf)

#### Bibtex
If you find this code useful in your research, please cite:

```
@INPROCEEDINGS{petrovich21actor,
  title     = {Action-Conditioned 3{D} Human Motion Synthesis with Transformer {VAE}},
  author    = {Petrovich, Mathis and Black, Michael J. and Varol, G{\"u}l},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year      = {2021}
}
```
## Requirements

This code is tested on Python3.8, Pytorch versoin 1.11.0+cu113, torch-geometric version 2.0.4 . Requirments can be install by running

      pip install -r requirements.txt
    
Install mesh processing libraries from [MPI-IS/mesh](https://github.com/MPI-IS/mesh). Note that the python3 version of mesh package library is needed for this.
