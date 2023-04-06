# gnn-paper

## Key files

```
├── data: scripts and classes related to data processing
├── gnn: ML modules
    ├── models.py: classes for main models
    ├── blocks.py: blocks which are used by the models
├── train_[model].py: training script for each model
```
Our main models are:
- LatticeGNN: equivariant approach without frame of reference
We use **mace** architecture with equivariant message passing. 
- SpectralGNN: equivariant approach with prediction of frame of reference

Benchmark models are:
- LatticeAttention: equivariant approach in which I tried to fully-connect 
the graphs and use attention mechanisms. Performance dropped
- CrystGraphConv (CGC): non-equivariant approach. With enough data, the model 
can empirically learn the equivariance. However, it is not guaranteed.
https://doi.org/10.1103/PhysRevLett.120.145301
- MCrystGraphConv (MCGC): non-equivariant approach similar to CGC, but 
here edge features are also updated. Again, with enough data, the model 
learns the equivariance, but it is not guaranteed. Performance is not much
better than CGC.
https://doi.org/10.1016/j.matdes.2022.111175

