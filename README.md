# Simple Hierarchical Planning with Diffusion
Chang Chen, Fei Deng, Kenji Kawaguchi, Caglar Gulcehre, Sungjin Ahn

https://arxiv.org/pdf/2401.02644

Abstract: Diffusion-based generative methods have proven effective in modeling trajectories with offline datasets. However, they often face computational challenges and can falter in generalization, especially in capturing temporal abstractions for longhorizon tasks. To overcome this, we introduce the Hierarchical Diffuser, a simple, fast, yet surprisingly effective planning method combining the advantages of hierarchical and diffusion-based planning. Our model adopts a “jumpy” planning strategy at the higher level, which allows it to have a larger receptive field but at a lower computational cost—a crucial factor for diffusion-based planning methods, as we have empirically verified. Additionally, the jumpy sub-goals guide our lowlevel planner, facilitating a fine-tuning stage and further improving our approach’s effectiveness. We conducted empirical evaluations on standard offline reinforcement learning benchmarks, demonstrating our method’s superior performance and efficiency in terms of training and planning speed compared to the non-hierarchical Diffuser as well as other hierarchical planning methods. Moreover, we explore our model’s generalization capability, particularly on how our method improves generalization capabilities on compositional out-of-distribution tasks.

## Installation

```
conda env create -f environment.yml
conda activate hier_diffusion
pip install -e .
```

## Model Training

The high-level and low-level planner can be trained in parrallel as follows:
- Train the high-level planner:
```
python scripts/train.py --config config.maze2d_hl --dataset maze2d-large-v1
```
- Train the low-level planner
 ```
python scripts/train.py --config config.maze2d_ll --dataset maze2d-large-v1
```

## Model Evaluation
After training, the hierarchical planning model can be evaluated with:
```
python scripts/hd_plan_maze2d.py --dataset maze2d-large-v1
```

## Citation
```
@inproceedings{
chen2024simple,
title={Simple Hierarchical Planning with Diffusion},
author={Chang Chen and Fei Deng and Kenji Kawaguchi and Caglar Gulcehre and Sungjin Ahn},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=kXHEBK9uAY}
}
```

## Acknowledgements
This code is based on Michael Janner's [Diffuser](https://github.com/jannerm/diffuser) repo. We thank the authors for their great works!
