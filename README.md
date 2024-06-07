# Simple Hierarchical Planning with Diffusion
Chang Chen, Fei Deng, Kenji Kawaguchi, Caglar Gulcehre, Sungjin Ahn

https://arxiv.org/pdf/2401.02644

Abstract: Diffusion-based generative methods have proven effective in modeling trajectories with offline datasets. However, they often face computational challenges and can falter in generalization, especially in capturing temporal abstractions for longhorizon tasks. To overcome this, we introduce the Hierarchical Diffuser, a simple, fast, yet surprisingly effective planning method combining the advantages of hierarchical and diffusion-based planning. Our model adopts a “jumpy” planning strategy at the higher level, which allows it to have a larger receptive field but at a lower computational cost—a crucial factor for diffusion-based planning methods, as we have empirically verified. Additionally, the jumpy sub-goals guide our lowlevel planner, facilitating a fine-tuning stage and further improving our approach’s effectiveness. We conducted empirical evaluations on standard offline reinforcement learning benchmarks, demonstrating our method’s superior performance and efficiency in terms of training and planning speed compared to the non-hierarchical Diffuser as well as other hierarchical planning methods. Moreover, we explore our model’s generalization capability, particularly on how our method improves generalization capabilities on compositional out-of-distribution tasks.

![hd_generation](https://github.com/changchencc/Simple-Hierarchical-Planning-with-Diffusion/assets/22546741/0c59068d-0222-418f-b823-46acb54f28ae)

## Installation
This branch contains the code for training and evaluating the model on the Gym-MuJoCo tasks. The maze_2d branch contains code for hierarchical planning on the Maze2D tasks.

```
conda env create -f environment.yml
conda activate hier_diffusion
pip install -e .
```

## Model Training

- The high-level and low-level planner can be trained in parrallel as follows:
```
# Train the high-level planner
python scripts/train.py --config config.locomotion_hl --dataset walker2d-medium-replay-v2
# Train the low-level planner
python scripts/train.py --config config.locomotion_ll --dataset walker2d-medium-replay-v2
```

- Train the value predictor:
```
# Train the value predictor for the high-level planner
python scripts/train_values.py --config config.locomotion_hl --dataset walker2d-medium-replay-v2
# Train the value predictor for the low-level planner
python scripts/train_values.py --config config.locomotion_ll --dataset walker2d-medium-replay-v2
```

## Model Evaluation
To evaluate the model, follow the command provided below. You may need to adjust the guidance-related hyper-parameters. Please see the example in `scripts/test.sh`.
```
python scripts/hd_plan_guided.py --dataset halfcheetah-medium-expert-v2
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
