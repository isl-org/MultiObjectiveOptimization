# Multi-Task Learning as Multi-Objective Optimization

This code repository includes the source code for the [Paper](https://arxiv.org/abs/1810.04650):

```
Multi-Task Learning as Multi-Objective Optimization
Ozan Sener, Vladlen Koltun
Neural Information Processing Systems (NeurIPS) 2018 
```

The experimentation framework is based on PyTorch; however, the proposed algorithm (MGDA_UB) is implemented largely Numpy with no other requirement. So, it should be trivial to extend to other deep learning frameworks. PyTorch version is implemented in `min_norm_solvers.py`, generic version using only Numpy is implemented in file `min_norm_solvers_numpy.py`.

This repo includes more than the implementation of the paper. It imlpements both Frank-Wolfe and projected gradient descent method. It also has smart initialization and gradient normalization tricks which are described with inline comments.

The source code and dataset (MultiMNIST) are released under the MIT License. See the License file for details.


# Requirements and References
The code uses the following Python packages and they are required: ``tensorboardX, pytorch, click, numpy, torchvision, tqdm, scipy, Pillow``

The code is only tested in ``Python 3`` using ``Anaconda`` environment.

We adapt and use some code snippets from:
* [CSAILVision Semanti Segmentation](https://github.com/CSAILVision/semantic-segmentation-pytorch)
* [PyTorch-SemSeg](https://github.com/meetshah1995/pytorch-semseg/)



# Usage
The code base uses `configs.json` for the global configurations like dataset directories, etc.. Experiment specific parameters are provided seperately as a json file. See the `sample.json` for an example.

To train a model, use the command: 
```bash
python multi_task/train_multi_task.py --param_file=./sample.json
```

# Contact
For any question, you can contact ozan.sener@intel.com

# Citation
If you use this codebase or any part of it for a publication, please cite:
```
@incollection{NeurIPS2018_Sener_Koltun,
title = {Multi-Task Learning as Multi-Objective Optimization},
author = {Sener, Ozan and Koltun, Vladlen},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {525--536},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf}
}
```
