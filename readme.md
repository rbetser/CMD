This repository is the official implementation of:
['Enhancing Neural Training via a Correlated Dynamics Model' (ICLR 2024)](https://iclr.cc/virtual/2024/poster/18304)


**Requirements:**
- Python 3.8.8
- pytorch 1.13.1
- CUDA Version 11.6.112

Additional python library:
- ordered_set


**Usage**:

1) The default example is executed as followed:
```sh
python train_cmd.py
```

Default example details:
 - ResNet18 on CIFAR10 (SGD optimizer, learning rate - 0.05, momentum - 0.9, 150 epochs, no scheduler)
 - 3 methods are compared - SGD, Post-hoc CMD, Online CMD
 - CMD is performed with 10 modes (M), 1000 sampled weights (K) and 20 warm-up epochs (F).
 The number of modes and sampled weights are hard-coded per model type, warm-up epochs are given as an argument.
 
2) Add P-BFGS to the compared methods:
```sh
python train_cmd.py --p_bfgs True
```
Default is P-BFGS with 80 warm-up epochs and 40 dimensions.

3) Remove one of the 4 methods (example on Post-hoc CMD):
```sh
python train_cmd.py --cmd_PostHoc ''
```

4) Example of comparing only Online CMD to P-BFGS:
```sh
python train_cmd.py --p_bfgs True --sgd '' --cmd_PostHoc ''
```

Embedded CMD is also available, with an argument similar to the P-BFGS argument. Note that the embedding algorithm hyperparameters are given as arguments, the default values are suited to ResNet18.

All the arguments are listed in the bottom of the main file (train_cmd.py) with explanations.  
Available model types - Resnet18, PreRes164, WideRes28-10, LeNet-5, GoogleNet, ViT-b-16
Training parameters (optimizer type, scheduler type, number of epochs, learning rate, momentum, number of CMD modes, etc.) are different per model type.
Therefore, these parameters are hard coded and not coded as arguments. To access them see the function - 
get_model() in base_utils.py in the utils directory. These parameters follow the parameters used in the results section in the paper.
