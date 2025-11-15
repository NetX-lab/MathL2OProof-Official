
This repo implements the proposed model in the NeurIPS 2025 paper: [Learning Provably Improves the Convergence of Gradient Descent](https://arxiv.org/abs/2501.18092). We comprehensively prove the convergence of Learning to Optimize (L2O) on training.


## Overview

This repository implements and compares different optimizer architectures including LISTA-based methods, LSTM-based coordinate-wise optimizers, and DNN-based coordinate-wise optimizers (our theorem) to empirically evaluate our theorems. It includes both classic optimizers (e.g., Proximal Gradient Descent, Adam) and learned optimizers (e.g., LISTA, CoordMathLSTM, CoordMathDNN).

## Features

- **Multiple Optimizer Architectures**:
  - Classic optimizers: ProximalGradientDescent, Adam
  - LISTA-based: LISTA, LISTACPSS, LISTACPSSSTEP, LISTACPSSWOnly
  - LSTM-based: CoordMathLSTM
  - DNN-based: CoordMathDNN

- **Flexible Training Modes**:
  - `main_train_anlys.py`: Training with analysis capabilities
  - `main_train_nn.py`: Training for neural network optimization tasks
  - `main_unroll_listas.py`: Training LISTA-based optimizers with unrolling

- **Configurable Experiments**: YAML-based configuration files for easy experiment management

- **Comprehensive Logging**: Training logs, loss tracking, and TensorBoard support

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- SciPy
- configargparse
- TensorBoard (optional, for visualization)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd MathL2OProof
```

2. Install dependencies:
```bash
pip install torch numpy scipy configargparse tensorboard
```

## Project Structure

```
MathL2OProof/
├── config_parser.py          # Configuration parser with all command-line arguments
├── main_train_anlys.py       # Main training script with analysis
├── main_train_nn.py          # Training script for neural network tasks
├── main_unroll_listas.py     # Training script for LISTA-based optimizers
├── configs/                  # YAML configuration files
│   ├── 1_qp_training_train_anlys.yaml
│   └── 2_qp_testing.yaml
├── optimizees/               # Optimization problem definitions
│   ├── base.py              # Base class for optimizees
│   ├── qp.py                # Quadratic programming problems
│   ├── lasso.py             # LASSO problems
│   ├── lasso_lista.py       # LASSO for LISTA training
│   ├── logistic_l1.py       # Logistic regression with L1
│   └── cnn.py               # CNN training tasks
├── optimizers/              # Optimizer implementations
│   ├── prox_gd.py           # Proximal gradient descent
│   ├── adam.py              # Adam optimizer
│   ├── lista.py             # LISTA optimizer
│   ├── lista_cpss.py        # LISTA-CPSS variants
│   ├── coord_math_lstm.py   # Coordinate-wise LSTM optimizer
│   └── coord_math_dnn.py    # Coordinate-wise DNN optimizer
├── utils/                   # Utility functions
│   ├── utils.py             # General utilities
│   ├── util_train_loss_saver.py
│   └── plots/               # Plotting utilities
├── scripts/                 # Training and evaluation scripts
└── results/                 # Output directory for experiments
```

## Usage

### Training

#### 1. Training with Analysis (`main_train_anlys.py`)

Train a learned optimizer on optimization problems:

```bash
python main_train_anlys.py \
    --config configs/1_qp_training_train_anlys.yaml \
    --optimizer CoordMathDNN \
    --optimizee-type QuadraticUnconstrained \
    --input-dim 32 \
    --output-dim 20 \
    --sparsity 5 \
    --device cuda:0 \
    --save-dir results/my_experiment
```

#### 2. Training Neural Networks (`main_train_nn.py`)

Train optimizers for neural network optimization:

```bash
python main_train_nn.py \
    --optimizer CoordMathLSTM \
    --optimizee-type MnistCNN \
    --device cuda:0 \
    --save-dir results/cnn_training
```

#### 3. Training LISTA-based Optimizers (`main_unroll_listas.py`)

Train LISTA or LISTA-CPSS optimizers:

```bash
python main_unroll_listas.py \
    --optimizer LISTA \
    --optimizee-type LASSO_LISTA \
    --input-dim 500 \
    --output-dim 250 \
    --sparsity 50 \
    --layers 100 \
    --init-lr 1e-7 \
    --device cuda:0 \
    --save-dir results/lista_training
```

### Testing

Run trained optimizers on test problems:

```bash
python main_train_anlys.py \
    --config configs/2_qp_testing.yaml \
    --test \
    --ckpt-path results/my_experiment/CoordMathDNN.pth \
    --test-length 100 \
    --test-size 1024
```

### Configuration Files

You can use YAML configuration files to manage experiments:

```yaml
optimizee-type: QuadraticUnconstrained
input-dim: 32
output-dim: 20
sparsity: 5
rho: 0

optimizer: CoordMathDNN
lstm-layers: 2
lstm-hidden-size: 1024

device: cuda:0
init-lr: 1e-2
global-training-steps: 50000
optimizer-training-steps: 100
unroll-length: 100
train-batch-size: 32

save-dir: "my_experiment"
```

Then run with:
```bash
python main_train_anlys.py --config configs/my_config.yaml
```

## Key Parameters

### Optimizer Selection
- `--optimizer`: Choose from `CoordMathLSTM`, `CoordMathDNN`, `LISTA`, `LISTACPSS`, `Adam`, `ProximalGradientDescent`, etc.

### Problem Configuration
- `--optimizee-type`: Problem type (`QuadraticUnconstrained`, `LASSO_LISTA`, `LogisticL1`, `MnistCNN`)
- `--input-dim`: Dimension of optimization variable
- `--output-dim`: Dimension of output/labels
- `--sparsity`: Sparsity level for sparse problems
- `--rho`: Regularization parameter

### Training Configuration
- `--global-training-steps`: Total number of training iterations
- `--optimizer-training-steps`: Number of optimization steps per problem instance
- `--unroll-length`: Unrolling length for backpropagation
- `--init-lr`: Initial learning rate for meta-optimizer
- `--train-batch-size`: Batch size for training
- `--loss-func`: Loss function (`last`, `mean`, `weighted_sum`)

### LISTA-specific Parameters
- `--layers`: Number of LISTA layers
- `--lamb`: Regularization parameter for LISTA
- `--p`, `--max-p`: Support selection percentages for LISTA-CPSS
- `--w-shared`, `--s-shared`, `--theta-shared`: Parameter sharing options

### LSTM/DNN Parameters
- `--lstm-layers`: Number of LSTM layers
- `--lstm-hidden-size`: Hidden size of LSTM
- `--a-use`, `--p-use`: Enable bias/preconditioner terms
- `--a-norm`, `--p-norm`: Normalization methods (`sigmoid`, `tanh`, `exp`, etc.)

## Output

Training produces the following outputs in the `save-dir`:
- `train.log`: Training log file
- `train_loss.log`: Training loss log
- `train_loss`: Training loss array (numpy format)
- `config.yaml`: Saved configuration
- `{OptimizerName}.pth`: Model checkpoint
- `test_losses.txt`: Test losses (when testing)

## Examples

See the `scripts/` directory for example training and evaluation scripts:
- `scripts/lista_eval.sh`: LISTA training and evaluation
- `scripts/lista_cpss_eval.sh`: LISTA-CPSS training and evaluation
- `scripts/math_l2o_train/`: Mathematical L2O training scripts

## Citation

Please cite our paper if you find our codes helpful in your research or work.

```bibtex
@inproceedings{song2025mathl2oproof,
  title     = {{Learning Provably Improves the Convergence of Gradient Descents}},
  author    = {Song, Qingyu and Lin, Wei and Xu, Hong},
  booktitle = {{NeurIPS}},
  year      = {2025}
}
```


## Acknowledgement
This repository is developed based on the official implementation [1] of Math-L2O [2].


## References
[1]. Jialin Liu, Xiaohan Chen, HanQin Cai, (2023 Aug 2). MS4L2O. Github. https://github.com/xhchrn/MS4L2O.

[2]. Jialin Liu, Xiaohan Chen, Zhangyang Wang, Wotao Yin, and HanQin Cai. Towards Constituting Mathematical Structures for Learning to Optimize. In *ICML*, 2023.