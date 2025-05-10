# Artifical Evaluation for SAVE@ATC'25

# Overview

## Structure
```
.
├── README.md
├── src
│   ├── base            # Code for executing base, DrDNA and RedNet
│   │   ├── pytorch
│   │   └── vision
│   └── save            # Save changed code for pytorch
│       └── pytorch
└── test
    ├── Accuracy        # Figure 13 and 14
    ├── AccurateLatency # Figure 12
    └── Analysis        # Range Analysis
```
## Special Instructions
The main results we present are the core AccurateLatency test results (Figure 12) and the Accuracy test results (Figures 13 and 14) from the paper. 
Due to the varying execution dependencies of different models, we selected several representative test points to demonstrate the effects with minimizing configuration, in order to reduce the efforts of configuring different environments as much as possible.
We rewrote the original code to ensure that, while showing the effects, we could also provide a more concise interface and more readable code.

Our main claim of the paper contains three part:
1. The latency of SAVE can be as small as 9% for end-to-end performance.
2. The accuracy of SAVE can be maintained under even 4K bit flips.
3. The range analysis can check the vulnerability of different values.

## Global setup
You need to create two separate environments (it's recommended to use virtualenv or anaconda for isolation) and compile the required PyTorch for the base and the PyTorch needed for SAVE in each environment respectively. 
Both of them require the vision compiled by base as torchvision.
Below are the step-by-step instructions.

### Build pytorch
#### Build the base environment
```bash
conda create -y -n base_environment python==3.10
conda activate base_environment
cd src/base/pytorch
git submodule sync
git submodule update --init --recursive

conda install cmake ninja
conda install rust
pip install -r requirements.txt

pip install mkl-static mkl-include
pip install expecttest flake8 typing mypy pytest pytest-mock scipy requests
conda install -c pytorch magma-cuda124

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# This step may require 90 minutes to build
python setup.py develop
```
After build, run the following code to check whether it is well-built.
```bash
conda activate base_environment
python -c "import torch; print(torch.__version__);"
```
The expected output should contain `test-base`. 
Once the base pytorch is built, run the following code:
```bash
conda activate base_environment
export BASE_PYTHON=$(which python)
```

#### Build the save environment
Similar to instructions of building the base environment, a full build for pytorch is necessary.

```bash
conda create -y -n save_environment python==3.10
conda activate save_environment
cd src/save/pytorch
git submodule sync
git submodule update --init --recursive

conda install cmake ninja
conda install rust
pip install -r requirements.txt

pip install mkl-static mkl-include
pip install expecttest flake8 typing mypy pytest pytest-mock scipy requests
conda install -c pytorch magma-cuda124

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# This step may require 90 minutes to build
python setup.py develop
```
After build, run the following code to check whether it is well-built.
```bash
conda activate save_environment
python -c "import torch; print(torch.__version__);"
```
The expected output should contain `test-save`. 
Once the base pytorch is built, run the following code:
```bash
conda activate save_environment
export SAVE_PYTHON=$(which python)
```

#### Build the torchvision module
`torchvision` needs to be installed separately in both environments.
```bash
cd src/base/vision

# install torchvision in the base environment
conda activate base_environment
python setup.py develop

# install torchvision in the save environment
conda activate save_environment
python setup.py develop
```

### Python export
Before beginning any evaluation, ensure that two specific environment variables are set. 

```bash
conda activate base_environment
export BASE_PYTHON=$(which python)

conda activate save_environment
export SAVE_PYTHON=$(which python)
```