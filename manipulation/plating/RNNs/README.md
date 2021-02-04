# Attention
Repo for making a sequence to sequence neural network with RNNs

# Installation

## Optional: Create and source virtual environment

### Create virtual environment for this application. Note that Python3.6 is the expected Python version used for this library.
### Replace path to virtual env with environment name
`virtualenv --system-site-packages -p /usr/bin/python3.6 <path to virtual env>`

### Activate virtual environment
`source <path to virtual env>/bin/activate`

### Install some packages
`pip install --upgrade setuptools wheel tqdm twine`

You'll need also to install the [perception package](https://berkeleyautomation.github.io/perception/install/install.html). Be careful with this package downgrading packages.

You might need to use the following commands:

`pip install --upgrade --force-reinstall matplotlib`

`pip install --upgrade --force-reinstall ipython`

## Install package with requirements (cd to directory containing setup.py)
`pip install .`
