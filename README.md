# Download resources

```sh
$ make -j
```

# Install

Tested on [![Python version](https://img.shields.io/badge/python-3.9.7%20-blue.svg)](https://cs.tinyiu.com/sbp-env)

Create a virtual environment for the required packages. We suggest `conda` and provide the requirements in `environment.yaml`.

Once the environment is created, run the following command to install the local resources.

```sh
$ conda env create -f environment.yaml
$ conda activate sigsvgd
$ pip install -e .
$ pip install pytorch_lightning==2.2.0 --no-deps
```
> **NOTE**: In macOs, the following extra steps are required to install `signatory`.
>
>```sh
>export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
>export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
>export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
>MACOSX_DEPLOYMENT_TARGET=11.3.1 CC=clang CXX=clang++ pip install signatory==1.2.6.1.9.0 --no-binary signatory
>```

To run some of the examples, you'll also need to install `pybullet-planning`. Please refer to their [repository](https://github.com/caelan/pybullet-planning) for installation instructions.

# Misc.

- Rebuttal experimental seed are located at https://github.com/lubaroli/stein_mpc/releases/tag/rebuttal-experiment-result