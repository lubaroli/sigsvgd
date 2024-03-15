
# Download resources

```sh
$ make -j
```

# Install

Test on [![Python version](https://img.shields.io/badge/python-3.9.7%20-blue.svg)](https://cs.tinyiu.com/sbp-env)

```sh
$ conda create -n stein_mpc python=3.9.7
$ conda activate stein_mpc
$ pip install cython torch==1.9.0  # needs to be installed first as prerequisites
$ pip install -e .
$ pip install pytorch_lightning==2.2.0 --no-deps
```


# Misc.

- Rebuttal experimental seed are located at https://github.com/lubaroli/stein_mpc/releases/tag/rebuttal-experiment-result
