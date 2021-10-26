## Build instruction for C++ module

The C++ module for obtaining fast gradient in **Multivariate Gaussian Distribution** and **Gaussian Mixture Model** is named as `fast_gmm_diff`.

### Requirements

- pybind11 (for building python binding )

- CMake (C++17)

### Steps

First make sure the dependencies are fully cloned with

```
$ git submodule update --init --recursive
```

Then build the module with the following. Make sure to activate your Python virtual environment first if you are using one.

```sh
$ mkdir build
$ cd build/
$ cmake ..
$ make
```

Test everything is working with

```sh
python tests/benchmark_fast_gmm_diff.py
```


