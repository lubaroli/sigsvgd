import io
import os
import re
import setuptools

here = os.path.realpath(os.path.dirname(__file__))


name = "sigsvgd"

# we actually store the version in the __version__ attribute in the source
with io.open(os.path.join(here, name, "__init__.py")) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

author = "Lucas Barcelos"

author_email = "lucas.barcelos@sydney.edu.au"

description = (
    "Planning and Control for Robotic environments using Stein Variational "
    "MPC with diverse kernels and model inference. "
)

with io.open(os.path.join(here, "README.md"), "r", encoding="utf-8") as f:
    readme = f.read()

url = "https://github.com/lubaroli/sigsvgd"

license = "GPL-3"

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Information Technology",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: Unix",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering :: Mathematics",
]

python_requires = ">=3.5, <4"

install_requires = [
    "torch==1.9.0",
    "signatory==1.2.6.1.9.0",
    "plotly>=5",
    "pandas",
    "kaleido",
    "altair",
    "seaborn",
    "numpy-quaternion",
    "cython",
    "matplotlib",
    # "pytorch_lightning==2.2.0",
    "differentiable-robot-model @ git+https://github.com/soraxas/differentiable-robot-model@improved-performance#egg=differentiable-robot-model-0.1.1-6",
    "pybullet-planning @ git+https://github.com/soraxas/pybullet-planning@f580bad01479d657b9ee549b440e33706b10318d#egg=pybullet-planning-0.0.1",
    "torchcubicspline @ git+https://github.com/patrick-kidger/torchcubicspline@d16c6bf5b63d03dbf2977c70e19a320653b5e4a8#egg=torchcubicspline-0.0.3",
    "sigkernel @ git+https://github.com/crispitagorico/sigkernel@3b2373982e12b3d499a80228311a04debcc1bea1#egg=sigkernel-0.0.1",
    "gpytorch<=1.11",
    "tqdm",
]

setuptools.setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    long_description=readme,
    url=url,
    license=license,
    classifiers=classifiers,
    zip_safe=False,
    python_requires=python_requires,
    install_requires=install_requires,
    packages=[name],
)
