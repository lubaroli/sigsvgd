import io
import os
import re
import setuptools

here = os.path.realpath(os.path.dirname(__file__))


name = "stein_mpc"

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

url = "https://github.com/lubaroli/stein_mpc"

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

install_requires = ["torch>=1.5.0"]

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
