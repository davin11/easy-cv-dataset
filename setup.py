import os
import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

def get_version(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")
    
setup(
    name='easy-cv-dataset',
    version=get_version("easy_cv_dataset/__init__.py"),
    description='A library for dataset loading',
    url='https://github.com/davin11/easy-cv-dataset',
    author='davin11',
    author_email='davide.cozzolino@unina.it',
    long_description=README,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    packages=['easy_cv_dataset', ],
    python_requires=">=3.9",
    install_requires=['keras-cv==0.8.2', ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
)