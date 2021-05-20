# stage-m1

Contains codes that: 
*undergo temporal integration with x.array.sim_lab for drifters in waves (temporal_integration.py) 
___________

# Useful links

[x.array.sim_lab documentation](https://xarray-simlab.readthedocs.io/en/latest/create_model.html) gives a great and detailed example for running simulation with x.array.sim_lab

____________

# Installation

Download the repository:

    git clone https://github.com/margot-demol/stage-m1-equinox.git

For pre/post processing, install an appropriate conda-environment. Download Miniconda3 (i.e for python3)
from the conda website: https://docs.conda.io/en/latest/miniconda.html  and run:

    bash Miniconda3-latest-Linux-x86_64.sh
    bash
    conda update conda
    conda create -n equinox -c conda-forge python=3.9 dask-jobqueue \
                xarray zarr netcdf4 python-graphviz \
                tqdm \
                jupyterlab ipywidgets \
                cartopy geopandas descartes \
                seaborn \
                hvplot geoviews datashader nodejs \
                intake-xarray gcsfs \
                cmocean gsw \
                xhistogram \
                pytide pyinterp \
                xarray-simlab parcels
    conda activate equinox
    pip install git+https://github.com/xgcm/xrft.git
    pip install h3
    conda install -c conda-forge zstandard  # maybe not necessary with following line:
    conda install -c conda-forge fastparquet
    conda install pywavelets
    pip install git+git://github.com/psf/black


#
cd stage-m1-equinox; pip install -e .