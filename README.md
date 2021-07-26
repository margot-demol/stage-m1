# stage-m1-equinox

* *testxarray.ipynb* allow to become familiar with the x.array.sim_lab

## stagem1
Contains codes: 
* library *temporal_integration.py* process temporal integration with x.array.sim_lab to compute trajectories of drifters in waves  from an analytical velocity field. Also contains some useful functions for error diagnosis (see description*temporal_integration_library_description.md*)
* library *int_temp_integration.py* process temporal integration and spatio-temporal interpolation with x.array.sim_lab to compute trajectories of drifters in waves from a velocity spatio-temporal grid. Also contains some useful functions for error diagnosis (see description*int_temp_integration_library_description.md*)

## development
* *simple_integration.ipynb*: test to compute trajectories thanks to temporal integration with x.array.sim_lab (codes are then put in *temporal_integration.py*)
* *temporal_int_comparison.ipynb*: compare displacement error over trajectories linked with Euler, RK2 or RK4 methods and the displacement error dependency to model of currents parameters.
* *test_vel_traj.ipynb*: identify the different regimes of trajectories depending on model parameters
* *domaines_traj.ipynb*: linked with *test_vel_traj.ipynb* or *temporal_int_comparison.ipynb* results, represent the convergence domains of velocity in space $U_m/c, U_w/c$. Also allow to plot the explanatory convergence plot.
* *test_bilinear_lagrange_interpolation.ipynb*: test algorithms for bilinear and Lagrange interpolation 
* *error_inter_ana_v.ipynb*: compare the interpolated velocity to the analytic one on trajectory depending on the integration time step $\delta t$ and plot the different trajectories on the spatio-temporal grid.
* *error_inter_ana_acc.ipynb*: study the impact of period $T$, wavelength $\lambda$ over acceleration error (=differentiated acceleration-analytic acceleration) for different parameters cases and both bilinear and Lagrange interpolation.
* *error_interpolation.ipynb*: visualisation of interpolated trajectories, and study of displacement, velocity and acceleration norms depending on $\delta t=dt$
___________

# Useful links

* [x.array.sim_lab documentation](https://xarray-simlab.readthedocs.io/en/latest/create_model.html) gives a great and detailed example for running simulation with x.array.sim_lab

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

    cd stage-m1-equinox
    pip install -e .