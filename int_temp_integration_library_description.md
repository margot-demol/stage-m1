# Library stagem1.int_temp_integration description 
 Allows to run a simulation with xarray-simlab which calculates trajectories (positions and velocities) of particules knowing velocity on a grid(x,t) using Euler or Runge Kutta (order 2 or 4) method and bilinear or Lagrangian interpolation in SetUp class.
To import: `import stagem1.int_temp_integration as iti`


## Main parameters:
- `d2s` a day in seconds
- `h2s` an hours in seconds
- `w2`  default wave pulsation correponding to a 12 hours period
- `km` a kilometer in meter
- `dt` simulation time step in seconds
- `L`  wave length in meters
- `k2` wave vector corresponding to an 100 km wavelength

## SetUp class
### Members
- model
- in_ds
- out_ds:dataset with  position__p, position_km, displacement, velocity__v, otime, otime_day, VF + velocity parameters

### Parameters of SetUp constructor
Default is :
- the integration method: `intmethod=Euler`
- the position initiation class: `init_p=InitPRegular`
- time parameters of the simulation: `time=list(np.arange(0,d2s*4, h2s/2)), otime=list(np.arange(0, d2s*4, h2s))`, 
- parameters for positions initiations (min, max, particules number): `init_mini=0, init_maxi=200*km, init_N=100`,
- parameters for the velocity field : `um=0.1, uw=0.1, w=w2, k=k2, advected=1` and for the grid: `t_step=h2s` and `x_step=1*km`


You are free to change them building your object giving them to SetUp constructor: `x=sti.SetUp(um=1)` for example.
integration method `intmethod` should be among 'Euler', 'Runge_Kutta2', 'Runge_Kutta2_1' or 'Runge_Kutta4'.
`time` and `otime` list should not be the same (x array sim-lab doesn't like it)
and `inter_method` among 'bilinear' or lagrange'

### Methods to use on SetUp object x
- To change the integration method: `x.update_intmethod(sti.newmethod)` with new method in 'Euler', 'Runge_Kutta2', 'Runge_Kutta2_1' or 'Runge_Kutta4'
- To change parameters: `x.update_parameters(par1=..., par2=...)` with par1 and par2 in `v_field__inter_method`, `v_field__um`, `v_field__uw`, `v_field__k`, `v_field__w`,`v_field__t_step`,`v_field__x_step`, `init_mini`, `init_maxi`, `init_N`.
- To change the clock: `x.update_clock(time=time_list, otime=otime_list)`
- To plot positions: `x.print_positions(slice_step=10)` for trajectories for every ten particules,  or `x.print_positions_fac(slice_step=10)` for positions for all particules at every t spaced of ten time step.
- To plot velocities: `x.print_velocities(slice_step=10)`
- To print the advancement for some particules: `x.print_dis(slice_step=10)`
- To compute the difference with the velocity out and the velocity computed from positions out via the analytical formula: `x.analytical_comparison()`
- To get positions, velocities, advancement, out times our velocity parameters: `x["p"]`,`x["p_km"]` ,`x["v"]`, `x["dis"]`,`x['otime']`... define thanks to `__getitem__`
- To get displacement dataset for one parameters varying: `x.batch_parameters(var_name, var_list)` where the changing variable name`var_name` should be among '`v_field__um', 'v_field__uw','v_field__w', 'v_field__k' and var_list the list of values the changing variable will take.
- To get the analytical velocity dataset for all drifters and for all times: `x.analytical()`
- To get the analytical acceleration over trajectories dataset for all drifters and for all times: `x.analytical_acc()`


# The `run_DT(DT,Tmax=24,a=5, xlim=None, **args)` function 

Arguments:
- `DT`: list of lagrangian time step $\delta t$ we want to explore
- `Tmax`: time in hours of the global simulation
- `xlim`: time axe limit
- `a` the eulerian coordinate of the drifter we want to study (should be <100, default is five) 
- `**args`: arguments for iti.SetUp    

Out: pour chaque $\delta t$ en colonne
* First line: interpolated and analytic velocity over the trajectory 
* Second line: difference between these two velocities
* Third line: distance from drifter position to the nearest spatio-temporal grid point (=minimum between x['p']%x['dx']/dx et x['dx']-x['p']%x['dx']/dx)
* Fourth line: time gap between current time and time of the nearest point of the spatio-temporal grid. (=minimum between x['otime']%x['dt']/dt et x['dt']-x['otime']%x['dt']/dt)
* Fifth line: multiplication of interpolated velocity (less mean velocity)  and time gap from the grid.

Default fixed parameters are:  $U_m=U_w=0.1 m/s$, $dx=1km$, $dt=1h$, w=iti.w2 (period 12h) et k=iti.k2 (wavelength 100km)  


