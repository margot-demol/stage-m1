# Library stagem1.temporal_integration description 
 Allows to run a simulation with xarray-simlab which calculates trajectories (positions and velocities) of particules knowing the analytical formula of velocity using Euler or Runge Kutta (order 2 or 4) method in SetUp class. Then the Temp_Int_Comp offers to compare results of the three last integration methods.

To import: `import stagem1.temporal_integration as sti`


## Main parameters:
- `d2s` a day in seconds
- `h2s` an hours in seconds
- `w2`  default wave pulsation correponding to a 12 hours period
- `km` a kilometer in meter
- `dt` simulation time step in seconds
- `L`  wave length in meters
- `k2` wave vector

## SetUp class
### Members
- model
- in_ds
- out_ds:dataset with  position__p, position_km, displacement, velocity__v, otime, otme_day + velocity parameters

### Parameters of SetUp constructor
Default is :
- the integration method: `intmethod=Euler`
- the position initiation class: `init_p=InitPRegular`
- time parameters of the simulation: `time=list(np.arange(0,d2s*4, h2s/2)), otime=list(np.arange(0, d2s*4, h2s))`, 
- parameters for positions initiations (min, max, particules number): `init_mini=0, init_maxi=200*km, init_N=100`,
- parameters for the analytical velocity : `um=0.1, uw=0.1, w=w2, k=k2, advected=1`

You are free to change them building your object giving them to SetUp constructor: `x=sti.SetUp(um=1)` for example.
integration method `intmethod` should be among 'Euler', 'Runge_Kutta2', 'Runge_Kutta2_1' or 'Runge_Kutta4'.
`time` and `otime` list should not be the same (x array sim-lab doesn't like it)

### Methods to use on SetUp object x
- To change the integration method: `x.update_intmethod(sti.newmethod)` with new method in 'Euler', 'Runge_Kutta2', 'Runge_Kutta2_1' or 'Runge_Kutta4'
- To change parameters: `x.update_parameters(par1=..., par2=...)` with par1 and par2 in `velocity__um`, `velocity__uw`, `velocity__k`, `velocity__w`, `init_mini`, `init_maxi`, `init_N`.
- To change the clock: `x.update_clock(time=..., otime=...)`
- To plot positions: `x.print_positions(slice_step=10)` for trajectories for every ten particules,  or `x.print_positions_fac(slice_step=10)` for positions for all particules at every t spaced of ten time step.
- To plot velocities: `x.print_velocities(slice_step=10)`
- To print the advancement for some particules: `x.print_dis(slice_step=10)`
- To compute the velocity_field with `r_t*len(otime)` time point and `r_x*len(x["p"])` positions points in position and time reached by simulation: `x.velocity_field(t_t=2, r_x=2)` will return velocity dataarray with coordinates `t[day]` and `x[km]`.
- To compute the difference with the velocity out and the velocity computed from positions out via the analytical formula: `x.analytical_comparison()`
- To get positions, velocities, advancement, out times our velocity parameters: `x["p"]`,`x["p_km"]` ,`x["v"]`, `x["dis"]`,`x['otime']`... define thanks to `__getitem__`
- To get displacement dataset for one parameters varying: `x.batch_parameters(var_name, var_list)` where the changing variable name`var_name` should be among 'velocity__um', 'velocity__uw','velocity__w', 'velocity__k' and var_list the list of values the changing variable will take.


## Temp_Int_Comp class
### Members
- ds: dataset containing dis, dis_km, position, position_km, diff_dis, diff_dis_km, velocities for all methods. diff_ds is the deplacement difference with the reference (RK4 10 min)

### Parameters of Temp_Int_constructor
- `x` a `SetUp(**arg)` object
- `**args` in case parameters are not default ones

### Methods to use on Temp_int_Comp comp
- To plot deplacement in km in fonction of otime in day for all method for particules a=traj: `comp.print_diff_dis(traj=20)`
- To plot deplacement meaned on particules in km in fonction of otime in day for all method for particules a=traj: `comp.print_diff_dis_mean(traj=20)`
- To plot velocities in fonction of otime in day for all method for particules a=traj :`comp.print_diff_velocites(traj=20)`
- To plot trajectories for a particule a=traj for all method: `comp.print_traj(traj=20)`

## Dependency_ds function
`dependency_ds(list_Var, Dt, T, OT, mean_b, mean_e, list_Var_name, dim_name, **kwargs)`
Create a dataset containing the square root of the mean value over particules and over 48 hours (day 4 and 6) of the square displacement error for all methods.
### Arguments
- `list_Var`: list of list of SetUp parameters values we want to run
- `Dt`: list of simulation time step we want to run
- `T` : list of `time` list argument for the clock, whose time step should correspond to Dt list
- `OT`: list of `otime` list corresponding to `T` list
- `mean_b` time in hours of the beginning of the mean over time
- `mean_e` time in hours of the end of the mean over time
- `list_Var_name=list_Var_Name=['velocity__um', 'velocity__uw','velocity__w', 'velocity__k']`: list with variable name in dataset, in the same order they were given in list_Var
- `dim_name=['um', 'uw','w','k']` : name for out dataset dimension
- `**kwargs`: arguments to build SetUp objects in the function (exemple: k=0)

### Out 
Dataset containing the square root of the mean value over particules and over 48 hours (day 4 and 6) of the square displacement error for all methods, with coordinates `um`,`uw`,`w`, `k`, `delta_t`, `delta_t_min`, `Lambda` the wave length in km, `Ts` the period in hours.

## Dependency_ds_max function
`dependency_ds_max(list_Var, Dt, T, OT, mean_b, mean_e, list_Var_name, dim_name, **kwargs)`  
Same as dependency_ds but containing maximum absolute displacement error over both otime and particules a.
