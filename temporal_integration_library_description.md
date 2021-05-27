# Library stagem1.temporal_integration description 
 Allows to run a simulation with xarray-simlab which calculates trajectories (positions and velocities) of particules knowing the analytical formula of velocity using Euler or Runge Kutta (order 2 or 4) method in SetUp class. Then the Temp_Int_Comp offers to compare results of the three last integration methods.

To import: `import stagem1.temporal_integration as sti`

### SetUp class
#### Members
- model
- in_ds
- out_ds

#### Parameters of SetUp constructor
Default is :
- the integration method: `intmethod=Euler`
- the position initiation class: `init_p=InitPRegular`
- time parameters of the simulation: `time=list(np.arange(0,d2s*4, h2s/2)), otime=list(np.arange(0, d2s*4, h2s))`, 
- parameters for positions initiations (min, max, particules number): `init_mini=0, init_maxi=200*km, init_N=100`,
- parameters for the analytical velocity : `um=0.1, uw=0.1, w=w2, k=k2, advected=1`

You are free to change them building your object giving them to SetUp constructor: `x=sti.SetUp(um=1)` for example.

#### Methods to use
- To change the integration method: `update_model(intmethod=sti.newmethod)` with new method in 'Euler', 'Runge_Kutta2' or 'Runge_Kutta4'
- To change parameters: `update_parameters(par1=..., par2=...)` with par1 and par2 in um, uw, k, w, init_mini, init_maxi, init_N.
- To print positions: `print_positions(slice_step=10)` for trajectories for every ten particules,  or `print_positions_fac(slice_step=10)` for positions for all particules at every t spaced of ten time step.
- To print the advancement for some particules: `print_adv(slice_step=10)`
- To compute the difference with the velocity out and the velocity computed from positions out via the analytical formula: `analytical_comparison()`
- To get positions, velocities, advancement or out times: `x["p"]`, `x["v"]`, `x["adv"]` or `x['otime']`


### Temp_Int_Comp class

