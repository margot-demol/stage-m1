import numpy as np
import xarray as xr
import pandas as pd

from matplotlib import pyplot as plt

import xsimlab as xs

#MAIN PARAMETERS
d2s=86400 #24h in s
h2s=3600  #1h in s
w2=2*2*np.pi/86400 #wave pulsation
km=1e3
dt=1*h2s # model step
L=100*km
k2=2*np.pi/L




#POSITION
@xs.process
class Position:
    """Compute the evolution of positions"""

    p_vars = xs.group("p_vars")
    p = xs.variable(dims="a", intent="inout", description="positions of particules", attrs={"units": "m", "long_name":"Positions"})
    
    def run_step(self):
        self._delta_p = sum((x for x in self.p_vars))
        
    def finalize_step(self):
        self.p += self._delta_p #p(t+dt)

        
        
        
#VELOCITY
def analytical_velocity_advected(t, x, um, uw, w, k):
    return um + uw*np.cos(w*t-k*(x-um*t))

def analytical_velocity_unadvected(t, x, um, uw, w, k):
    return (um + uw*np.cos(w*t-k*x))

@xs.process
class AnaVelocity:
    """Calculate velocity at t and in all positions.
    """
    
    # parameters
    um = xs.variable(description="uniform and constant flow amplitude", attrs={"units":"m/s"})
    uw = xs.variable(description="wave amplitude", attrs={"units":"m/s"})
    w = xs.variable(description="wave pulsation", attrs={"units":"s^-1"})
    
    k = xs.variable(description="wave number", attrs={"units":"m⁻1"})
    advected = xs.variable(description="advected wave", attrs={"units":"1"})#booléen
    
    # variables
    v = xs.variable(dims="a", intent="out", description="velocity of particules", attrs={"units": "m/s", "long_name":"Velocities"})
    p = xs.foreign(Position, "p", intent="in")
    
    def velocity_func(self, *args):
        if self.advected:
            return analytical_velocity_advected(*args)
        else:
            return analytical_velocity_unadvected(*args)
        
    #velocity initialisation
    @xs.runtime(args="sim_start") #this way t is the beginning time of the simulation
    def initialize(self,t):
        self.v = self.velocity_func(t, self.p, self.um, self.uw, self.w, self.k)
    
    
    @xs.runtime(args=["step_start"])
    def run_step(self, t):
         self.v = self.velocity_func(t, self.p, self.um, self.uw, self.w, self.k)  # v(x(t),t)
        
        
    #velocity t time and position p
    @xs.runtime(args=["step_end"])
    def finalize_step(self, t):
        self.v = self.velocity_func(t, self.p, self.um, self.uw, self.w, self.k)  # v(x(t+dt),t+dt)
        
    
    
    
#POSITIONS REGULAR INITIALISATION 
@xs.process
class InitPRegular:
    """Initialize `positions` profile with N regular values in a giving an interval."""

    mini = xs.variable(description="minimum for initial position", static=True)
    maxi = xs.variable(description="maximum for initial position", static=True)
    N = xs.variable(description="number of particules", static=True)
    
    a = xs.index(dims="a")

    p = xs.foreign(Position, "p", intent="out")
    
    def initialize(self):
        self.a = np.linspace(self.mini, self.maxi, self.N)
        self.p = self.a.copy()

        
        
#EULER
@xs.process
class Euler:
    """Calculate positions at t+dt using Euler method.
"""
    p_advected = xs.variable(dims="a", intent="out", groups="p_vars")
    v = xs.foreign(AnaVelocity, "v", intent="in")

    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        self.p_advected = self.v*dt #self.v=v(x,t)
        
        
#RUNGE KUTTA 2
@xs.process
class Runge_Kutta2:
    """Calculate positions at t+dt using Runge-Kutta method of order 2.
"""
    p_advected = xs.variable(dims="a", intent="out", groups="p_vars")

    v = xs.foreign(AnaVelocity, "v", intent="in")
    p = xs.foreign(Position, "p", intent="in")
    
    #parameters
    advected = xs.foreign(AnaVelocity, "advected", intent="in")
    um = xs.foreign(AnaVelocity, "um")
    uw = xs.foreign(AnaVelocity, "uw")
    w = xs.foreign(AnaVelocity, "w")
    k = xs.foreign(AnaVelocity, "k")
    
    def velocity_func(self, *args):
        if self.advected:
            return analytical_velocity_advected(*args)
        else:
            return analytical_velocity_unadvected(*args)
    
    @xs.runtime(args=["step_delta", "step_start"])
    def run_step(self, dt,t):
        self._p1 =self.p + self.v*dt
        self.p_advected = dt/2*(self.v + self.velocity_func(t, self._p1, self.um, self.uw, self.w, self.k))

        
        
#RUNGE KUTTA 4
@xs.process
class Runge_Kutta4:
    """Calculate positions at t+dt using Runge-Kutta method of order 4.
"""
    p_advected = xs.variable(dims="a", intent="out", groups="p_vars")

    v = xs.foreign(AnaVelocity, "v", intent="in")
    p = xs.foreign(Position, "p", intent="in")
    
    #parameters
    advected = xs.foreign(AnaVelocity, "advected", intent="in")
    um = xs.foreign(AnaVelocity, "um")
    uw = xs.foreign(AnaVelocity, "uw")
    w = xs.foreign(AnaVelocity, "w")
    k = xs.foreign(AnaVelocity, "k")

    def velocity_func(self, *args):
        if self.advected:
            return analytical_velocity_advected(*args)
        else:
            return analytical_velocity_unadvected(*args)
    
    @xs.runtime(args=["step_delta", "step_start"])
    def run_step(self, dt,t):
        
        t2=t+dt/2

        self._p1 = self.p + dt/2 * self.v
        self._v1 = self.velocity_func(t2, self._p1, self.um, self.uw, self.w, self.k)# v(p1, t+dt/2)
        
        self._p2 = self.p + dt/2 * self._v1 
        self._v2 = self.velocity_func(t2, self._p2, self.um, self.uw, self.w, self.k)# v(p2, t+dt/2)
        
        self._p3 = self.p + dt * self._v2
        self._v3 = self.velocity_func(t+dt, self._p3, self.um, self.uw, self.w, self.k)# v(p3, t+dt)
        
        
        self.p_advected = dt/6 *( self.v + 2*self._v2 + self._v3 + 2*self._v1)

        
        
#SET UP
class SetUp:
    def __init__(self,
                 intmethod=Euler,
                 init_p=InitPRegular,
                 time= list(np.arange(0,d2s*4, h2s/2)),
                 otime=list(np.arange(0, d2s*4, h2s)),
                 init_mini=0, init_maxi=200*km, init_N=100,
                 um=0.1, uw=0.1, w=w2, k=k2, advected=1):

        self.model= xs.Model({
            "position": Position,
            "init": init_p,
            "intmethod": intmethod,
            "velocity": AnaVelocity
        })
        self.in_ds=xs.create_setup(model=self.model,
                            clocks={'time': time,
                                    'otime': otime}, 
                        master_clock='time',
                        input_vars={'init': {'mini': init_mini, 'maxi':init_maxi, 'N':init_N},
                                    'velocity': {'um': um, 'uw': uw, 'w':w, 'k':k, 'advected':advected},
                                    },
                        output_vars={'position__p' : 'otime','velocity__v' : 'otime'})
        
        self.out_ds=self.in_ds.xsimlab.run(model=self.model)
        self.add_()

   
    def add_(self): #give attrs and update adv

        self.out_ds['advancement'] = self.out_ds.position__p-self.out_ds.position__p.isel(otime=0)
        self.out_ds.advancement.attrs={"units":"m", "long_name":"Advancement"}
        self.out_ds.otime.attrs={"units":'s', 'long_name':'Time'}
        
        otime_day = self.out_ds.otime/(24*3600)

        self.out_ds.coords['otime_day']=otime_day
        self.out_ds.otime_day.attrs={"units":"day", "long_name":"Time"}
        
        self.out_ds.a.attrs={"units":"m", "long_name":"Particule initial position"}


    
    def __getitem__(self, item):
        if item=="p":
            return self.out_ds.position__p
        if item=="v":
            return self.out_ds.velocity__v
        if item=="adv":
            return self.out_ds.advancement
        if item=='otime':
            return self.out_ds.otime
        
        
    def update_model(self,**process):#update processes of the model ex: change Euler->Runge Kutta: **process = intmethod=Runge_Kutta2
        self.model = (self.model).update_processes(process)
        self.out_ds= self.in_ds.xsimlab.run(model=self.model)
        self.add_()
            
    def update_parameters(self,**parameters):#change one or several parameters 
        self.in_ds = self.in_ds.xsimlab.update_vars(model=self.model, input_vars=parameters)
        self.out_ds= self.in_ds.xsimlab.run(model=self.model)
        self.add_()
    
    def print_positions(self, slice_step=10):#print positions trajectories
        self.out_ds.position__p.isel(a=slice(0,None,slice_step)).plot(x="otime", hue="a", figsize=(9,9))
        
    def print_positions_fac(self,slice_step=10 ):#print positions trajectories
        fg=self.out_ds.isel(otime=slice(0,None,slice_step)).plot.scatter(x="a", y="position__p", marker='.', s=10,col='otime')
        for ax in fg.axes[0]:
                self.out_ds.isel(otime=0).plot.scatter(x="a", y="position__p", marker='.', c='red', s=1, ax=ax)
    
    def print_velocities(self, slice_step=10):#print velocities for different otime
        self.out_ds.isel(otime=slice(0,None,slice_step)).plot.scatter(x="a", y="velocity__v", marker='.', s=10,col='otime')
    
    def print_adv(self, slice_step=10):
        self.out_ds.advancement.isel(a=slice(0,None,slice_step)).plot(x="otime", hue="a", figsize=(9,9))

    
    def analytical_comparison(self):#verify model respects the analytical solution
        if self.out_ds.velocity__advected:
            _va=analytical_velocity_advected(self.out_ds.otime, self.out_ds.position__p,self.out_ds.velocity__um, self.out_ds.velocity__uw, self.out_ds.velocity__w, self.out_ds.velocity__k)
        else:
            _va=analytical_velocity_unadvected(self.out_ds.otime, self.out_ds.position__p,self.out_ds.velocity__um, self.out_ds.velocity__uw, self.out_ds.velocity__w, self.out_ds.velocity__k)
        
        return np.all(_va==self.out_ds.velocity__v)  

    
#TEMPORAL INTEGRATION COMPARISON    
class Temp_Int_Comp:
    
    def __init__(self, x):#x un objet SetUp

        ae2=x['adv']**2
        ve=x['v']

        x.update_model(intmethod=Runge_Kutta2)
        ark22=x['adv']**2
        vrk2=x['v']

        x.update_model(intmethod=Runge_Kutta4)
        ark42=x['adv']**2
        vrk4=x['v']

        x_ref=SetUp(time= list(np.arange(0,d2s*4, h2s/6)))#10 min step
        x_ref.update_model(intmethod=Runge_Kutta4)
        ark42_ref=x_ref['adv']**2
        v_ref=x_ref['v']

        self.ds=xr.concat([ae2, ark22, ark42, ark42_ref], pd.Index(["Euler", "RK2", "RK4", "Reference"], name="int_method"))
        self.ds.name='square advancement'
        self.ds=self.ds.to_dataset(name='square_adv')

        self.ds['square_adv_km']=self.ds.square_adv/1000000
        self.ds.square_adv_km.attrs={"units":"km²", "long_name":"Square advancement"}
        self.ds['velocities']=xr.concat([ve, vrk2, vrk4, v_ref], pd.Index(["Euler", "RK2", "RK4", "Reference"], name="int_method"))
        
        aref=self.ds.sel(int_method='Reference')

        Aref=xr.concat([aref, aref, aref, aref], pd.Index(["Euler", "RK2", "RK4", "Reference"], name="int_method"))

        self.ds['diff_sqr_adv']=self.ds.square_adv-Aref.square_adv
        self.ds.diff_sqr_adv.attrs={"units":"m²", "long_name":"Square advancement difference with reference"}

        self.ds['diff_sqr_adv_km']=self.ds.square_adv_km-Aref.square_adv_km
        self.ds.diff_sqr_adv_km.attrs={"units":"km²", "long_name":"Square advancement difference with reference"}

        self.ds['diff_velocities']=self.ds.velocities-Aref.velocities
        self.ds.diff_velocities.attrs={"units":"m/s", "long_name":"Velocity difference with reference"}

    def print_diff_sqr_adv(self, traj=20):
        self.ds.square_adv_km.isel(a=traj, otime=np.arange(75,95)).plot(x="otime_day", marker='.', figsize=(15,15), hue="int_method" )
        self.ds.diff_sqr_adv_km.isel(a=traj, otime=np.arange(0,95), int_method=[0,1,2]).plot(marker='.',hue="int_method", figsize=(9,9))
        self.ds.diff_sqr_adv.isel(a=traj, otime=np.arange(0,95), int_method=2).plot(marker='.',hue="int_method", figsize=(9,9))
        
    def print_diff_velocities(self, traj=20):
        self.ds.diff_velocities.isel(a=traj, otime=np.arange(0,95), int_method=[0,1,2]).plot(marker='.',hue="int_method", figsize=(9,9))
        self.ds.diff_velocities.isel(a=traj, otime=np.arange(0,95), int_method=2).plot(marker='.',hue="int_method", figsize=(9,9))
    
    def __getitem__(self, item):
        if item=='ds':
            return self.ds