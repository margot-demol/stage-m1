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
        self._p1 =self.p + self.v*dt/2
        self.p_advected = dt*self.velocity_func(t+dt/2, self._p1, self.um, self.uw, self.w, self.k)
        #self.p_advected = dt/2*(self.velocity_func(t+dt, self._p1, self.um, self.uw, self.w, self.k)+self.velocity_func(t, self.p, self.um, self.uw, self.w, self.k))

#RUNGE KUTTA 2
@xs.process
class Runge_Kutta2_1:
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
        self._p1 =self.p + self.v*dt/2
        #self.p_advected = dt*self.velocity_func(t+dt/2, self._p1, self.um, self.uw, self.w, self.k)
        self.p_advected = dt/2*(self.velocity_func(t+dt, self._p1, self.um, self.uw, self.w, self.k)+self.velocity_func(t, self.p, self.um, self.uw, self.w, self.k))
        
        
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
                 time= list(np.arange(0,d2s*6, h2s)),
                 otime=list(np.arange(0, d2s*6-h2s, h2s)),
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

        self.out_ds['position_km'] = self.out_ds.position__p/km
        self.out_ds.position_km.attrs={"units":"km", "long_name":"Position"}
        
        self.out_ds['displacement'] = self.out_ds.position__p-self.out_ds.position__p.isel(otime=0)
        self.out_ds.displacement.attrs={"units":"m", "long_name":"Displacement"}
        self.out_ds.otime.attrs={"units":'s', 'long_name':'Time'}
        
        otime_day = self.out_ds.otime/(24*3600)

        self.out_ds.coords['otime_day']=otime_day
        self.out_ds.otime_day.attrs={"units":"day", "long_name":"Time"}
        
        self.out_ds.a.attrs={"units":"m", "long_name":"Particule initial position"}


    
    def __getitem__(self, item):
        if item=="p":
            return self.out_ds.position__p
        if item=="p_km":
            return self.out_ds.position_km
        if item=="v":
            return self.out_ds.velocity__v
        if item=="dis":
            return self.out_ds.displacement
        if item=='otime':
            return self.out_ds.otime
        if item=='um':
            return float(self.out_ds.velocity__um)
        if item=='uw':
            return float(self.out_ds.velocity__uw)
        if item=='w':
            return float(self.out_ds.velocity__w)
        if item=='k':
            return float(self.out_ds.velocity__k)
        if item=='advected':
            return float(self.out_ds.velocity__advected)
        
    def update_intmethod(self,intmethod):#update processes of the model ex: change Euler->Runge Kutta: **process = intmethod=Runge_Kutta2
        
        #self.model = (self.model).update_processes(process)
        self.model= xs.Model({
            "position": Position,
            "init": InitPRegular,
            "intmethod": intmethod,
            "velocity": AnaVelocity
        })
        #CAUTION: CREATE NEW MODEL (PROBLEM WITH P_VARS OTHERWISE)

        self.out_ds= self.in_ds.xsimlab.run(model=self.model)
        self.add_()
        
        
    def update_parameters(self,**parameters):#change one or several parameters, velocity__uw ...
        self.in_ds = self.in_ds.xsimlab.update_vars(model=self.model, input_vars=parameters)
        self.out_ds= self.in_ds.xsimlab.run(model=self.model)
        self.add_()
    
    def update_clock(self,**clock): 
        self.in_ds = self.in_ds.xsimlab.update_clocks(model=self.model, clocks=clock)
        self.out_ds= self.in_ds.xsimlab.run(model=self.model)
        self.add_()
        
        
    def batch_parameters(self, var_name, var_list):
        in_ds_b = self.in_ds.xsimlab.update_vars(model=self.model, input_vars={var_name: ('batch', var_list)})
        out_ds_b=in_ds_b.xsimlab.run(model=self.model, batch_dim='batch')
        out_ds_b['displacement'] = out_ds_b.position__p-out_ds_b.position__p.isel(otime=0)
        out_ds_b.displacement.attrs={"units":"m", "long_name":"Displacement"}
        out_ds_b.otime.attrs={"units":'s', 'long_name':'Time'}
        
        otime_day = out_ds_b.otime/(24*3600)

        out_ds_b.coords['otime_day']=otime_day
        out_ds_b.otime_day.attrs={"units":"day", "long_name":"Time"}
        
        out_ds_b['position_km'] =out_ds_b.position__p/km
        out_ds_b.position_km.attrs={"units":"km", "long_name":"Position"}
        
        out_ds_b.a.attrs={"units":"m", "long_name":"Particule initial position"}
        return out_ds_b
    
    def print_positions(self, slice_step=10):#print positions trajectories
        self.out_ds.position__p.isel(a=slice(0,None,slice_step)).plot(x="otime", hue="a", figsize=(9,9))
        
    def print_positions_fac(self,slice_step=10 ):#print positions trajectories
        fg=self.out_ds.isel(otime=slice(0,None,slice_step)).plot.scatter(x="a", y="position__p", marker='.', s=10,col='otime')
        for ax in fg.axes[0]:
                self.out_ds.isel(otime=0).plot.scatter(x="a", y="position__p", marker='.', c='red', s=1, ax=ax)
    
    def print_velocities(self, slice_step=10):#print velocities for different otime
        #self.out_ds.isel(otime=slice(0,None,slice_step)).plot.scatter(x="a", y="velocity__v", marker='.', s=10,col='otime')
        self.out_ds.velocity__v.isel(a=slice(0,None,slice_step)).plot(x="otime", hue="a", figsize=(9,9))
    
    def print_dis(self, slice_step=10):
        self.out_ds.displacement.isel(a=slice(0,None,slice_step)).plot(x="otime", hue="a", figsize=(9,9))
    
    def velocity_func(self, *args):
        if self['advected']:
            return analytical_velocity_advected(*args)
        else:
            return analytical_velocity_unadvected(*args)    
    
    
    def velocity_field(self, r_t=2, r_x=5):
        t=self.out_ds.otime
        T=np.linspace(float(t.min('otime')),float(t.max('otime')),r_t*len(t))
        X=np.linspace(float(self['p'].min(dim=['a','otime'])),float(self['p'].max(dim=['a','otime'])),r_x*len(self["p"]))
        len_t=len(T)
        len_x=len(X)
        VF=np.zeros((len_t, len_x))
        for i in range(len_t):
            for j in range (len_x):
                VF[i,j]=self.velocity_func(T[i], X[j], self['um'], self['uw'], self['w'], self['k'])
        return xr.DataArray(data=VF, dims=["t", "x"], coords=dict(t=(["t"], T/(24*3600.)),x=(["x"], X/1000)),attrs={'units':'m/s', 'long_name':'Velocity'})
        
        
    def analytical_comparison(self):#verify model respects the analytical solution
        if self.out_ds.velocity__advected:
            _va=analytical_velocity_advected(self.out_ds.otime, self.out_ds.position__p,self.out_ds.velocity__um, self.out_ds.velocity__uw, self.out_ds.velocity__w, self.out_ds.velocity__k)
        else:
            _va=analytical_velocity_unadvected(self.out_ds.otime, self.out_ds.position__p,self.out_ds.velocity__um, self.out_ds.velocity__uw, self.out_ds.velocity__w, self.out_ds.velocity__k)
        
        return np.all(_va==self.out_ds.velocity__v)  

    
    
    
#TEMPORAL INTEGRATION COMPARISON    
class Temp_Int_Comp:
    
    def __init__(self, x, **arg):#x un objet SetUp
        
        x.update_intmethod(Euler)
        pe=x['p']
        ae=x['dis']
        ve=x['v']

        x.update_intmethod(Runge_Kutta2)
        prk2=x['p']
        ark2=x['dis']
        vrk2=x['v']
        
        x.update_intmethod(Runge_Kutta2_1)
        prk2_1=x['p']
        ark2_1=x['dis']
        vrk2_1=x['v']
        

        x.update_intmethod(Runge_Kutta4)
        prk4=x['p']
        ark4=x['dis']
        vrk4=x['v']

        x_ref=SetUp(time= list(np.arange(0,d2s*6, h2s/6)), **arg)#10 min step
        #x.update_clock(time= list(np.arange(0,d2s*6, h2s/6)))#10 min step
        x_ref.update_intmethod(Runge_Kutta4)
        p_ref=x_ref['p']
        ark4_ref=x_ref['dis']
        v_ref=x_ref['v']
        
        x_crash=SetUp(time=list(np.arange(0,d2s*6, h2s*3)),otime=list(np.arange(0,d2s*6-h2s, h2s*3)),**arg)#10 min step
        x_crash.update_intmethod(Runge_Kutta4)
        p_crash=x_crash['p']
        ark4_crash=x_crash['dis']
        v_crash=x_crash['v']

        
        self.ds=xr.concat([ae, ark2, ark2_1, ark4, ark4_crash, ark4_ref], pd.Index(["Euler", "RK2 (1)","RK2 (2)", "RK4", 'RK4 3h', "RK4 10min (reference)"], name="int_method"))
        self.ds.name='displacement'
        self.ds=self.ds.to_dataset(name='dis')

        self.ds['position']=xr.concat([pe, prk2, prk2_1, prk4, p_crash, p_ref], pd.Index(["Euler", "RK2 (1)","RK2 (2)", "RK4", 'RK4 3h', "RK4 10min (reference)"], name="int_method"))
        self.ds.position.attrs={"units":"m", "long_name":"Position"}
        
        self.ds['position_km']=self.ds.position/1000
        self.ds.position_km.attrs={"units":"km", "long_name":"Position"}
        self.ds['dis_km']=self.ds.dis/1000
        self.ds.dis_km.attrs={"units":"km", "long_name":"Displacement"}
        
        self.ds['velocities']=xr.concat([ve, vrk2,vrk2_1, vrk4, v_crash, v_ref], pd.Index(["Euler", "RK2 (1)","RK2 (2)","RK4", 'RK4 3h', "RK4 10min (reference)"], name="int_method"))
        
        aref=self.ds.sel(int_method='RK4 10min (reference)')

        Aref=xr.concat([aref, aref, aref, aref,aref,aref], pd.Index(["Euler", "RK2 (1)","RK2 (2)", "RK4", 'RK4 3h', "RK4 10min (reference)"], name="int_method"))

        self.ds['diff_dis']=(self.ds.dis-Aref.dis)
        self.ds.diff_dis.attrs={"units":"m", "long_name":"Displacement difference with reference"}

        self.ds['diff_dis_km']=self.ds.dis_km-Aref.dis_km
        self.ds.diff_dis_km.attrs={"units":"km", "long_name":"Displacement difference with reference"}

        self.ds['diff_velocities']=self.ds.velocities-Aref.velocities
        self.ds.diff_velocities.attrs={"units":"m/s", "long_name":"Velocity difference with reference"}
        
        
    
    def print_diff_dis(self, traj=20, **kwargs):
        LABEL=self.ds.int_method.values
        self.ds.diff_dis.isel(a=traj,int_method=[0,1,2,3,4]).plot(x="otime_day",hue='int_method',ls='', marker='.',markersize=6, label=LABEL[:-1],**kwargs)

        
    def print_diff_dis_mean(self, traj=20):
        LABEL=self.ds.int_method.values
        #abs(self.ds.dis_km).mean(dim='a').plot(x="otime_day", marker='.', figsize=(9,9), hue="int_method" )
        abs(self.ds.diff_dis).isel(int_method=[0,1,2,3]).mean(dim='a').plot(x="otime_day",hue='int_method',ls='', marker='.',markersize=6, label=LABEL[:-1],  figsize=(6,6))
 
        
    def print_diff_velocities(self, traj=20):
        LABEL=self.ds.int_method.values
        self.ds.diff_velocities.isel(a=traj).plot(x="otime_day",hue='int_method',ls='', marker='.',markersize=6, label=LABEL[:-1],  figsize=(6,6))
    
   
    def print_traj(self,traj=20):
        LABEL=self.ds.int_method.values
        self.ds.position.isel(a=traj,int_method=[0,1,2,3]).plot(x="otime_day",hue='int_method',ls='', marker='.',markersize=6, label=LABEL[:-1],  figsize=(6,6))
        
        
        
#DEPENDENCY COMP       
def dependency_ds(list_Var, Dt, T, OT,mean_b=96, mean_e=144,
                        list_Var_Name=['velocity__um', 'velocity__uw','velocity__w', 'velocity__k'], 
                        dim_name=['um', 'uw','w','k'],
                        **kwargs):
    selected_time=list(np.arange(mean_b,mean_e,1))    
    x_ref=SetUp(time= list(np.arange(0,d2s*6,h2s/6)), **kwargs)#10 min step 
    x_ref.update_intmethod(Runge_Kutta4)

    def batch_time(x,ad_ref): 
        list_ad=[]    
        for j in range (len(Dt)):
            x.update_clock(time=T[j], otime=OT[j])
            ds_b=x.out_ds
            dtamp=abs(ds_b.displacement-ad_ref)**2
            ad=np.sqrt(dtamp.where(dtamp.where(dtamp.otime<6*24*3600).otime>4*24*3600).mean('otime').mean('a'))
            list_ad.append(ad)
        return xr.concat(list_ad, pd.Index((Dt), name="delta_t"))

    #Velocity Variables
    for i in range (len(list_Var)):
        ds_b=x_ref.batch_parameters(list_Var_Name[i], list_Var[i])
        dref=ds_b.displacement
     
        
        x=SetUp(**kwargs)
        ds_b=x.batch_parameters(list_Var_Name[i], list_Var[i])
        dtamp=(ds_b.displacement-dref)**2
        de=np.sqrt(dtamp.where(dtamp.where(dtamp.otime<6*24*3600).otime>4*24*3600).mean('otime').mean('a'))

    
        x.update_intmethod(Runge_Kutta2)
        ds_b=x.batch_parameters(list_Var_Name[i], list_Var[i])
        dtamp=(ds_b.displacement-dref)**2
        drk2=np.sqrt(dtamp.where(dtamp.where(dtamp.otime<6*24*3600).otime>4*24*3600).mean('otime').mean('a'))
    
        x.update_intmethod(Runge_Kutta4)
        ds_b=x.batch_parameters(list_Var_Name[i], list_Var[i])
        dtamp=(ds_b.displacement-dref)**2
        drk4=np.sqrt(dtamp.where(dtamp.where(dtamp.otime<6*24*3600).otime>4*24*3600).mean('otime').mean('a'))
    


    
        ds=xr.concat([de, drk2, drk4], pd.Index(["Euler", "RK2", "RK4"], name="int_method"))
        ds=ds.assign_coords({dim_name[i]:("batch", list_Var[i])})
        ds=ds.assign_attrs(units='m')
        ds=ds.rename({'batch':dim_name[i]})
        if i==0:
            DS=ds.to_dataset(name='error_dis_'+dim_name[0], promote_attrs=True)
        else:
            DS=DS.assign({'error_dis_'+ dim_name[i]: ds})
    
    #Delta Time  
    x=SetUp(**kwargs)
    x_ref=SetUp(time= list(np.arange(0,d2s*6,h2s/6)),**kwargs)#10 min step 
    x_ref.update_intmethod(Runge_Kutta4)
    ad_ref=x_ref.out_ds.displacement
    list_dm=[batch_time(x,ad_ref)]
    x.update_intmethod(Runge_Kutta2)
    list_dm.append(batch_time(x,ad_ref))
    x.update_intmethod(Runge_Kutta4)
    list_dm.append(batch_time(x,ad_ref))
    ds=xr.concat(list_dm, pd.Index(["Euler", "RK2", "RK4"], name="int_method"))
    ds=ds.assign_attrs(units='m')
    DS=DS.assign({'error_dis_delta_time': ds})
    
    #ATTRS
    DS.um.attrs={'units':'m/s', "long_name":"mean velocity"}
    DS.uw.attrs={'units':'m/s', "long_name":"wave velocity"}
    DS.w.attrs={'units':'s⁻¹', "long_name":"wave pulsation"}
    DS.k.attrs={'units':'m⁻¹', "long_name":"wave vector"}
    DS.delta_t.attrs={'units':'s',"long_name":"simulation time step"}
    
    DS.coords['delta_t_min']=DS.delta_t/60
    DS.delta_t_min.attrs={"units":"min", "long_name":"simulation time step"}
    DS.coords['Lambda']=2*np.pi/(DS.k)/km
    DS.Lambda.attrs={"units":"km", "long_name":"wave lenght"}
    DS.coords['Ts']=2*np.pi/(DS.w)/(3600)
    DS.Ts.attrs={"units":"hours", "long_name":"wave period"}
    return DS



#DEPENDENCY COMP       
def dependency_ds_max(list_Var, Dt, T, OT,mean_b=96,mean_e=144,
                        list_Var_Name=['velocity__um', 'velocity__uw','velocity__w', 'velocity__k'], 
                        dim_name=['um', 'uw','w','k'],
                        **kwargs):
                                                                
    selected_time=list(np.arange(mean_b,mean_e,1))   
    x_ref=SetUp(time= list(np.arange(0,d2s*6,h2s/6)),**kwargs)#10 min step 
    x_ref.update_intmethod(Runge_Kutta4)

    def batch_time(x,ad_ref): 
        list_ad=[]    
        for j in range (len(Dt)):
            x.update_clock(time=T[j], otime=OT[j])
            ds_b=x.out_ds
            dtamp=abs(ds_b.displacement-ad_ref)
            ad=dtamp.max('otime').max('a')#-(ds_b.advancement-ad_ref).sel(otime=selected_time*sti.h2s).min('otime'))
            list_ad.append(ad)
        return xr.concat(list_ad, pd.Index((Dt), name="delta_t"))

    #Velocity Variables
    for i in range (len(list_Var)):
        ds_b=x_ref.batch_parameters(list_Var_Name[i], list_Var[i])
        dref=ds_b.displacement
     
        
        x=SetUp(**kwargs)
        ds_b=x.batch_parameters(list_Var_Name[i], list_Var[i])
        #de=((ds_b.displacement-dref).isel(otime=selected_time).max('otime')-(ds_b.displacement-dref).isel(otime=selected_time).min('otime')).mean('a')
        dtamp=abs(ds_b.displacement-dref)
        de=dtamp.max('otime').max('a')

    
        x.update_intmethod(Runge_Kutta2)
        ds_b=x.batch_parameters(list_Var_Name[i], list_Var[i])
        dtamp=abs(ds_b.displacement-dref)
        drk2=dtamp.max('otime').max('a')
    
        x.update_intmethod(Runge_Kutta4)
        ds_b=x.batch_parameters(list_Var_Name[i], list_Var[i])
        dtamp=abs(ds_b.displacement-dref)
        drk4=dtamp.max('otime').max('a')
    


    
        ds=xr.concat([de, drk2, drk4], pd.Index(["Euler", "RK2", "RK4"], name="int_method"))
        ds=ds.assign_coords({dim_name[i]:("batch", list_Var[i])})
        ds=ds.assign_attrs(units='m')
        ds=ds.rename({'batch':dim_name[i]})
        if i==0:
            DS=ds.to_dataset(name='error_dis_'+dim_name[0], promote_attrs=True)
        else:
            DS=DS.assign({'error_dis_'+ dim_name[i]: ds})
    
    #Delta Time  
    x=SetUp(**kwargs)
    x_ref=SetUp(time= list(np.arange(0,d2s*6,h2s/6)),**kwargs)#10 min step 
    x_ref.update_intmethod(Runge_Kutta4)
    ad_ref=x_ref.out_ds.displacement
    list_dm=[batch_time(x,ad_ref)]
    x.update_intmethod(Runge_Kutta2)
    list_dm.append(batch_time(x,ad_ref))
    x.update_intmethod(Runge_Kutta4)
    list_dm.append(batch_time(x,ad_ref))
    ds=xr.concat(list_dm, pd.Index(["Euler", "RK2", "RK4"], name="int_method"))
    ds=ds.assign_attrs(units='m')
    DS=DS.assign({'error_dis_delta_time': ds})
    
    #ATTRS
    DS.um.attrs={'units':'m/s', "long_name":"mean velocity"}
    DS.uw.attrs={'units':'m/s', "long_name":"wave velocity"}
    DS.w.attrs={'units':'s⁻¹', "long_name":"wave pulsation"}
    DS.k.attrs={'units':'m⁻¹', "long_name":"wave vector"}
    DS.delta_t.attrs={'units':'min',"long_name":"simulation time step"}
    
    DS.coords['delta_t_min']=DS.delta_t/60
    DS.delta_t_min.attrs={"units":"min", "long_name":"simulation time step"}
    DS.coords['Lambda']=2*np.pi/(DS.k)/km
    DS.Lambda.attrs={"units":"km", "long_name":"wave lenght"}
    DS.coords['Ts']=2*np.pi/(DS.w)/(3600)
    DS.Ts.attrs={"units":"hours", "long_name":"wave period"}
    return DS

