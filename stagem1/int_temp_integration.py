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

        
        
        
#VELOCITY FIELD
def analytical_velocity_advected(t, x, um, uw, w, k):
    return um + uw*np.cos(w*t-k*(x-um*t))

def analytical_velocity_unadvected(t, x, um, uw, w, k):
    return (um + uw*np.cos(w*t-k*x))


@xs.process
class Velocity_Field:
    """Initiate the Eulerian velocity field on a grid"""
    
    #velocity parameters
    um = xs.variable(description="uniform and constant flow amplitude", attrs={"units":"m/s"})
    uw = xs.variable(description="wave amplitude", attrs={"units":"m/s"})
    w = xs.variable(description="wave pulsation", attrs={"units":"s^-1"})
    k = xs.variable(description="wave number", attrs={"units":"m⁻1"})
    advected = xs.variable(description="advected wave", attrs={"units":"1"})#booléen
    
    #grid parameters
    t_i = xs.variable(description="starting time", attrs={"units":"s"})
    t_e = xs.variable(description="ending time", attrs={"units":"s"})
    t_step = xs.variable(description="time step", attrs={"units":"s"})
    x_i = xs.variable(description="first limit of positions", attrs={"units":"m"})
    x_e = xs.variable(description="end limit of positions", attrs={"units":"m"})
    x_step = xs.variable(description="position step", attrs={"units":"m"})
    
    #analytical function
    def velocity_func(self, *args):
        if self.advected:
            return analytical_velocity_advected(*args)
        else:
            return analytical_velocity_unadvected(*args)
        
    # variables
    VF= xs.variable(dims=("t","x"), description="Eulerian velocity field",intent='out')
    x = xs.index(dims="x")
    t = xs.index(dims="t")
    
    @xs.runtime(args=["step_end"])
    def initialize(self,te):
        self.t=np.arange(self.t_i,self.t_e+self.t_step*2, self.t_step)#never reach border even with Lagrangian interpolation
        self.x=np.arange(self.x_i,self.x_e+self.x_step*2,self.x_step)
        len_t=len(self.t)
        len_x=len(self.x)
        self.VF=np.zeros((len_t, len_x))
        for i in range(len_t):
            for j in range (len_x):
                self.VF[i,j]=analytical_velocity_advected(self.t[i], self.x[j], self.um, self.uw, self.w, self.k)
    
                
#INTERPOLATION FUNCTIONS
def bilinear_int(p_liste, ts, VF, x, t, t_step, x_step):
    v=[]
    it1 = np.searchsorted(t,ts, side='right')-1
    it2 = it1 + 1
    #print(it1,it2)
    if ts!=ts:
        return np.zeros_like(p_liste)
    elif it2>=len(t):
        print('t out of velocity field :'+ str(ts))
        return np.zeros_like(p_liste)
    for p in p_liste:
        ip1 = np.searchsorted(x,p, side='right')-1
        if ip1+1>=len(x):
            print('x out of velocity field'+str(p))
            v.append(0)
        else:
            ip2 = ip1 + 1
            #print(ip1,ip2)
        
            alpha=p-x[ip1]
            beta=ts-t[it1]
        
    
            delta_vx = VF[it1,ip2]-VF[it1,ip1]
            delta_vy = VF[it2,ip1]-VF[it1,ip1]
            delta_vxy = VF[it1,ip1] + VF[it2,ip2] - VF[it1,ip2] - VF[it2,ip1]
            v.append(delta_vx*alpha/x_step + delta_vy*beta/t_step + delta_vxy*alpha*beta/x_step/t_step + VF[it1,ip1])
    return np.array(v)

def lagrange_int(p_liste,ts,VF,x,t,t_step,x_step):
    #print('lagrange')
    v=[]
    it1 = np.searchsorted(t,ts, side='right')-1
    it=[it1+o for o in [-1,0,1,2]]

    #print(it1,it2)
    if ts!=ts:
        return np.zeros_like(p_liste)
    elif it1+1>=len(t) or it1==0:
        print('t out of velocity field :'+ str(ts))
        return np.zeros_like(p_liste)
    for p in p_liste:
        ip1 = np.searchsorted(x,p, side='right')-1
        if ip1+2>=len(x) or ip1==0:
            print('x out of velocity field :'+str(p)+' '+str(ip1))
            v.append(0)
        else:
            ip=[ip1+o for o in [-1,0,1,2]]
            v_sum=0
            for i in range(4):
                for j in range(4):
                    prod1=1
                    prod2=1
                    prod3=1
                    prod4=1
                    for r in range(4):
                        if r!=i:
                            prod1*=p-x[ip[r]]
                            prod2*=x[ip[i]]-x[ip[r]]
                        if r!=j:
                            prod3*=ts-t[it[r]]
                            prod4*=t[it[j]]-t[it[r]]
                    v_sum+=VF[it[j], ip[i]]*(prod1*prod3)/(prod2*prod4)
            v.append(v_sum)
    return np.array(v)

#VELOCITY
@xs.process
class Velocity:
    """Compute the evolution of positions"""
    
    inter_method = xs.variable(description="Interpolation_method")
    v = xs.variable(dims="a", intent="out", description="velocity of particules", attrs={"units": "m"})
    
    VF = xs.foreign(Velocity_Field, 'VF')
    t = xs.foreign(Velocity_Field, 't')
    x = xs.foreign(Velocity_Field, 'x')
    t_step = xs.foreign(Velocity_Field, 't_step')
    x_step = xs.foreign(Velocity_Field, 'x_step')
    p = xs.foreign(Position, 'p')
     
    #interpolation function
    def interpolation_func(self,p, ts):
        if self.inter_method == 'bilinear':
            return bilinear_int(p, ts, self.VF,self.x,  self.t,  self.t_step, self.x_step)
        if self.inter_method == 'lagrange':
            return lagrange_int(p, ts, self.VF,self.x, self.t,   self.t_step, self.x_step)
        
    
    #INITIALISATION OF VELOCITY
    @xs.runtime(args="sim_start") #this way t is the beginning time of the simulation
    def initialize(self,ts):
        self.v = self.interpolation_func(self.p, ts)

         
    @xs.runtime(args=["step_start"])
    def run_step(self, ts):
        self.v = self.interpolation_func(self.p, ts)# v(x(t),t)
        if len(self.v)==50:
            print(ts)
        
        
    #CALCULATION OF VELOCITY AT t time and position p
    @xs.runtime(args=["step_end"])
    def finalize_step(self, te):
        self.v = self.interpolation_func(self.p, te) # v(x(t+dt),t+dt)
                        
            
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
    v = xs.foreign(Velocity, "v", intent="in")


    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        self.p_advected = self.v*dt #self.v=v(x,t)
        
        
#RUNGE KUTTA 2
@xs.process
class Runge_Kutta2:
    """Calculate positions at t+dt using Runge-Kutta method of order 2.
"""
    p_advected = xs.variable(dims="a", intent="out", groups="p_vars")
    

    inter_method = xs.foreign(Velocity, "inter_method", intent="in")
    v = xs.foreign(Velocity, "v", intent="in")
    p = xs.foreign(Position, "p", intent="in")
    VF = xs.foreign(Velocity_Field, 'VF', intent='in')
    t_step = xs.foreign(Velocity_Field, 't_step')
    x_step = xs.foreign(Velocity_Field, 'x_step')
    t = xs.foreign(Velocity_Field, 't')
    x = xs.foreign(Velocity_Field, 'x')

    #interpolation function
    def interpolation_func(self,*args):
        if self.inter_method == 'bilinear':
            return bilinear_int(*args)
        if self.inter_method == 'lagrange':
            return lagrange_int(*args)
        
        
    @xs.runtime(args=["step_delta", "step_start"])
    def run_step(self, dt,ts):
        self._p1 =self.p + self.v*dt/2
        self.p_advected = dt*self.interpolation_func(self._p1,ts+dt/2,self.VF,self.x,self.t, self.t_step, self.x_step)

#RUNGE KUTTA 2
@xs.process
class Runge_Kutta2_1:
    """Calculate positions at t+dt using Runge-Kutta method of order 2.
"""
    p_advected = xs.variable(dims="a", intent="out", groups="p_vars")
    
    inter_method = xs.foreign(Velocity, "inter_method", intent="in")
    v = xs.foreign(Velocity, "v", intent="in")
    p = xs.foreign(Position, "p", intent="in")
    VF = xs.foreign(Velocity_Field, 'VF', intent='in')
    t_step = xs.foreign(Velocity_Field, 't_step')
    x_step = xs.foreign(Velocity_Field, 'x_step')
    t = xs.foreign(Velocity_Field, 't')
    x = xs.foreign(Velocity_Field, 'x')

    #interpolation function
    def interpolation_func(self,*args):
        if self.inter_method == 'bilinear':
            return bilinear_int(*args)
        if self.inter_method == 'lagrange':
            return lagrange_int(*args)
        
        
    @xs.runtime(args=["step_delta", "step_start"])
    def run_step(self, dt,ts):
        self._p1 =self.p + self.v*dt/2
        self.p_advected = dt/2*(self.v + self.interpolation_func(self._p1,ts,self.VF,self.x,self.t, self.t_step, self.x_step))
        
        
#RUNGE KUTTA 4
@xs.process
class Runge_Kutta4:
    """Calculate positions at t+dt using Runge-Kutta method of order 4.
"""
    p_advected = xs.variable(dims="a", intent="out", groups="p_vars")
    
    inter_method = xs.foreign(Velocity, "inter_method", intent="in")
    v = xs.foreign(Velocity, "v", intent="in")
    p = xs.foreign(Position, "p", intent="in")
    VF = xs.foreign(Velocity_Field, 'VF', intent='in')
    t_step = xs.foreign(Velocity_Field, 't_step')
    x_step = xs.foreign(Velocity_Field, 'x_step')
    t = xs.foreign(Velocity_Field, 't')
    x = xs.foreign(Velocity_Field, 'x')
    

    #interpolation function
    def interpolation_func(self,*args):
        if self.inter_method == 'bilinear':
            return bilinear_int(*args)
        if self.inter_method == 'lagrange':
            return lagrange_int(*args)
        

    @xs.runtime(args=["step_delta", "step_start"])
    def run_step(self, dt,ts):
        
        t2=ts+dt/2

        self._p1 = self.p +  self.v*dt/2 
        self._v1 = self.interpolation_func(self._p1, t2,self.VF,self.x,self.t, self.t_step, self.x_step)# v(p1, t+dt/2)

        self._p2 = self.p +self._v1*dt/2 
        self._v2 = self.interpolation_func(self._p2, t2,self.VF,self.x,self.t, self.t_step, self.x_step)# v(p2, t+dt/2)

        self._p3 = self.p + self._v2*dt 
        self._v3 = self.interpolation_func(self._p3, ts+dt,self.VF,self.x,self.t, self.t_step, self.x_step)# v(p3, t+dt)

        
        self.p_advected = (self.v + self._v2*2. + self._v3 + self._v1*2.)*dt/6

        
        
#SET UP
class SetUp:
    def __init__(self,
                 intmethod=Euler,
                 init_p=InitPRegular,
                 time= list(np.arange(0,d2s*6, h2s)),
                 otime=list(np.arange(0, d2s*6-h2s, h2s)),
                 init_mini=0, init_maxi=200*km, init_N=100,
                 um=0.1, uw=0.1, w=w2, k=k2, advected=1,
                inter_method='bilinear',
                t_step=h2s,
                x_step=1*km):

        self.model= xs.Model({
            "v_field": Velocity_Field,
            "position": Position,
            "init": InitPRegular,
            "intmethod": intmethod,
            "velocity": Velocity
        })
        
        self.in_ds=xs.create_setup(model=self.model,
                      clocks={'time': time,
                                    'otime': otime, 'vf':[0]},
                      master_clock='time',
                      input_vars={'init': {'mini': init_mini, 'maxi':init_maxi, 'N':init_N},
                                    'v_field': {'um': um, 'uw': uw, 'w':w, 'k':k, 'advected':advected, 't_i':-2*t_step, 't_e':2*t_step+6*d2s, 't_step':t_step, 'x_i':init_mini-(um+uw)*6*d2s, 'x_e':init_maxi+(um+uw)*6*d2s, 'x_step':x_step},
                                  'velocity':{'inter_method': inter_method}
                                    },
                        output_vars={'position__p' : 'otime', 'velocity__v':'otime', 'v_field__VF':'vf'})
        
        self.out_ds=self.in_ds.xsimlab.run(model=self.model)
        self.add_()

   
    def add_(self): #give attrs and update adv

        self.out_ds['position_km'] = self.out_ds.position__p/km
        self.out_ds.position_km.attrs={"units":"km", "long_name":"Position"}
        
        self.out_ds['displacement'] = self.out_ds.position__p-self.out_ds.position__p.isel(otime=0)
        self.out_ds.displacement.attrs={"units":"m", "long_name":"Displacement"}
        self.out_ds.otime.attrs={"units":'s', 'long_name':'Time'}
        
        self.out_ds['displacement_km'] = (self.out_ds.position__p-self.out_ds.position__p.isel(otime=0))/km
        self.out_ds.displacement.attrs={"units":"km", "long_name":"Displacement"}

        
        otime_day = self.out_ds.otime/(24*3600)

        self.out_ds.coords['otime_day']=otime_day
        self.out_ds.otime_day.attrs={"units":"day", "long_name":"Time"}
        
        self.out_ds.a.attrs={"units":"m", "long_name":"Particule initial position"}
        
        self.out_ds['velocity_field']=self.out_ds.v_field__VF.isel(vf=0)
        self.out_ds=self.out_ds.drop('v_field__VF').drop('vf')
        self.out_ds.x.attrs={"units":"m", "long_name":"Position"}
        self.out_ds.t.attrs={"units":"s", "long_name":"Time"}
        
        self.out_ds['CFL']=(self['um']+self['uw'])*(self.out_ds.time.isel(time=1)-self.out_ds.time.isel(time=0))/self.out_ds.v_field__x_step


    
    def __getitem__(self, item):
        if item=="p":
            return self.out_ds.position__p
        if item=="p_km":
            return self.out_ds.position_km
        if item=="v":
            return self.out_ds.velocity__v
        if item=="dis":
            return self.out_ds.displacement
        if item=="dis_km":
            return self.out_ds.displacement_km
        if item=='otime':
            return self.out_ds.otime
        if item=='um':
            return float(self.out_ds.v_field__um)
        if item=='uw':
            return float(self.out_ds.v_field__uw)
        if item=='w':
            return float(self.out_ds.v_field__w)
        if item=='k':
            return float(self.out_ds.v_field__k)
        if item=='advected':
            return float(self.out_ds.v_field__advected)
        if item=='dt':
            return float(self.out_ds.v_field__t_step)
        if item=='dx':
            return float(self.out_ds.v_field__x_step)
        if item=='CFL':
            return float(self.out_ds.CFL)
        
    def update_intmethod(self,intmethod):#update processes of the model ex: change Euler->Runge Kutta: **process = intmethod=Runge_Kutta2
        
        #self.model = (self.model).update_processes(process)
        self.model= xs.Model({
            "v_field":Velocity_Field,
            "position": Position,
            "init": InitPRegular,
            "intmethod": intmethod,
            "velocity": Velocity
        })
        #CAUTION: CREATE NEW MODEL (PROBLEM WITH P_VARS OTHERWISE)

        self.out_ds= self.in_ds.xsimlab.run(model=self.model)
        self.add_()
        
        
    def update_parameters(self,**parameters):#change one or several parameters, velocity__uw ...
        self.in_ds = self.in_ds.xsimlab.update_vars(model=self.model, input_vars=parameters)
        self.in_ds = self.in_ds.xsimlab.update_vars(model=self.model, input_vars={'v_field__t_i':-2*self.in_ds.v_field__t_step, 'v_field__t_e':2*self.in_ds.v_field__t_step+6*d2s, 'v_field__x_i':self.in_ds.init__mini-(self.in_ds.v_field__um+self.in_ds.v_field__uw)*6*d2s, 'v_field__x_e':self.in_ds.init__maxi+(self.in_ds.v_field__um+self.in_ds.v_field__uw)*6*d2s})#if um or/and uw change
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
    
    def print_positions(self, slice_step=10,**kwargs):#print positions trajectories
        self.out_ds.position__p.isel(a=slice(0,None,slice_step)).plot(x="otime", hue="a",**kwargs)
        
    def print_positions_fac(self,slice_step=10,**kwargs ):#print positions trajectories
        fg=self.out_ds.isel(otime=slice(0,None,slice_step)).plot.scatter(x="a", y="position__p", marker='.', s=10,col='otime')
        for ax in fg.axes[0]:
                self.out_ds.isel(otime=0).plot.scatter(x="a", y="position__p", marker='.', c='red', s=1, ax=ax,**kwargs)
    
    def print_velocities(self, slice_step=10,**kwargs):#print velocities for different otime
        #self.out_ds.isel(otime=slice(0,None,slice_step)).plot.scatter(x="a", y="velocity__v", marker='.', s=10,col='otime')
        self.out_ds.velocity__v.isel(a=slice(0,None,slice_step)).plot(x="otime", hue="a",**kwargs)
    
    def print_dis(self, slice_step=10,**kwargs):
        self.out_ds.displacement.isel(a=slice(0,None,slice_step)).plot(x="otime", hue="a",**kwargs)
    
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
        
        
    def analytical(self):#verify model respects the analytical solution
        if self.out_ds.v_field__advected:
            _va=analytical_velocity_advected(self.out_ds.otime, self.out_ds.position__p,self.out_ds.v_field__um, self.out_ds.v_field__uw, self.out_ds.v_field__w, self.out_ds.v_field__k)
        else:
            _va=analytical_velocity_unadvected(self.out_ds.otime, self.out_ds.position__p,self.out_ds.v_field__um, self.out_ds.v_field__uw, self.out_ds.v_field__w, self.out_ds.v_field__k)
        
        return _va 
