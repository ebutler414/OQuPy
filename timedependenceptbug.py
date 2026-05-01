# %%
import oqupy
import oqupy.operators as op
import scipy.integrate as spi
import numpy as np
from oqupy.iTEBD_TEMPO_useoqupybath import iTEBD_TEMPO_oqupy
from oqupy.process_tensor import TTInvariantProcessTensor
from oqupy.tti_tempo import TTITempo
from oqupy.tti_tempo import TTITempoCounting
import matplotlib.pyplot as plt


from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize,Bounds

pt_parameters = {'epsrel':10**(-7),
                 'alpha':0.1,
                 'omega_cutoff':1,                 
                 'temp':0.131,
                 'dt':0.2} #0.25

omega_cutoff = pt_parameters['omega_cutoff']
alpha = pt_parameters['alpha']
temperature = pt_parameters['temp']
epsrel = pt_parameters['epsrel']
# dt = 1./omega_cutoff/np.sqrt(3)
dt=pt_parameters['dt']


# spectral density (without cutoff)
def j(w):
    return 2*alpha*w


# %%

correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=1,
                                cutoff=omega_cutoff,
                                cutoff_type='exponential',
                                temperature=temperature)
bath = oqupy.Bath(op.sigma("z")/2.0, correlations)
parameters=oqupy.TempoParameters(dt=dt,epsrel=epsrel,dkmax=None)

# %%
# smooth switching function
lamconst=2
tf=30
numsteps=int(tf/dt)
times=np.arange(numsteps)*dt
def smswitch(t):
    if t<=tf:
        return (0.1+0.9*(t**lamconst/(t**(lamconst)+(tf-t)**(lamconst))))
    else:
        return 1.0
smv=np.vectorize(smswitch)
alpharamp=smv(times)

# %%
alpharamp=np.ones(len(times))
pttempotestswitch=oqupy.pt_tempo_compute(bath=bath,
                             start_time=0,
                             end_time=tf,
                             parameters=parameters,
                             alpha_t=alpharamp)


# %%
splitting=1.0
rhoini=op.spin_dm('x-')
system=oqupy.System(splitting*op.sigma('x')/2)
dynamicspttestswitch=oqupy.compute_dynamics(
    process_tensor=pttempotestswitch,        
    system=system,
    initial_state=rhoini,
    start_time=0)


# %%

states=dynamicspttestswitch.states
rescale=False
if rescale:
    states=states/(states[0].trace())
dynamicspttestswitch._states=states
t2,sx2=dynamicspttestswitch.expectations(op.sigma('x'),real=True)
#t3,sx3=dynamicspttestswitch2.expectations(op.sigma('x'),real=True)

runtempo=True
if runtempo:
    tempores=oqupy.tempo_compute(system=system,
                                bath=bath,
                                initial_state=rhoini,
                                start_time=0.0,
                                end_time=tf,
                                alpha_t=alpharamp,
                                parameters=parameters)

    t3,sx3=tempores.expectations(op.sigma('x'),real=True)

fig,ax=plt.subplots(1)
#ax.plot(t,sx,'o')
ax.plot(t2,sx2,label='PT-TEMPO')
if runtempo:
    ax.plot(t3,sx3,label='TEMPO')
#ax2=ax.twinx()
#ax2.plot(t,alpha_tramp)
ax.legend()
plt.show()



# %%
