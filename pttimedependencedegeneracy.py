import numpy as np
import matplotlib.pyplot as plt
import oqupy
from oqupy import process_tensor

# -----------------------------------------------------------------------------
# -- Test C: Collective Ising Chain with different bath coupling operator

testtempo=True
testpttempo=True

# Initial state:
initial_state_C = np.array([[1.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0]])

# System operator
j_value = -1.0
h = 1.0
s_z = np.array([[1.0,0.0,0.0],
                [0.0,0.0,0.0],
                [0.0,0.0,-1.0]])
s_x = np.array([[0.0,1.0,0.0],
                [1.0,0.0,1.0],
                [0.0,1.0,0.0]])
h_sys_C = (j_value/2) * s_z @ s_z + h * s_x

# Ohmic spectral density with exponential cutoff
coupling_operator_C = np.diag([1,1,2])
alpha_C = 0.3
cutoff_C = 5.0
temperature_C = 0.2

# end time
t_end_C = 5.0
dt=0.05

numsteps=int(t_end_C/dt)
times=np.arange(numsteps)*dt
lamconst=2
def smswitch(t):
    if t<=t_end_C:
        return (0.1+0.9*(t**lamconst/(t**(lamconst)+(t_end_C-t)**(lamconst))))
    else:
        return 1.0
smv=np.vectorize(smswitch)
alpharamp=smv(times)
#alpharamp=np.ones(len(times))

correlations_C = oqupy.PowerLawSD(alpha=alpha_C,
                                  zeta=1.0,
                                  cutoff=cutoff_C,
                                  cutoff_type="exponential",
                                  temperature=temperature_C,
                                  name="ohmic")
bath_C = oqupy.Bath(coupling_operator_C,
                    correlations_C,
                    name="bath with north degeneracies")
system_C = oqupy.System(h_sys_C)

tempo_params_C = oqupy.TempoParameters(
    dt=0.05,
    tcut=None,
    epsrel=10**(-7))


if testtempo:
    
    tempo_unique = oqupy.Tempo(
        system_C,
        bath_C,
        tempo_params_C,
        initial_state_C,
        start_time=0.0,
        unique=True,
        alpha_t=alpharamp)

    tempo_non_unique = oqupy.Tempo(
        system_C,
        bath_C,
        tempo_params_C,
        initial_state_C,
        start_time=0.0,
        unique=False,
        alpha_t=alpharamp)
    tempo_unique.compute(end_time=t_end_C)
    tempo_non_unique.compute(end_time=t_end_C)
    dyn_unique = tempo_unique.get_dynamics()
    dyn_non_unique = tempo_non_unique.get_dynamics()

    t,usz=dyn_unique.expectations(s_z,real=True)
    t,nusz=dyn_non_unique.expectations(s_z,real=True)
    t,usx=dyn_unique.expectations(s_x,real=True)
    t,nusx=dyn_non_unique.expectations(s_x,real=True)

    fig,ax=plt.subplots(1)
    ax.plot(t,usz,label='sz, Unique')
    ax.plot(t,nusz,'o',label='sz, Non-unique')

    ax.plot(t,usx,label='sx, Unique')
    ax.plot(t,nusx,'x',label='sx, Non-unique')

    ax.legend()
    plt.show()

if testpttempo:
    pttempo_unique = oqupy.PtTempo(
        bath_C,
        parameters=tempo_params_C,
        start_time=0.0,
        end_time=t_end_C,
        unique=True,
        alpha_t=alpharamp)
    pttempo_non_unique = oqupy.PtTempo(
        bath_C,
        parameters=tempo_params_C,
        start_time=0.0,
        end_time=t_end_C,
        unique=False,
        alpha_t=alpharamp)


    pttempo_unique.compute()
    pttempo_non_unique.compute()


    pttuni=pttempo_unique.get_process_tensor()
    pttnuni=pttempo_non_unique.get_process_tensor()



    dynuni=oqupy.compute_dynamics(
        process_tensor=pttuni,        
        system=system_C,
        initial_state=initial_state_C,
        start_time=0)

    dynnuni=oqupy.compute_dynamics(
        process_tensor=pttnuni,        
        system=system_C,
        initial_state=initial_state_C,
        start_time=0)



    t,usz=dynuni.expectations(s_z,real=True)
    t,nusz=dynnuni.expectations(s_z,real=True)

    t,usx=dynuni.expectations(s_x,real=True)
    t,nusx=dynnuni.expectations(s_x,real=True)


    fig,ax=plt.subplots(1)
    ax.plot(t,usz,label='sz, Unique')
    ax.plot(t,nusz,'o',label='sz, Non-unique')

    ax.plot(t,usx,label='sx, Unique')
    ax.plot(t,nusx,'x',label='sx, Non-unique')

    ax.legend()
    plt.show()


