import numpy as np

import oqupy
from oqupy import process_tensor

# -----------------------------------------------------------------------------
# -- Test C: Collective Ising Chain with different bath coupling operator

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
tempo_unique = oqupy.Tempo(
    system_C,
    bath_C,
    tempo_params_C,
    initial_state_C,
    start_time=0.0,
    unique=True)
tempo_non_unique = oqupy.Tempo(
    system_C,
    bath_C,
    tempo_params_C,
    initial_state_C,
    start_time=0.0,
    unique=False)
tempo_unique.compute(end_time=t_end_C)
tempo_non_unique.compute(end_time=t_end_C)
dyn_unique = tempo_unique.get_dynamics()
dyn_non_unique = tempo_non_unique.get_dynamics()

t,usz=dyn_unique.expectations(s_z,real=True)
t,nusz=dyn_non_unique.expectations(s_z,real=True)


fig,ax=plt.subplots(1)
ax.plot(t,usz,label='sz, Unique')
ax.plot(t,nusz,label='sz, Non-unique')
