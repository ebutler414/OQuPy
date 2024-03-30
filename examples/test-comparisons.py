
import numpy as np

import oqupy
from oqupy import process_tensor


# -----------------------------------------------------------------------------
# -- Test A: Spin boson model -------------------------------------------------

# Target state
target_state_I=np.array([[0,0.0],[0.0,1.0]])

# Initial state:
initial_state_I = np.array([[1.0,0.0],[0.0,0.0]])

# Markovian dissipation
gamma_I_1 = lambda t: 0.1 # with sigma minus
gamma_I_2 = lambda t: 0.2 # with sigma z

# Ohmic spectral density with exponential cutoff
coupling_operator_I = np.array([[0.5,0.0],[0.0,-0.5]])
alpha_I = 0.3
cutoff_I = 5.0
temperature_I = 0.2

# end time
t_end_I = 1.0

# result obtained with release code (made hermitian):
rho_I = np.array([[ 0.7809559 +0.j        , -0.09456333+0.16671419j],
                  [-0.09456333-0.16671419j,  0.2190441 +0.j        ]])

correlations_I = oqupy.PowerLawSD(alpha=alpha_I,
                                  zeta=1.0,
                                  cutoff=cutoff_I,
                                  cutoff_type="exponential",
                                  temperature=temperature_I,
                                  name="ohmic")
bath_I = oqupy.Bath(coupling_operator_I,
                    correlations_I,
                    name="phonon bath") 

# Gradient obtained from release code (made Hermitian)
grad_I = []

# Time step and number of steps
dt=0.05
num_steps=int(t_end_I/dt)
    
# Parameter at each time step
x0 = np.ones(2*num_steps)
x0=list(zip(x0))

# Parameterized system definition
def discrete_h_sys_I(hx):
    return hx * 0.5* oqupy.operators.sigma('x')

system_I = oqupy.ParameterizedSystem(hamiltonian=discrete_h_sys_I,
                        gammas=[gamma_I_1, gamma_I_2],
                        lindblad_operators=[lambda t: oqupy.operators.sigma("-"),
                                            lambda t: oqupy.operators.sigma("z")])

tempo_params_I =oqupy.TempoParameters(
    dt=0.05,
    tcut=None,
    epsrel=10**(-7))

pt = oqupy.pt_tempo_compute(
    bath_I,
    start_time=0.0,
    end_time=t_end_I,
    parameters=tempo_params_I)

grad,dyn = oqupy.compute_gradient_and_dynamics(system=system_I,
                                                parameters=x0,
                                                process_tensors=[pt],
                                                initial_state=initial_state_I,
                                                target_state=target_state_I.T
                                                )
grad_dict = oqupy.state_gradient(system=system_I,
                                                parameters=x0,
                                                process_tensors=[pt],
                                                initial_state=initial_state_I,
                                                target_state=target_state_I.T
                                                )


tensor_list = [i.tensor for i in grad]

fidelity=np.sum(dyn.states[-1]*target_state_I.T)

from pprint import pprint

pprint(tensor_list[0])
pprint(grad_dict['gradient'].real)