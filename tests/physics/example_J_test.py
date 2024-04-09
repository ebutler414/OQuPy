# Copyright 2020 The TEMPO Collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for the compute_gradient_and_dynamics and chain_rule functions in gradient.py
"""

import pytest
import numpy as np

import oqupy
from oqupy import process_tensor


# -----------------------------------------------------------------------------
# -- Test J: Spin-Boson model gradient (time-dependent system, functional target state) -------------------------------------------------

# Target derivative : e.g. derivative of the purity
target_derivative_J=lambda rho: 2*rho.T

# Initial state (mixed):
initial_state_J = np.array([[1.0,0.0],[0.0,1.0]])

# End time
t_end_J = 1.0

# Time step and number of steps
dt=0.05
num_steps=int(t_end_J/dt)
    
# Parameter at each time step
x0 = dt/2*np.arange(2*num_steps)
x0=list(zip(x0))

# Markovian dissipation
gamma_J_1 = lambda t: 0.1*t # with sigma minus
gamma_J_2 = lambda t: 0.2*t # with sigma z

# Ohmic spectral density with exponential cutoff
coupling_operator_J = np.array([[0.5,0.0],[0.0,-0.5]])
alpha_J = 0.3
cutoff_J = 5.0
temperature_J = 0.2

# Result obtained with release code (made hermitian):
rho_J = np.array([[ 0.95386881-5.06517571e-15j, -0.11612426-8.49192388e-03j],
              [-0.11612426+8.49192388e-03j,  1.04611851-5.85469173e-17j]])

correlations_J = oqupy.PowerLawSD(alpha=alpha_J,
                                  zeta=1.0,
                                  cutoff=cutoff_J,
                                  cutoff_type="exponential",
                                  temperature=temperature_J,
                                  name="ohmic")
bath_J = oqupy.Bath(coupling_operator_J,
                    correlations_J,
                    name="phonon bath") 

# Parameterized Hamiltonian definition
def discrete_h_sys_J(hx):
    return 0.5*hx * oqupy.operators.sigma('x')

system_J = oqupy.ParameterizedSystem(hamiltonian=discrete_h_sys_J,
                        gammas=[gamma_J_1, gamma_J_2],
                        lindblad_operators=[lambda t: oqupy.operators.sigma("-"),
                                            lambda t: oqupy.operators.sigma("z")])

# Derivative of F(T) w.r.t. hx(t) obtained from release code
grad_params_J= [[ 0.00042864], [ 0.00090963], [ 0.00090963], [ 0.00135798], [ 0.00135798], 
                [ 0.00174146], [ 0.00174146], [ 0.00205895], [ 0.00205895], [ 0.00232296], 
                [ 0.00232296], [ 0.00254775], [ 0.00254775], [ 0.00274492], [ 0.00274492], 
                [ 0.00292292], [ 0.00292292], [ 0.00308704], [ 0.00308704], [ 0.00324113], 
                [ 0.00324113], [ 0.00338389], [ 0.00338389], [ 0.00351423], [ 0.00351423], 
                [ 0.00362336], [ 0.00362336], [ 0.00369445], [ 0.00369445], [ 0.00369498], 
                [ 0.00369498], [ 0.00356663], [ 0.00356663], [ 0.00321363], [ 0.00321363], 
                [ 0.00250733], [ 0.00250733], [ 0.00135151], [ 0.00135151], [-0.00017009]]

def test_tempo_gradient_backend_J():
    tempo_params_J =oqupy.TempoParameters(
        dt=0.05,
        tcut=None,
        epsrel=10**(-7))
    pt = oqupy.pt_tempo_compute(
        bath_J,
        start_time=0.0,
        end_time=t_end_J,
        parameters=tempo_params_J)
    
    grad_prop,dyn = oqupy.compute_gradient_and_dynamics(system=system_J,
                                                    parameters=x0,
                                                    process_tensors=[pt],
                                                    initial_state=initial_state_J,
                                                    target_derivative=target_derivative_J
                                                    )

    np.testing.assert_almost_equal(dyn.states[-1], rho_J, decimal=4)

    get_props = system_J.get_propagators(dt,x0)
    get_prop_derivatives = system_J.get_propagator_derivatives(dt,x0)

    grad_params = oqupy.gradient._chain_rule(adjoint_tensor=grad_prop,
                                            dprop_dparam=get_prop_derivatives,
                                            propagators=get_props,
                                            num_steps=num_steps,
                                            num_parameters=1)
    
    np.testing.assert_almost_equal(grad_params.real,grad_params_J,decimal=4)
