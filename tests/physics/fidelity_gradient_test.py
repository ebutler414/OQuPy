# Copyright 2022 The TEMPO Collaboration
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
Test the physical sanity of oqupy.fidelity_gradient().
"""

import pytest
import numpy as np

import oqupy
import oqupy.operators as op

from oqupy import state_gradient

def test_fidelity_gradient():

    rho_final = op.spin_dm('x+')

    dt = 0.2
    num_steps = 50

    # -- bath --
    alpha = 0 # No bath interaction
    omega_cutoff = 4.0
    temperature = 1.6
    pt_dkmax = 40
    pt_epsrel = 1.0e-5

    # -- initial and target state --
    initial_state = op.spin_dm('x-')
    target_state = op.spin_dm('x+')

    # -- initial parameter guess --
    x0 = np.zeros(2*num_steps)
    y0 = np.zeros(2*num_steps)
    z0 = np.ones(2*num_steps) * (np.pi) / (2*dt*num_steps)
    
    correlations = oqupy.PowerLawSD(
    alpha=alpha,
    zeta=1,
    cutoff=omega_cutoff,
    cutoff_type='exponential',
    temperature=temperature)

    bath = oqupy.Bath(0.0 * op.sigma('y'), correlations)
    pt_tempo_parameters = oqupy.TempoParameters(
        dt=dt,
        epsrel=pt_epsrel,
        dkmax=pt_dkmax)
    
    process_tensor = oqupy.pt_tempo_compute(
        bath=bath,
        start_time=0.0,
        end_time=num_steps * dt,
        parameters=pt_tempo_parameters,
        progress_type='bar')
    
    def hamiltonian(x, y, z):
        h = np.zeros((2,2),dtype='complex128')
        for var, var_name in zip([x,y,z], ['x', 'y', 'z']):
            h += var * op.sigma(var_name)
        return h
    
    parameterized_system = oqupy.ParameterizedSystem(hamiltonian)

    gradient_dict = state_gradient(
            system=parameterized_system,
            initial_state=initial_state,
            target_state=target_state.T,
            process_tensor=[process_tensor],
            parameters=list(zip(x0,y0,z0)),
            return_fidelity=True,
            return_dynamics=True)
    
    dynamics = gradient_dict['dynamics']
    derivatives = gradient_dict['gradprop']
    
    np.testing.assert_almost_equal(rho_final,dynamics.states[-1]) # checking unitary dynamics

    assert True
