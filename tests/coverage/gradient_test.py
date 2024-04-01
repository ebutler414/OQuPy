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
Tests for the oqupy.gradient module.
"""

import pytest
import numpy as np
from numpy import ndarray

from typing import Dict

import oqupy
import oqupy.operators as op

from oqupy.gradient import state_gradient

def test_state_gradient():
    start_time=0
    num_steps=3
    dt=0.2
    end_time=start_time+num_steps*dt
    initial_state = oqupy.operators.spin_dm('x-')
    target_derivative = oqupy.operators.spin_dm('x+')

    x0 = np.ones(2*num_steps)
    x0=list(zip(x0))

    def discrete_h_sys(hx):
        return 0.5*hx * oqupy.operators.sigma('x')

    system= oqupy.ParameterizedSystem(hamiltonian=discrete_h_sys)

    correlations = oqupy.PowerLawSD(alpha=3,
                                zeta=1,
                                cutoff=1.0,
                                cutoff_type='gaussian',
                                temperature=0.0)
    bath = oqupy.Bath(0.5 * oqupy.operators.sigma("x"), correlations)

    tempo_params =oqupy.TempoParameters(dt=0.2,tcut=None,epsrel=10**(-7))
    pt = oqupy.pt_tempo_compute(
        bath,
        start_time=start_time,
        end_time=end_time,
        parameters=tempo_params)
    
    grad_dict = state_gradient(system=system,
                               initial_state=initial_state,
                               target_derivative=target_derivative.T,
                               process_tensors=[pt],
                               parameters=x0)
    
    # Check return value is correct type
    assert isinstance(grad_dict,Dict)
    # Check shape of gradient list is that of the input parameters
    assert np.shape(grad_dict['gradient']) == np.shape(x0)
    # Check there are N gradient tensors
    assert len(grad_dict['gradprop']) == num_steps 
    # Check there are N+1 states
    assert len(grad_dict['dynamics']) == num_steps+1
    # Check the last element of the dynamics object is equal to the final state
    assert np.allclose(grad_dict['dynamics'].states[-1],grad_dict['final state'])


def test_chain_rule():

    num_steps=3
    dt=0.2

    x0 = np.ones(2*num_steps)
    x0=list(zip(x0))
    num_params=1

    def discrete_h_sys(hx):
        return 0.5*hx * oqupy.operators.sigma('x')
    system= oqupy.ParameterizedSystem(hamiltonian=discrete_h_sys)

    propagators = system.get_propagators(dt,x0)
    propagator_derivatives = system.get_propagator_derivatives(dt,x0)

    # rank-4 tensor
    dummy_tensor = np.ones((4,4,4,4,4))
    
    tot_derivs = oqupy.gradient._chain_rule(adjoint_tensor=dummy_tensor,
                                            dprop_dparam=propagator_derivatives,
                                            propagators=propagators,
                                            num_steps=num_steps,
                                            num_parameters=num_params)
    
    # Check the shapes of the derivatives and parameters are correct
    assert np.shape(tot_derivs) == (2*num_steps,num_params)
    assert np.shape(x0) == (2*num_steps,num_params)
    # Check the propagators and derivatives are callables
    assert callable(propagators)
    assert callable(propagator_derivatives)
    # Check the tot_derivs is a numpy array
    assert isinstance(tot_derivs,ndarray)