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
Frontend for computing the gradient of some objective function w.r.t. some
control parameters.
"""
from typing import Dict, Iterable, List, Optional, Tuple, Callable

import numpy as np
import tensornetwork as tn

from numpy import ndarray, sqrt, zeros

from oqupy.contractions import compute_gradient_and_dynamics
from oqupy.process_tensor import BaseProcessTensor
from oqupy.system import ParameterizedSystem
from oqupy.config import NpDtype, INTEGRATE_EPSREL, SUBDIV_LIMIT


def state_gradient(
        system: ParameterizedSystem,
        initial_state: ndarray,
        target_state: ndarray,
        process_tensor: BaseProcessTensor,
        parameters: List[Tuple],
        time_steps: Optional[ndarray] = None,
        return_dynamics: Optional[bool] = False,
        return_gradprop: Optional[bool] = False,
        return_gradparam: Optional[bool] = False
        ) -> Dict:
    """
    Compute system dynamics and gradient of an objective function with respect to a parameterized Hamiltonian,
    for a given set of control parameters, accounting for the interaction with an environment described by a 
    process tensor. The objective function Z is the (specify) product of the final state and target_state.
    Inputs:
        system : ParameterizedSystem object to compute the dynamics
        initial_state : the initial density matrix to propagate forwards
        target_state : either the state to propagate backwards, or 
                        a function, which will be called with the final state and should return the 
                        state to be back-propagated.

    The return dictionary has the fields:
      'final state' : the final state after evolving the initial state
      'gradprop' : derivatives of Z with respect to half-step propagators  
      'gradient' : derivatives of Z with respect to parameters
                   a tuple list of floats
                   ['gradient'][i][n] ... is the derivative with respect to
                                          the i-th parameter at the n-th
                                          half-time step.
      'dynamics' : a Dynamics object (optional) 

    """

    # compute propagator list and pass this to forward_backward_propagation, 

    fb_prop_result = forward_backward_propagation(
        system,
        initial_state,
        target_state,
        process_tensor,
        parameters)
    
    
    grad_prop = fb_prop_result['derivatives']
    dynamics = fb_prop_result['dynamics']

    if time_steps is None:
        time_steps = range(2*len(process_tensor))

    num_parameters = len(parameters[0])
    dt = process_tensor.dt

    get_prop_derivatives = system.get_propagator_derivatives(dt=dt,parameters=parameters)
    get_half_props= system.get_propagators(process_tensor.dt,parameters)

    final_derivs = _chain_rule(
        adjoint_tensor=grad_prop,
        dprop_dparam=get_prop_derivatives,
        propagators=get_half_props,
       num_half_steps=2*len(process_tensor),
       num_parameters=num_parameters)
    
    return_dict = {
        'final state':dynamics.states[-1],
        'gradprop':grad_prop,
        'gradient':final_derivs,
        'dynamics':dynamics
    }
    
    return return_dict

def forward_backward_propagation(
        system: ParameterizedSystem,
        initial_state: ndarray,
        target_state: ndarray,
        process_tensor: BaseProcessTensor,
        parameters: List[Tuple], # unnecessary?
        return_fidelity: Optional[bool] = True,
        return_dynamics: Optional[bool] = True) -> Dict:
    """
    ToDo:
    the return dictionary has the fields:
      'derivative' : List[ndarrays]
      'pre_propagators' : List[ndarrays]
      'post_propagators' : List[ndarrays]
      'pre_control' : List[Union[ndarrays,NoneType]]
      'post_control' : List[Union[ndarrays,NoneType]]
      'fidelity' : Optional[float]
      'dynamics' : Optional[Dynamics]
    """
    adjoint_tensors,dynamics = compute_gradient_and_dynamics(
        system=system,
        initial_state=initial_state,
        target_state=target_state,
        process_tensor=process_tensor,
        parameters=parameters,
        record_all=return_dynamics)

    fidelity = None
    if return_fidelity:
        sqrt_final_state = sqrt(dynamics.states[-1])
        intermediate_1 = sqrt_final_state @ target_state
        inside_of_sqrt = intermediate_1 @ sqrt_final_state
        fidelity = (sqrt(inside_of_sqrt).trace())**2

    return_dict = {
        'derivatives':adjoint_tensors,
        'fidelity':fidelity,
        'dynamics':dynamics
    }

    return return_dict

def _chain_rule(
        adjoint_tensor:ndarray,
        dprop_dparam:Callable[[int], Tuple[ndarray,ndarray]],
        propagators:Callable[[int], Tuple[ndarray,ndarray]],
        num_half_steps:int,
        num_parameters:int):

    def combine_derivs(
            target_deriv,
            pre_prop,
            post_prop):

            target_deriv_node = tn.Node(target_deriv)
            pre_node=tn.Node(pre_prop)
            post_node=tn.Node(post_prop)

            target_deriv_node[0] ^ pre_node[0]
            target_deriv_node[1] ^ pre_node[1]
            target_deriv_node[2] ^ post_node[0] 
            target_deriv_node[3] ^ post_node[1] 

            final_node = target_deriv_node @ pre_node \
                            @ post_node

            tensor = final_node.tensor

            return tensor
    
    total_derivs = np.zeros((num_half_steps,num_parameters),dtype='complex128')

    for i in range(0,num_half_steps-1,2): # populating two elements each step
            
        first_half_prop,second_half_prop=propagators(i//2)
        first_half_prop_derivs,second_half_prop_derivs = dprop_dparam(i//2) # returns two lists containing the derivatives w.r.t. each parameter

        for j in range(0,num_parameters):
            
            total_derivs[i][j] = combine_derivs(
                            adjoint_tensor[i//2],
                            first_half_prop,
                            second_half_prop_derivs[j])

            total_derivs[i+1][j] = combine_derivs(
                            adjoint_tensor[i//2],
                            first_half_prop_derivs[j],
                            second_half_prop)

    return total_derivs