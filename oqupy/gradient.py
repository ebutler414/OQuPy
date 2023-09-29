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
from typing import Dict, Iterable, List, Optional, Tuple

from numpy import ndarray, sqrt

from oqupy.contractions import compute_gradient_and_dynamics
from oqupy.process_tensor import BaseProcessTensor
from oqupy.system import ParametrizedSystem


def fidelity_gradient(
        system: ParametrizedSystem,
        initial_state: ndarray,
        target_state: ndarray,
        process_tensor: BaseProcessTensor,
        parameters: Tuple[List],
        time_steps: Optional[Iterable] = None,
        return_fidelity: Optional[bool] = True,
        return_dynamics: Optional[bool] = False) -> Dict:
    """
    ToDo:
    the return dictionary has the fields:
      'gradient' : a tuple of list of floats
                   ['gradient'][i][n] ... is the derivative with respect to
                                          the i-th parameter at the n-th
                                          half-time step.
      'fidelity' : float (optional)
      'dynamics' : a Dynamics object (optional)

    """

    fb_prop_result = forward_backward_propagation(
        system,
        initial_state,
        target_state,
        process_tensor,
        parameters)

    if time_steps is None:
        time_steps = range(2*len(process_tensor))

    for n in time_steps:
        pass # ToDo

    return NotImplemented


def forward_backward_propagation(
        system: ParametrizedSystem,
        initial_state: ndarray,
        target_state: ndarray,
        process_tensor: BaseProcessTensor,
        parameters: Tuple[List], # unnecessary?
        return_fidelity: Optional[bool] = True,
        return_dynamics: Optional[bool] = False) -> Dict:
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
