from typing import Callable, Dict, List, Optional, Text, Tuple, Union

import numpy as np
from numpy import ndarray
import tensornetwork as tn

from oqupy.system_dynamics import _compute_dynamics_input_parse, \
    _apply_system_superoperator, _apply_derivative_pt_mpos, _get_pt_mpos, \
    _get_pt_mpos_backprop, _get_caps, _apply_caps, _apply_pt_mpos
from oqupy.control import Control
from oqupy.dynamics import Dynamics
from oqupy.process_tensor import BaseProcessTensor
from oqupy.system import ParameterizedSystem
from oqupy.util import get_progress, check_isinstance

def compute_dynamical_map(system: ParameterizedSystem,
        process_tensors: List[BaseProcessTensor],
        parameters: ndarray,
        start_time: Optional[float] = 0.0,
        num_steps: Optional[int]=None,
        progress_type: Optional[Text] = None)-> List:
    
    propagators=system.get_propagators()
    mpo_list=[]

    for step in range(num_steps):
        mpos = _get_pt_mpos(process_tensors, step)
        mpo_list.append(mpos)

        # get propagators for the current step
        first_half_prop, second_half_prop = propagators(step)

        #apply propagators to the mpo
        prop_node,prop_edges=_apply_propagators(mpo_list,first_half_prop,second_half_prop)
    	
        # save influence modified propagator
        modified_propagators.append(tn.replicate_nodes([current_node])[0])

        if step==0:
            continue

        current_node,current_edges=


    return dynamical_map