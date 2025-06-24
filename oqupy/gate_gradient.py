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
        dt:float,
        start_time: Optional[float] = 0.0,
        num_steps: Optional[int]=None,
        progress_type: Optional[Text] = None)-> List:
    
    propagators=system.get_propagators(dt,parameters)
 
    x=0
    for step in range(num_steps):
        mpos = _get_pt_mpos(process_tensors, step)
  

        mpo_node=tn.Node(mpos[0])
        mpo_edges=mpo_node[:]

        first_half_prop, second_half_prop = propagators(step)

        mpo_node,mpo_edges=_apply_propagators(mpo_node,mpo_edges,first_half_prop,second_half_prop)
        print(step)
        if step==0:
            current_node,current_edges=mpo_node,mpo_edges
            x=2
            continue
        print(current_node.shape)
        current_node,current_edges=_apply_pt_mpos_gate(current_node,current_edges,mpo_node,mpo_edges,x)
    
    final_node=tn.Node(np.squeeze(current_node.tensor))
    current_edges=final_node[:]
    
    caps = _get_caps(process_tensors, num_steps)
    dynamical_map = _apply_caps(final_node, current_edges, caps)

    return dynamical_map

def _apply_propagators(mpo_node,mpo_edges,first_half_prop,second_half_prop):
    first_prop=tn.Node(first_half_prop.T)
    second_prop=tn.Node(second_half_prop.T)

    first_edges,second_edges=first_prop[:],second_prop[:]
    first_edges[1] ^ mpo_node[-2]
    second_edges[0]^ mpo_node[-1]

    new_mpo= first_prop@mpo_node@second_prop

    new_mpo.reorder_edges([new_mpo[1],new_mpo[2],new_mpo[3],new_mpo[0]])

    mpo_edges=new_mpo[:]
    return new_mpo, mpo_edges

def _apply_pt_mpos_gate(current_node,current_edges,mpo_node,mpo_edges,x):
    current_edges[1]^mpo_edges[0]
    current_edges[3]^mpo_edges[2]

    new_node=current_node@mpo_node

    new_node.reorder_edges([new_node[0+x],new_node[2-x],new_node[1],new_node[3]])
    new_edges=new_node[:]

    return new_node, new_edges
