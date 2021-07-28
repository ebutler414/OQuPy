'''
Module for computing the gradient of a problem with respect to the control parameters

'''

from typing import Dict, Optional, Text, List


import os
import tempfile
from typing import Callable, Dict, List, Optional, Text, Tuple

import numpy as np
from numpy import ndarray
from scipy.linalg import expm
import tensornetwork as tn
import h5py
import time_evolving_mpo

from time_evolving_mpo.process_tensor import SimpleProcessTensor, BaseProcessTensor, compute_dynamics_from_system, _compute_dynamics
from time_evolving_mpo.base_api import BaseAPIClass
from time_evolving_mpo.config import NpDtype
from time_evolving_mpo.dynamics import Dynamics
from time_evolving_mpo.system import BaseSystem
from time_evolving_mpo import util

import time_evolving_mpo as tempo


class SimpleAdjointTensor(SimpleProcessTensor):
    pass # not implemented yet..........




def _compute_dynamics_forwards(
        process_tensor: BaseProcessTensor,
        controls: Callable[[int], Tuple[ndarray, ndarray]],
        initial_state: Optional[ndarray] = None,
        num_steps: Optional[int] = None,
        record_all: Optional[bool] = True) -> List[ndarray]:
    """See BaseProcessTensorBackend.compute_dynamics() for docstring. """
    hs_dim = process_tensor.hilbert_space_dimension

    initial_tensor = process_tensor.get_initial_tensor()
    assert (initial_state is None) ^ (initial_tensor is None), \
        "Initial state must be either (exclusively) encoded in the " \
        + "process tensor or given as an argument."
    if initial_tensor is None:
        initial_tensor = util.add_singleton(
            initial_state.reshape(hs_dim**2), 0)

    current = tn.Node(initial_tensor)
    current_bond_leg = current[0]
    current_state_leg = current[1]
    states = []

    if num_steps is None:
        __num_steps = len(process_tensor)
    else:
        __num_steps = num_steps

    for step in range(__num_steps):
        if record_all:
            # -- extract current state --
            try:
                cap = process_tensor.get_cap_tensor(step)
            except Exception as e:
                raise ValueError("There are either no cap tensors in the "\
                        +"process tensor or the process tensor is not "\
                        +"long enough") from e
            if cap is None:
                raise ValueError("Process tensor has no cap tensor "\
                    +f"for step {step}.")
            cap_node = tn.Node(cap)
            node_dict, edge_dict = tn.copy([current])
            edge_dict[current_bond_leg] ^ cap_node[0]
            state_node = node_dict[current] @ cap_node
            state = state_node.get_tensor().reshape(hs_dim, hs_dim)
            states.append(state)

        # -- propagate one time step --details_pt_tempo
        try:
            mpo = process_tensor.get_mpo_tensor(step)
        except Exception as e:
            raise ValueError("The process tensor is not long enough") from e
        if mpo is None:
            raise ValueError("Process tensor has no mpo tensor "\
                +f"for step {step}.")
        mpo_node = tn.Node(mpo)
        pre, post = controls(step)
        pre_node = tn.Node(pre)
        post_node = tn.Node(post)

        lam = process_tensor.get_lam_tensor(step)
        if lam is None:
            current_bond_leg ^ mpo_node[0]
            current_state_leg ^ pre_node[0]
            pre_node[1] ^ mpo_node[2]
            mpo_node[3] ^ post_node[0]
            current_bond_leg = mpo_node[1]
            current_state_leg = post_node[1]
            current = current @ pre_node @ mpo_node @ post_node
        else:
            lam_node = tn.Node(lam)
            current_bond_leg ^ mpo_node[0]
            current_state_leg ^ pre_node[0]
            pre_node[1] ^ mpo_node[2]
            mpo_node[1] ^ lam_node[0]
            mpo_node[3] ^ post_node[0]
            current_bond_leg = lam_node[1]
            current_state_leg = post_node[1]
            current = current @ pre_node @ mpo_node @ lam_node @ post_node

    # -- extract last state --
    cap = process_tensor.get_cap_tensor(__num_steps)
    if cap is None:
        raise ValueError("Process tensor has no cap tensor "\
            +f"for step {step}.")
    cap_node = tn.Node(cap)
    current_bond_leg ^ cap_node[0]
    final_state_node = current @ cap_node
    final_state = final_state_node.get_tensor().reshape(hs_dim, hs_dim)
    states.append(final_state)

    return states



