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

    first_mpos= _get_pt_mpos(process_tensors, 0)

    new_mpo=np.squeeze(first_mpos,0) # reshape MPO to work with algorithm
    new_mpo=np.squeeze(new_mpo,0) # reshape MPO to work with algorithm

    print(new_mpo.shape)

    current_node=tn.Node(new_mpo)
    current_edges=current_node[:]
    for step in range(1,num_steps+1):

        if step == num_steps:
            break

        #forwardprop_derivs_list.append(tn.replicate_nodes([current_node])[0])

        # -- propagate one time step --
        first_half_prop, second_half_prop = propagators(step)

        pt_mpos = _get_pt_mpos(process_tensors, step)

        current_node, current_edges = _apply_system_superoperator(
            current_node, current_edges, first_half_prop)
        current_node, current_edges = _apply_pt_mpos(
            current_node, current_edges, pt_mpos)

        current_node, current_edges = _apply_system_superoperator(
            current_node, current_edges, second_half_prop)

    # -- extract last state --
    caps = _get_caps(process_tensors, num_steps)
    dynamical_map = _apply_caps(current_node, current_edges, caps)

    return dynamical_map


'''
    # target_ndarray.shape = tuple([1]*num_envs+[hs_dim**2])
    # target_ndarray = np.outer(caps,target_ndarray)

    if len(process_tensors)>1: #allows for multiple environments
        reshaped = []
        for i, v in enumerate(caps):
            shape = [1] * len(process_tensors)     # all ones
            shape[i] = -1                  # dimension to fill
            reshaped.append(v.reshape(shape))

        # outer product over all N vectors : (x1, x2, ..., xN)
        outer = reshaped[0]
        for v in reshaped[1:]:
            outer = outer * v  # 
        target_ndarray = outer[..., None] * target_ndarray

        # multiply with the target : (x1, ..., xN, d)
        current_node = tn.Node(target_ndarray)
        current_edges = current_node[:]
    else:
        target_ndarray = np.outer(caps,target_ndarray)
    combined_deriv_list = []
'''

def compute_gradient_and_map(
        system: ParameterizedSystem,
        process_tensors: List[BaseProcessTensor],
        parameters: ndarray,
        start_time: Optional[float] = 0.0,
        dt: Optional[float] = None,
        num_steps: Optional[int] = None,
        control: Optional[Control] = None,
        record_all: Optional[bool] = True,
        progress_type: Optional[Text] = None) -> Tuple[List, Dynamics]:
    """
    Compute some objective function and calculate its gradient w.r.t.
    some control parameters, accounting for interaction with an environment
    using one or more process tensors.

    Parameters:
    -----------
    system: ParametrizedSystem
        Parameterized system taking M parameters.
    initial_state: ndarray
        The initial density matrix to propagate forwards
    target_derivative: Union[Callable, ndarray]
        A pure target state transposed or derivative w.r.t. an objective
        function.
    process_tensors: List[BaseProcessTensor]
        A list of process tensors (each with N time steps) representing the
        environment.
    parameters: List[Tuple]
        A list of M-tuples with length 2N. Each tuple corresponds to the values
        of the parameters at a given half time step.
    start_time: float
        Optional start time offset.
    dt: float
        Length of a single time step.
    num_steps: int
        Optional number of time steps to be computed.
    control: Control
        Optional control operations.
    record_all: bool
        If `false` function only computes the final state.
    progress_type: str (default = None)
        The progress report type during the computation. Types are:
        {``silent``, ``simple``, ``bar``}. If `None` then
        the default progress type is used.

    Returns:
    --------
    propagator_derivatives: List[ndarray]
        List of 4-rank tensors. The nth entry corresponds to the derivative of
        the objective function with respect to a propagator at the nth
        time step. The axis are ordered as follows:
            * [0] : output leg of 2nd half-propagator from step (n-1)
            * [1] : input system leg of MPO from step n
            * [2] : output system leg of MPO from step n
            * [3] : input lef of 1st half-propagator from step (n+1)
    dynamics: Dynamics
        The system dynamics for the given system Hamiltonian
        (accounting for the interaction with the environment).
    """

    num_envs = len(process_tensors)

    if num_steps is None:
        num_steps=len(process_tensors[0])

    # -- prepare propagators --
    propagators = system.get_propagators(dt, parameters)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~ Forwardpropagation ~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # -- initialize computation --
    #
    #  Initial state including the bond legs to the environments with:
    #    edges 0, 1, .., num_envs-1    are the bond legs of the environments
    #    edge  -1                      is the state leg

    first_mpos= _get_pt_mpos(process_tensors, 0)

    new_mpo=np.squeeze(first_mpos,0) # reshape MPO to work with algorithm
    new_mpo=np.squeeze(new_mpo,0) # reshape MPO to work with algorithm

    print(new_mpo.shape)

    current_node=tn.Node(new_mpo)
    current_edges=current_node[:]

    

    forwardprop_derivs_list = []
    mpo_list=[]

    for step in range(1,num_steps+1):

        if step == num_steps:
            break

        forwardprop_derivs_list.append(tn.replicate_nodes([current_node])[0])

        # -- propagate one time step --
        first_half_prop, second_half_prop = propagators(step)

        pt_mpos = _get_pt_mpos(process_tensors, step)
        mpo_list.append(pt_mpos)

        current_node, current_edges = _apply_system_superoperator(
            current_node, current_edges, first_half_prop)
        current_node, current_edges = _apply_pt_mpos(
            current_node, current_edges, pt_mpos)

        current_node, current_edges = _apply_system_superoperator(
            current_node, current_edges, second_half_prop)

    # -- extract last state --
    caps = _get_caps(process_tensors, num_steps)
    map = _apply_caps(current_node, current_edges, caps)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~ Backpropagation ~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # -- initialize computation (except backwards) --
    #
    #  Initial state including the bond legs to the environments with:
    #    edges 0, 1, .., num_envs-1    are the bond legs of the environments
    #    edge  -1                      is the state leg

    title = "--> Compute backward propagation:"
    prog_bar = get_progress(progress_type)(num_steps, title)
    prog_bar.enter()

    forwardprop_tensor = forwardprop_derivs_list[num_steps-2]
    combined_deriv_list=[]

    pt_mpos = mpo_list[num_steps-2]

    #new_mpo=np.squeeze(pt_mpos,0) # reshape MPO to work with algorithm

    current_node=tn.Node(new_mpo)
    current_edges=current_node[:]
    backprop_tensor = tn.replicate_nodes([current_node])[0]

    fwd_edges = forwardprop_tensor[:]
    deriv_forwardprop_tensor, fwd_edges = _apply_derivative_pt_mpos(
        forwardprop_tensor,fwd_edges,pt_mpos)

    for i, _ in enumerate(pt_mpos):
        fwd_edges[i] ^ backprop_tensor[i]

    deriv = deriv_forwardprop_tensor @ backprop_tensor

    combined_deriv_list.append(tn.replicate_nodes([deriv])[0])

    for loop, step in enumerate(reversed(range(1,num_steps))):

        prog_bar.update(loop)

        # -- now the backpropagation part --
        first_half_prop, second_half_prop = propagators(step)
        pt_mpos = _get_pt_mpos_backprop(mpo_list, step)

        current_node, current_edges = _apply_system_superoperator(
            current_node, current_edges, second_half_prop.T)

        current_node, current_edges = _apply_pt_mpos(
            current_node, current_edges, pt_mpos)

        current_node, current_edges = _apply_system_superoperator(
            current_node, current_edges, first_half_prop.T)

        forwardprop_tensor = forwardprop_derivs_list[step-1]

        backprop_tensor =  tn.replicate_nodes([current_node])[0]

        pt_mpos = mpo_list[step-1]

        fwd_edges = forwardprop_tensor[:]
        deriv_forwardprop_tensor,fwd_edges = _apply_derivative_pt_mpos(
            forwardprop_tensor,fwd_edges,pt_mpos)

        for i, _ in enumerate(pt_mpos):
            fwd_edges[i] ^ backprop_tensor[i]

        deriv = deriv_forwardprop_tensor @ backprop_tensor

        combined_deriv_list.append(deriv.get_tensor())
        # ordering of axis:
        # deriv[0] : output leg of 2nd half-propagator from step (n-1)
        # deriv[1] : input system leg of MPO from step n
        # deriv[2] : output system leg of MPO from step n
        # deriv[3] : input lef of 1st half-propagator from step (n+1)

    propagator_derivatives = list(reversed(combined_deriv_list))

    return propagator_derivatives, map

