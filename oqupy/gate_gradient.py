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
    map_list=[]
    hs_dim=2

    # (TODO: multiple process tensor compatibility)
    for step in range(0,num_steps):

        # -- creates short time propagator for each step --
        pt_mpos = _get_pt_mpos(process_tensors, step)

        current_node = tn.Node(pt_mpos[0])

        current_edges=current_node[:]

        first_propagator,second_propagator=propagators(step)

        current_node,current_edges=apply_mpo_propagators(current_node,current_edges,first_propagator,second_propagator)

        if step==0:
            prev_node,prev_edges=current_node,current_edges
            caps = _get_caps(process_tensors, 1)
            map_tensor = _apply_caps(current_node, current_edges[1:], caps)
            map_list.append(map_tensor[0])
            continue
        
        new_node,new_edges=stitch_mpos(prev_node,prev_edges,current_node,current_edges)
        new_edges=new_node[:]

        caps = _get_caps(process_tensors, step)
        map_tensor = _apply_caps(new_node, new_edges[1:], caps)
        map_list.append(map_tensor)

        prev_node,prev_edges=new_node,new_edges

    return map_list

def compute_dynamical_map_and_grad(system: ParameterizedSystem,
        process_tensors: List[BaseProcessTensor],
        parameters: ndarray,
        dt:float,
        start_time: Optional[float] = 0.0,
        num_steps: Optional[int]=None,
        progress_type: Optional[Text] = None)-> List:
    
    propagators=system.get_propagators(dt,parameters)
    map_list=[]
    hs_dim=2

    #start_cap=tn.Node(np.array([1.]))
    d = 1  # dimension of each leg
    start_cap = np.zeros((d, d, d))
    for i in range(d):
        start_cap[i, i, i] = 1.0
    
    forward_nodes=[tn.Node(start_cap)]
    short_time_props=[]

    # (TODO: multiple process tensor compatibility)
    # forward propagation
    for step in range(0,num_steps):

        # -- creates short time propagator for each step --
        pt_mpos = _get_pt_mpos(process_tensors, step)

        current_node = tn.Node(pt_mpos[0])

        current_edges=current_node[:]

        first_propagator,second_propagator=propagators(step)

        current_node,current_edges=apply_mpo_propagators(current_node,current_edges,first_propagator,second_propagator)

        short_time_props.append(tn.replicate_nodes([current_node])[0])

        if step==0:
            prev_node,prev_edges=current_node,current_edges
            caps = _get_caps(process_tensors, 1)
            map_tensor = _apply_caps(current_node, current_edges[1:], caps)
            map_list.append(map_tensor[0])
            continue
        
        fwd_node=tn.Node(prev_node.tensor[0])
        forward_nodes.append(tn.replicate_nodes([fwd_node])[0])

        
        new_node,new_edges=stitch_mpos(prev_node,prev_edges,current_node,current_edges)
        new_edges=new_node[:]

        caps = _get_caps(process_tensors, step)
        map_tensor = _apply_caps(new_node, new_edges[1:], caps)
        map_list.append(map_tensor)

        prev_node,prev_edges=new_node,new_edges

    grad_list=[]
    # back propagation
    for step in reversed(range(0,num_steps)):
        
        pt_mpos = _get_pt_mpos(process_tensors, step)
 
        forwardprop_node=forward_nodes[step]
        fwd_edges=forwardprop_node[:]
        
        # applying mpo without propagators to fwd prop tensor
 
        current_node,current_edges = apply_derivative_gate(
        forwardprop_node,fwd_edges,pt_mpos[0])

        if step==num_steps-1: # first step of backprop

            caps = _get_caps(process_tensors, 1)
            grad_tensor = _apply_caps(current_node, current_edges, caps)
            grad_list.append(grad_tensor)

            prev_node,prev_edges=short_time_props[step],short_time_props[step][:]
            continue
        
        # construct derivative
        back_prop_node=tn.replicate_nodes([prev_node])[0]

        back_prop_edges=back_prop_node[:]
        current_edges[0] ^ back_prop_edges[0]

        deriv_node = tn.contract_between(current_node, back_prop_node)

        deriv_edges=[back_prop_edges[1],current_edges[1],current_edges[2],current_edges[3],current_edges[4],back_prop_edges[2],back_prop_edges[3]]
        deriv_node.reorder_edges(deriv_edges)
        
        caps = _get_caps(process_tensors, 1)
        grad_tensor = _apply_caps(deriv_node,deriv_edges, caps)

        grad_list.append(grad_tensor)

        if step==0: # last step (only need N-1 back prop tensors)
            break

        # do back propagation
        short_time_node,short_time_edge=short_time_props[step],short_time_props[step][:]

        prev_edges[0]^short_time_edge[1] # bond edges
        prev_edges[2]^short_time_edge[3] # system edges
        new_node = tn.contract_between(prev_node, short_time_node)
        new_edges = [short_time_edge[0], prev_edges[1], short_time_edge[2], prev_edges[3]]
        new_node.reorder_edges(new_edges)

        prev_node,prev_edges=new_node,new_edges
    
    grad_list=list(reversed(grad_list))

    grad_list[0]=np.squeeze(grad_list[0]) # remove dummy legs that come from start_cap
    
    return map_list,grad_list

def gate_chain_rule(
        adjoint_tensor:ndarray,
        dprop_dparam:Callable[[int], Tuple[ndarray,ndarray]],
        propagators:Callable[[int], Tuple[ndarray,ndarray]],
        num_steps:int,
        num_parameters:int,
        progress_type: Optional[Text] = None):

    def combine_derivs(
            target_deriv,
            pre_prop,
            post_prop):
        target_node = tn.Node(target_deriv)
        pre_node=tn.Node(pre_prop)
        post_node=tn.Node(post_prop)

        target_node[1]^pre_node[0]
        target_node[2]^pre_node[1]
        target_node[3]^post_node[0]
        target_node[4]^post_node[1]

        final_node = target_node @ pre_node \
                        @ post_node
        tensor = final_node.tensor

        return tensor
    
    hs_dim=2
    d = hs_dim**2
    total_derivs = np.zeros((2*num_steps, num_parameters, d, d), dtype='complex128')


    title = "--> Apply chain rule:"
    prog_bar = get_progress(progress_type)(num_steps, title)
    prog_bar.enter()

    for i in range(0,num_steps): # populating two elements each step

        first_half_prop, second_half_prop = propagators(i)
        first_half_prop_derivs,second_half_prop_derivs = dprop_dparam(i)

        prog_bar.update(i)

        for j in range(0,num_parameters):
            total_derivs[2*i][j] = combine_derivs(
                            adjoint_tensor[i],
                            first_half_prop_derivs[j].T,
                            second_half_prop.T)
            total_derivs[2*i+1][j] = combine_derivs(
                adjoint_tensor[i],
                first_half_prop.T,
                second_half_prop_derivs[j].T)

    prog_bar.update(num_steps)
    prog_bar.exit()

    return total_derivs


def apply_mpo_propagators(current_node,current_edges,first_propagator,second_propagator):
    
    current_edges= current_node[:]

    first_node = tn.Node(first_propagator.T)
    second_node= tn.Node(second_propagator.T)

    new_input=first_node[0]
    new_output=second_node[1]
    new_bond_in=current_node[0]
    new_bond_out=current_node[1]

    current_edges[2] ^ first_node[1]
    current_edges[3] ^ second_node[0]

    new_node = current_node @ first_node @ second_node
    new_edges=new_node[:]
    new_edges[3] = new_output
    new_edges[2] = new_input
    new_edges[1]=new_bond_out
    new_edges[0]=new_bond_in

    new_node.reorder_edges(new_edges) 
 
    return new_node,new_edges

def stitch_mpos(prev_node,prev_edges,curr_node,curr_edges):

    prev_edges=prev_node[:]
    curr_edges=curr_node[:]

    prev_edges[3] ^ curr_edges[2]   
    prev_edges[1] ^ curr_edges[0]  
    new_node = tn.contract_between(prev_node, curr_node)
    new_edges = [prev_edges[0], curr_edges[1], prev_edges[2], curr_edges[3]]
    new_node.reorder_edges(new_edges)

    return new_node,new_edges

def apply_derivative_gate(prev_node,prev_edges,pt_mpo):

    mpo_node=tn.Node(pt_mpo)

    mpo_edges=mpo_node[:]
    prev_edges=prev_node[:]

    prev_edges[0]^mpo_edges[0]
    new_node=tn.contract_between(prev_node,mpo_node)
    new_edges=new_node[:]

    new_edges=[mpo_edges[1],prev_edges[1],prev_edges[2],mpo_edges[2],mpo_edges[3]]

    new_node.reorder_edges(new_edges)

    return new_node,new_edges





