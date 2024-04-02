#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0,'.')
from IPython import embed

import numpy as np
import matplotlib.pyplot as plt

import oqupy
import oqupy.operators as op

from scipy.optimize import minimize, Bounds
from typing import List,Tuple

# --- Parameters --------------------------------------------------------------

# parameters used in https://arxiv.org/abs/2303.16002

# -- time steps --
dt = 0.05 # 0.2

# -- bath --
alpha =  0.126 #0.08
omega_cutoff = 3.04 #4
temperature = 5 * 0.1309 #1.6
pt_dkmax =60 # 40   60
pt_epsrel = 10**(-7) #1.0e-5

# -- initial and target state --
initial_state = op.spin_dm('x-')
target_derivative = op.spin_dm('x+')

num_params=2

def parameters_var(num_steps):
    x0 = np.zeros(num_steps)
    z0 = np.ones(num_steps) * (np.pi) / (dt*num_steps)
    return x0,z0

# --- Compute process tensors -------------------------------------------------

correlations = oqupy.PowerLawSD(
    alpha=alpha,
    zeta=3,
    cutoff=omega_cutoff,
    cutoff_type='gaussian',
    temperature=temperature)
bath = oqupy.Bath(0.5* op.sigma('z'), correlations)
pt_tempo_parameters = oqupy.TempoParameters(
    dt=dt,
    epsrel=pt_epsrel,
    dkmax=pt_dkmax)

process_tensor_list = []
parameters_list = []
bond_dims = []

steps_list = [25,50,100]
for num_steps in steps_list:
    process_tensor=oqupy.pt_tempo_compute(
        bath=bath,
        start_time=0.0,
        end_time=num_steps * dt,
        parameters=pt_tempo_parameters,
        progress_type='bar')

    parameters_list.append([item for pair in parameters_var(num_steps) for item in pair])

    bond_dims.append(max(process_tensor.get_bond_dimensions()))
    process_tensor_list.append(process_tensor)

# --- Define parametrized system ----------------------------------------------

hs_dim = 2

def hamiltonian(x,z):
    h = 0.5 * x*oqupy.operators.sigma('x') + 0.5*z*oqupy.operators.sigma('z')
    return h

parametrized_system = oqupy.ParameterizedSystem(hamiltonian)

# ----------- Optimisation of control parameters w.r.t. infidelity ---------------

def infid(paras,process_tensor):
    """""
    Take a numpy array [hx0, hz0, hx1, hz1, ...] over full timesteps and
    return the fidelity only
    """
    reshapedparas=[i for i in (paras.reshape((-1,num_params))).tolist() for j in range(2)]

    gradient_dict=oqupy.state_gradient(system=parametrized_system,
        initial_state=initial_state,
        target_derivative=target_derivative.T,
        process_tensors=[process_tensor],
        dynamics_only=True,
        parameters=reshapedparas)
    
    fs=gradient_dict['final state']
    fidelity=np.sum(fs*target_derivative.T)

    return 1-fidelity

def infidandgrad(paras,process_tensor):
    """""
    Take a numpy array [hx0, hz0, hx1, hz1, ...] over full timesteps and
    return the infidelity and gradient of the infidelity to the global target_derivative
    """

    # Reshape flat parameter list to form accepted by state_gradient: [[hx0,hz0],[hx1,hz1,]...]
    reshapedparas=[i for i in (paras.reshape((-1,num_params))).tolist() for j in range(2)]

    gradient_dict=oqupy.state_gradient(system=parametrized_system,
        initial_state=initial_state,
        target_derivative=target_derivative.T,
        process_tensors=[process_tensor],
        parameters=reshapedparas)
    
    fs=gradient_dict['final state']
    gps=gradient_dict['gradient']
    fidelity=np.sum(fs*target_derivative.T)

    # Adding adjacent elements
    for i in range(0,gps.shape[0],2): 
        gps[i,:]=gps[i,:]+gps[i+1,:]
        
    gps=gps[0::2]

    # Return the minus the gradient as infidelity is being minimized 
    return 1-fidelity.real,(-1.0*gps.reshape((-1)).real).tolist()


def optimize(params,process_tensor,num_steps):

    # Set upper and lower bounds on control parameters
    x_bound = [-5*np.pi,5*np.pi]
    z_bound = [-np.pi,np.pi]

    bounds = np.zeros((num_steps*num_params,2))

    for i in range(0, num_params*num_steps,num_params):
            bounds[i] = x_bound
            bounds[i+1] = z_bound

    optimization_result = minimize(
                            fun=infid,
                            x0=params,
                            args=process_tensor,
                            method='L-BFGS-B',
                            jac=False,
                            bounds=bounds,
                            options = {'disp':True, 'gtol': 7e-05}
    )

    return optimization_result

def gradient_optimize(params,process_tensor,num_steps):

    # Set upper and lower bounds on control parameters
    x_bound = [-5*np.pi,5*np.pi]
    z_bound = [-np.pi,np.pi]

    bounds = np.zeros((num_steps*num_params,2))

    for i in range(0, num_params*num_steps,num_params):
        bounds[i] = x_bound
        bounds[i+1] = z_bound


    optimization_result = minimize(
                            fun=infidandgrad,
                            x0=params,
                            args=process_tensor,
                            method='L-BFGS-B',
                            jac=True,
                            bounds=bounds,
                            options = {'disp':True, 'gtol': 7e-05}
    )

    return optimization_result

optimization_results = []
num_calls = []
num_gradcalls = []

for parameters,process_tensor,steps in zip(parameters_list,process_tensor_list,steps_list):
    opt = optimize(parameters,process_tensor,steps)
    grad_opt = gradient_optimize(parameters,process_tensor,steps)

    num_calls.append(opt.nfev)
    num_gradcalls.append(grad_opt.nfev)
    
plt.plot(num_calls,bond_dims,label='no gradient')
plt.plot(num_gradcalls,bond_dims,label='with gradient')
plt.legend()
plt.show()

# -----------------------------------------------------------------------------

embed()
