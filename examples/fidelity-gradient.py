#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0,'.')
from IPython import embed

import numpy as np
import matplotlib.pyplot as plt

import oqupy
import oqupy.operators as op

from scipy.optimize import minimize
from typing import List,Tuple


# --- Parameters --------------------------------------------------------------

# -- time steps --
dt = 0.2
num_steps = 20

# -- bath --
alpha = 0.08
omega_cutoff = 4.0
temperature = 1.6
pt_dkmax = 40
pt_epsrel = 1.0e-5

# -- initial and target state --
initial_state = op.spin_dm('z-')
target_state = op.spin_dm('z+')

# -- initial parameter guess --
x0 = np.zeros(2*num_steps)
y0 = np.ones(2*num_steps) * (np.pi/2) / (dt*num_steps)
z0 = np.zeros(2*num_steps)

parameter_list = list(zip(x0,y0,z0))

# --- Compute process tensors -------------------------------------------------

correlations = oqupy.PowerLawSD(
    alpha=alpha,
    zeta=1,
    cutoff=omega_cutoff,
    cutoff_type='exponential',
    temperature=temperature)
bath = oqupy.Bath(0.5 * op.sigma('y'), correlations)
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

# --- Define parametrized system ----------------------------------------------

def hamiltonian(x, y, z):
    h = np.zeros((2,2),dtype='complex128')
    for var, var_name in zip([x,y,z], ['x', 'y', 'z']):
        h += var * op.sigma(var_name)
    return h

parametrized_system = oqupy.ParameterizedSystem(hamiltonian)

# --- Compute fidelity, dynamics, and fidelity gradient -----------------------

from oqupy.gradient import state_gradient, forward_backward_propagation
from oqupy.contractions import compute_dynamics

fidelity_dict = state_gradient(
        system=parametrized_system,
        initial_state=initial_state,
        target_state=target_state,
        process_tensor=process_tensor,
        parameters=parameter_list,
        return_fidelity=True,
        return_dynamics=True)

print(f"the fidelity is {fidelity_dict['fidelity']}")
print(f"the fidelity gradient is {fidelity_dict['gradient']}")
t, s_x = fidelity_dict['dynamics'].expectations(op.sigma("x"))
plt.plot(t,s_x)

# ----------- Optimisation of control parameters w.r.t. infidelity ---------------

def flatten_list(parameter_list):
    assert np.shape(parameter_list) == (2*process_tensor.__len__(),3)
    parameter_list_flat = [
    x
    for xs in parameter_list
    for x in xs
    ]
    print(np.shape(parameter_list_flat))
    return parameter_list_flat

def unflatten_list(flat_list):
    assert np.shape(flat_list) == (2*3*process_tensor.__len__(),)
    parameter_list = np.reshape(flat_list,(2*process_tensor.__len__(),3))
    return parameter_list

def infidelity(parameter_list_flat):

    parameter_list_var = unflatten_list(parameter_list_flat)

    return_dict = state_gradient(system=parametrized_system,
        initial_state=initial_state,
        target_state=target_state,
        process_tensor=process_tensor,
        parameters=parameter_list_var,
        return_fidelity=True,
        return_dynamics=False)
    
    return -float(return_dict['fidelity'])

def fidelity_jacobian(parameter_list_flat):

    parameter_list_var = unflatten_list(parameter_list_flat)

    fidelity_dict = state_gradient(
        system=parametrized_system,
        initial_state=initial_state,
        target_state=target_state,
        process_tensor=process_tensor,
        parameters=parameter_list_var)

    fidelity_jacobian = fidelity_dict['gradient']

    fort_jac =np.asfortranarray(flatten_list(fidelity_jacobian))

    return fort_jac


optimization_result = minimize(
                        fun=infidelity,
                        x0=flatten_list(parameter_list),
                        method='L-BFGS-B',
                        jac=fidelity_jacobian,
                        callback=infidelity,
                        options = {'disp':True}
)

print("The maximal fidelity was found to be : ",-optimization_result.fun)

print("The jacobian was found to be : ",optimization_result.jac)

optimized_parameters_flat = optimization_result.x


print(max(abs(optimized_parameters_flat)))
print(min(abs(optimized_parameters_flat)))

times = np.arange(0,40)
optimized_parameters = unflatten_list(optimized_parameters_flat)

optimized_dynamics = state_gradient(
        system=parametrized_system,
        initial_state=initial_state,
        target_state=target_state,
        process_tensor=process_tensor,
        parameters=optimized_parameters,
        return_fidelity=True,
        return_dynamics=True)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1) 

optimized_x = np.array([parameter[0] for parameter in optimized_parameters])
optimized_y = np.array([parameter[1] for parameter in optimized_parameters])
optimized_z = np.array([parameter[2] for parameter in optimized_parameters])

ax1.plot(times,optimized_x,label='x')
ax1.plot(times,optimized_y,label='y')
ax1.plot(times,optimized_z,label='z')

ax1.legend()

dynamics = optimized_dynamics['dynamics']

hs_dim = 2
v_final_state = target_state.reshape(hs_dim**2)
fidelity=[]

for state in dynamics.states:
    v_state = state.reshape(hs_dim**2)
    fidelity.append(v_state@v_final_state.T)

times = np.arange(0,21)

ax2.plot(times,fidelity)

#print(f"the fidelity is {fidelity_dict['fidelity']}")
#print(f"the fidelity gradient is {fidelity_dict['gradient']}")
#t, s_x = fidelity_dict['dynamics'].expectations(op.sigma("x"))
#plt.plot(t,s_x)

plt.legend()

# -----------------------------------------------------------------------------

embed()
