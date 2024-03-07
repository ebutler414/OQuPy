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
'''
# -- time steps --
dt = 0.05
num_steps = 200

# -- bath --
alpha = 0.08
omega_cutoff = 4.0
temperature = 1.6
pt_dkmax = 40
pt_epsrel = 1.0e-5
'''

# parameters used in https://arxiv.org/abs/2303.16002
# -- time steps --
dt = 0.05 # 0.2
total_steps = 100# 20

# -- bath --
alpha =  0.126 #0.08
omega_cutoff = 3.04 #4
temperature = 5 * 0.1309 #1.6
pt_dkmax =60 # 40   60
pt_epsrel = 10**(-7) #1.0e-5

# -- initial and target state --
initial_state = op.spin_dm('x-')
target_state = op.spin_dm('x+')

# -- initial parameter guess --
y0 = np.zeros(2*total_steps)
z0 = np.ones(2*total_steps) * (np.pi) / (dt*total_steps)
x0 = np.zeros(2*total_steps)

parameter_list = list(zip(x0,y0,z0))
num_params = len(parameter_list[0])

y0 = np.zeros(total_steps)
z0 = np.ones(total_steps) * (np.pi) / (dt*total_steps)
x0 = np.zeros(total_steps)

test_parameter_list = list(zip(x0,y0,z0))
# --- Choose timestep of Fidelity -----------

num_steps = total_steps

end_step= total_steps
timesteps=None

if end_step == 0 or end_step == total_steps: 
     num_steps=total_steps
     timesteps=None
elif end_step <  total_steps:
    parameter_list = parameter_list[0:2*end_step]
    num_steps=end_step
    timesteps=range(2*end_step)
else:
    for i in range(2*total_steps,2*end_step):
        parameter_list.append(parameter_list[0])
    num_steps=end_step
    timesteps=range(2*end_step)


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
        h += 0.5* var * op.sigma(var_name)
    return h

parametrized_system = oqupy.ParameterizedSystem(hamiltonian)

# --- Compute fidelity, dynamics, and fidelity gradient -----------------------

from oqupy.gradient import state_gradient

fidelity_dict = state_gradient(
        system=parametrized_system,
        initial_state=initial_state,
        target_state=target_state.T,
        process_tensor=process_tensor,
        parameters=parameter_list,
        time_steps=timesteps,
        return_fidelity=True,
        return_dynamics=True)

print(f"the fidelity is {fidelity_dict['fidelity']}")
print(f"the fidelity gradient is {fidelity_dict['gradient']}")
t, s_x = fidelity_dict['dynamics'].expectations(op.sigma("x"))
t, s_y = fidelity_dict['dynamics'].expectations(op.sigma("y"))
t, s_z = fidelity_dict['dynamics'].expectations(op.sigma("z"))

plt.title("Pre-optimisation evolution of components")

plt.plot(t,s_x,label='x')
plt.plot(t,s_y,label='y')
plt.plot(t,s_z,label='z')

plt.legend()

# ----------- Optimisation of control parameters w.r.t. infidelity ---------------

def flatten_list(parameter_list):
    parameter_list_flat = [
    x
    for xs in parameter_list
    for x in xs
    ]
    return parameter_list_flat

def unflatten_list(flat_list):

    parameter_list = np.reshape(flat_list,(-1,num_params))
    return parameter_list

def copy_adjacent_elements(list:List)->List:
    double_list = []
    for entry in list:
         double_list.append(entry)
         double_list.append(entry)
    return double_list

def infidelity(parameter_list_flat):

    parameter_list = unflatten_list(parameter_list_flat)

    piecewiseconst_params=copy_adjacent_elements(parameter_list)

    return_dict = state_gradient(system=parametrized_system,
        initial_state=initial_state,
        target_state=target_state.T,
        process_tensor=process_tensor,
        parameters=piecewiseconst_params,
        time_steps=timesteps,
        return_fidelity=True,
        return_dynamics=False,
        dynamics_only=True)
    
    return -return_dict['fidelity'].real

# infidelity gradient to point optimiser in the right direction (note: minimising infidelity therefore jacobian multiplied by -1)
def fidelity_jacobian(parameter_list_flat):

    parameter_list = list(unflatten_list(parameter_list_flat))

    piecewiseconst_params=copy_adjacent_elements(parameter_list)

    fidelity_dict = state_gradient(
        system=parametrized_system,
        initial_state=initial_state,
        target_state=target_state.T,
        process_tensor=process_tensor,
        time_steps=timesteps,
        parameters=piecewiseconst_params)
    
    fidelity_jacobian = fidelity_dict['gradient']

    piecewiseconst_jacob = np.reshape(fidelity_jacobian,(num_steps,2,num_params)).sum(axis=1)

    flat_jacobian = flatten_list(piecewiseconst_jacob)

    fort_jac =np.asfortranarray(flat_jacobian)

    return -fort_jac.real

x_bound = [-5*np.pi,5*np.pi]
y_bound = [0,0] 
z_bound = [-np.pi,np.pi]

bounds = np.zeros((num_steps*num_params,2))

for i in range(0, num_params*num_steps,num_params):
        bounds[i] = x_bound
        bounds[i+1] = y_bound
        bounds[i+2] = z_bound

test_parameter_list_flat = flatten_list(test_parameter_list)

optimization_result = minimize(
                        fun=infidelity,
                        x0=test_parameter_list_flat,
                        method='L-BFGS-B',
                        jac=fidelity_jacobian,
                        bounds=bounds,
                        callback=infidelity,
                        options = {'disp':True, 'gtol': 7e-05}
)

print("The maximal fidelity was found to be : ",-optimization_result.fun)

print("The jacobian was found to be : ",optimization_result.jac)

optimized_parameters_flat = optimization_result.x

times = np.arange(0,num_steps)

optimized_parameters = np.reshape(optimized_parameters_flat,(num_steps,num_params))
optimized_parameters_double = copy_adjacent_elements(np.reshape(optimized_parameters_flat,(num_steps,num_params)))

optimized_dynamics = state_gradient(
        system=parametrized_system,
        initial_state=initial_state,
        target_state=target_state,
        process_tensor=process_tensor,
        parameters=optimized_parameters_double,
        time_steps=timesteps,
        return_fidelity=True,
        return_dynamics=True)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1) 
fig.suptitle("Optimisation results")

optimized_x = np.array([parameter[0] for parameter in optimized_parameters])
optimized_y = np.array([parameter[1] for parameter in optimized_parameters])
optimized_z = np.array([parameter[2] for parameter in optimized_parameters])

ax1.plot(times,optimized_x,label='x')
ax1.plot(times,optimized_y,label='y')
ax1.plot(times,optimized_z,label='z')

ax1.legend()

from scipy.linalg import sqrtm

dynamics = optimized_dynamics['dynamics']

t, bloch_x =optimized_dynamics['dynamics'].expectations(op.sigma("x"))
t, bloch_y = optimized_dynamics['dynamics'].expectations(op.sigma("y"))
t, bloch_z = optimized_dynamics['dynamics'].expectations(op.sigma("z"))

ax2.plot(t,bloch_x,label='x')
ax2.plot(t,bloch_y,label='y')
ax2.plot(t,bloch_z,label='z')
bloch_length = np.sqrt(bloch_x**2 +bloch_y**2 + bloch_z**2)
ax2.plot(t,bloch_length,label=r'$|\mathbf{\sigma}|$')

plt.legend()

# -----------------------------------------------------------------------------

embed()
