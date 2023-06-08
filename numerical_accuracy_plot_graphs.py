"""
@author=ebutler414
plots comparison of numerical acuracy of PT

NOTE: to run code, go into path.py (in top level folder) and set the path to the
directory that this file is running from

"""

import oqupy
import numpy as np
import matplotlib.pyplot as plt
from path import dynamics_path,pt_path

# ~~~~~~~~~~ params ~~~~~~~~~~~~~
# ===============================
generate_process_tensors = True
generate_data = True

max_time = 5.0

# np.array([dt,dkmax,esprel])
params_list = [
    [0.0085,30,10**(-7)],
    [0.0085,30,10**(-8)],
    [0.0085,30,10**(-9)],]

# ===============================

# can't generate the array cause i need to know the number of timesteps, so will
# generate array after I load the number of timesteps
sigma_x_array = None
sigma_y_array = None
sigma_z_array = None


for i in range(len(params_list)):
    params = params_list[i]
    
    filename_states = 'states_dt{}dkmax{}esprel{}'.format(
            params[0],params[1],params[2])
    states_name_replaced = filename_states.replace('.','-')
    filename_times = 'times_dt{}dkmax{}esprel{}'.format(
            params[0],params[1],params[2])
    times_name_replaced = filename_times.replace('.','-')
    times = np.load(dynamics_path + times_name_replaced + '.npy')
    states = np.load(dynamics_path + states_name_replaced + '.npy')
    # converts ndarray to list of arrays (necessary to work with oqupy)
    states_list = []
    for i in range(states.shape[0]):
        states_list.append(states[i,:,:])

    dynamics = oqupy.Dynamics(times.tolist(),states_list)

    t,sigma_x = dynamics.expectations(oqupy.operators.sigma("x"),real=True)
    t,sigma_y = dynamics.expectations(oqupy.operators.sigma("y"),real=True)
    t,sigma_z = dynamics.expectations(oqupy.operators.sigma("z"),real=True)

    if sigma_x_array is None:
        sigma_x_array = np.zeros((len(params_list),times.size))
        sigma_y_array = np.zeros((len(params_list),times.size))
        sigma_z_array = np.zeros((len(params_list),times.size))

    # this doesn't work because the number of timesteps varies, 
    # needs to use list instead, then figure out how to subtract
    # values
    # sigma_x_array[i,:] = sigma_x
    # sigma_y_array[i,:] = sigma_y
    # sigma_z_array[i,:] = sigma_z

    plt.figure(0)
    plt.plot(times,sigma_x,label=r'$\sigma_x$' 
            + ',dt={},dkmax={},esprel={}'.format(
            params[0],params[1],params[2]))
    plt.figure(1)
    plt.plot(times,sigma_y,label=r'$\sigma_y$' 
            + ',dt={},dkmax={},esprel={}'.format(
            params[0],params[1],params[2]))
    plt.figure(2)
    plt.plot(times,sigma_z,label=r'$\sigma_z$' 
            + ',dt={},dkmax={},esprel={}'.format(
            params[0],params[1],params[2]))
plt.figure(0)
plt.legend()
plt.figure(1)
plt.legend()
plt.figure(2)
plt.legend()
plt.show()
    