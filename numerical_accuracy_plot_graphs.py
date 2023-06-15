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


max_time = 20.0

# plot difference will only work if all of the timesteps in the comparison list
# are the same, otherwise will give error. (you could figure out how to do this,
# but i haven't bothered)
plot_difference = True
# index of the list to sum the difference between
reference_index = 2

# np.array([dt,dkmax,esprel])
# dkmax test, but with better params
params_list = [

    [0.01,70,10**(-8)],
    # [0.0085,82,10**(-8)], # same dkmax, smaller dt
    [0.01,70,10**(-7)],
    [0.01,80,10**(-8)],
    [0.01,90,10**(-8)],

    ]

# ===============================

# can't generate the array cause i need to know the number of timesteps, so will
# generate array after I load the number of timesteps
sigma_x_array = None
sigma_y_array = None
sigma_z_array = None


if plot_difference:
    # check that all of the dts are the same if taking difference between
    # parameters
    for i in range(len(params_list)):
        assert params_list[0][0] == params_list[reference_index][0],\
            'if subtracting the difference between arrays,'\
            + 'arrays must be the same length, i.e. same timestep'

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
    for j in range(states.shape[0]):
        states_list.append(states[j,:,:])

    dynamics = oqupy.Dynamics(times.tolist(),states_list)

    t,sigma_x = dynamics.expectations(oqupy.operators.sigma("x"),real=True)
    t,sigma_y = dynamics.expectations(oqupy.operators.sigma("y"),real=True)
    t,sigma_z = dynamics.expectations(oqupy.operators.sigma("z"),real=True)

    if plot_difference:
        if sigma_x_array is None:
            # have checked that all of the lengths are the same
            sigma_x_array = np.zeros((len(params_list),times.size))
            sigma_y_array = np.zeros((len(params_list),times.size))
            sigma_z_array = np.zeros((len(params_list),times.size))

        sigma_x_array[i,:] = sigma_x
        sigma_y_array[i,:] = sigma_y
        sigma_z_array[i,:] = sigma_z

    plt.figure(2)
    plt.plot(times,sigma_x,label=r'$\sigma_x$'
            + ',dt={},dkmax={},esprel={}'.format(
            params[0],params[1],params[2]))
    plt.figure(1)
    plt.plot(times,sigma_y,label=r'$\sigma_y$'
            + ',dt={},dkmax={},esprel={}'.format(
            params[0],params[1],params[2]))
    plt.figure(0)
    plt.plot(times,sigma_z,label=r'$\sigma_z$'
            + ',dt={},dkmax={},esprel={}'.format(
            params[0],params[1],params[2]))
if plot_difference:
    colours = ["C0", "C1", "C2", "C3", "C4", 
                "C5", "C6", "C7", "C8", "C9"]
    plt.figure(3)
    for i in range(len(params_list)):
        # don't subtract the reference from itself....
        if i != reference_index:
            difference_x = sigma_x_array[i,:] - sigma_x_array[reference_index,:]
            difference_y = sigma_y_array[i,:] - sigma_y_array[reference_index,:]
            difference_z = sigma_z_array[i,:] - sigma_z_array[reference_index,:]

            plt.plot(times,difference_x,linestyle='dashed',color=colours[i])
            plt.plot(times,difference_y,linestyle='dashdot',color=colours[i])
            plt.plot(times,difference_z,color=colours[i],
                    label='dt = {}, dkmax = {}, esprel = {}'.format(
                    params_list[i][0], params_list[i][1], params_list[i][2]))
    
    plt.legend()
    plt.title('difference between dt = {}, dkmax = {} esprel = {}, and:'.format(
        params_list[reference_index][0], params_list[reference_index][1], 
        params_list[reference_index][2]))

plt.figure(0)
plt.legend()
plt.figure(1)
plt.legend()
plt.figure(2)
plt.legend()
plt.show()
