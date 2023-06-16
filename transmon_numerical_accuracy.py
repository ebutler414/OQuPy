"""
@author=ebutler414
generates data for numerical accuracy of PT

NOTE: to run code, go into path.py (in top level folder) and set the path to the
directory that this file is running from

"""

import oqupy
import numpy as np
import matplotlib.pyplot as plt
from path import dynamics_path,pt_path

# ~~~~~~~~~~ params ~~~~~~~~~~~~~
# ===============================
generate_process_tensors = False
generate_data = True
print_bond_dimensions = False

max_time = 20.0

# np.array([dt,dkmax,esprel])
params_list = [

    [0.01,70,10**(-8)],
    [0.0085,82,10**(-8)], # same dkmax, smaller dt
    [0.01,70,10**(-7)],
    [0.01,80,10**(-8)],
    [0.01,90,10**(-8)],

    ]

# ===============================

sigma_x = oqupy.operators.sigma("x")
sigma_y = oqupy.operators.sigma("y")
sigma_z = oqupy.operators.sigma("z")
up_density_matrix = oqupy.operators.spin_dm("z+")
down_density_matrix = oqupy.operators.spin_dm("z-")
mixed_density_matrix = oqupy.operators.spin_dm("mixed")

if generate_process_tensors:

    def sd(w):
        return ((0.002) * w)/(1 + (w**2/(800*np.pi))**2)


    temperature = 0.01

    correlations = oqupy.CustomSD(j_function = sd ,
                                            cutoff = 400,
                                            cutoff_type = 'hard',
                                            temperature = temperature)

    bath = oqupy.Bath(sigma_x, correlations)


    for i in range(len(params_list)):
        params = params_list[i]

        tempo_parameters = oqupy.TempoParameters(
            dt=params[0],
            dkmax=params[1],
            epsrel=params[2])
        process_tensor = oqupy.pt_tempo_compute(bath=bath,
                                    start_time= 0,
                                    end_time= max_time,
                                    parameters=tempo_parameters)
        # generate a filename based off the parameters so we can import it in
        # future based off what parameters we want to use and not have to
        # generate the PT every single time (the curly braces and .format() call
        # inserts the variables at the time into the arrays)
        filename = 'pt_dt{}dkmax{}esprel{}maxtime{}'.format(
            params[0],params[1],params[2],max_time)
        # because the dt and possibly esprel have . in them and because these
        # are filenames, it might look like i'm calling this a file extension
        # (in case you don't know, file extensions on mac and linux are just a
        # convention and don't have any effect). While this isn't strictly
        # necessary, i'm going to substitute every . in the filename with a -
        # (that's what the next line does)
        filename_replaced = filename.replace('.','-')
        # pt_path specifies the path for a seperate folder (so we don't have all
        # of the pts dumped in our main folder, keep things tidy)
        # (also with strings, 'string a ' + 'string b' = 'string a string b'),
        # last term is filename (as stated above, not necessary)
        process_tensor.export(pt_path + filename_replaced + '.pt',
            overwrite=True)

if generate_data or print_bond_dimensions:
    # first, attempt to load all the process tensors, just to make sure they
    # exist

    # doing this seperately, so don't waste time doing loads of calculations
    # only for it to crash mid way because one PT didn't actually exist
    for i in range(len(params_list)):
        params = params_list[i]
        filename = 'pt_dt{}dkmax{}esprel{}maxtime{}'.format(
            params[0],params[1],params[2],max_time)
        filename_replaced = filename.replace('.','-')
        pt = oqupy.import_process_tensor(
            pt_path + filename_replaced + '.pt',
            process_tensor_type='simple')

        if print_bond_dimensions:
            print('bond dimensions for')
            print('dt = {}, dkmax = {}, esprel = {}'.format(
            params[0],params[1],params[2]))
            print(pt.get_bond_dimensions())

if generate_data:
    # now compute dynamics for each PT (need to reload PT, but that is quick)
    for i in range(len(params_list)):
        params = params_list[i]
        filename = 'pt_dt{}dkmax{}esprel{}maxtime{}'.format(
            params[0],params[1],params[2],max_time)
        filename_replaced = filename.replace('.','-')
        pt = oqupy.import_process_tensor(
            pt_path + filename_replaced + '.pt',
            process_tensor_type='simple')

        initial_state = mixed_density_matrix
        Omega = 8.0
        system = oqupy.System(0.5 * Omega * (sigma_z))
        dyns = oqupy.compute_dynamics(
            system=system,
            initial_state=initial_state,
            process_tensor=pt)
        states = dyns.states
        times = dyns.times

        filename_states = 'states_dt{}dkmax{}esprel{}maxtime{}'.format(
                params[0],params[1],params[2],max_time)
        states_name_replaced = filename_states.replace('.','-')
        filename_times = 'times_dt{}dkmax{}esprel{}maxtime{}'.format(
                params[0],params[1],params[2],max_time)
        times_name_replaced = filename_times.replace('.','-')

        np.save(dynamics_path+states_name_replaced,states)
        np.save(dynamics_path+times_name_replaced,times)

