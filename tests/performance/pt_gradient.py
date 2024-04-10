import sys
sys.path.insert(0,'.')
from IPython import embed

import numpy as np
import matplotlib.pyplot as plt

import oqupy
import oqupy.operators as op

from typing import List,Tuple

"""
Performance tests for PT-Gradient computations
"""

def pt_gradient_performance_A(process_tensor_name,
                          number_of_pts,
                          number_of_steps,
                          model,
                          pt_epsrel):
    """
    Checks runtime scaling of gradient and dynamics computations with process duration

    Parameters
    ----------
    process_tensor_name:
        Name of the process tensor that represents the environment (following
        the convention defined in /tests/data/generate_pts.py)
    number_of_steps:
        Total number of steps in the propagation.
    number_of_pts:
        Number of environments
    pt_epsrel:
        Relative SVD cutoff of the PT-TEMPO algorithm.
"""
    N = number_of_steps
    K = number_of_pts
    dt=0.1

    """"
    # -- process tensor --
    pt_file_path = os.path.join(PT_DIR_PATH, f"{process_tensor_name}.hdf5")
    pt = oqupy.import_process_tensor(pt_file_path)
    """
    pt=oqupy.pt_tempo_compute(
    bath=bath,
    start_time=0.0,
    end_time=N * dt,
    parameters=pt_tempo_parameters,
    progress_type='bar')

    T=N*dt

    # -- chain hamiltonian --
    if model=="z pi pulse":
        h = np.array([[0.0, np.pi / T, 0.0]]*2*N)
    elif model=="":
        h = np.array([[0.0, 0.0, 1.0]]*N)
        J = np.array([[1.3, 0.7, 1.2]]*(N-1))
    else:
        raise ValueError(f"Model '{model}' not implemented!")

    system_chain = oqupy.ParameterizedSystem(h)

    # -- initial state --
    initial_augmented_mps = oqupy.AugmentedMPS(
        [op.spin_dm("z+")] + [op.spin_dm("z-")] * (N-1))

    # -- pt-tebd computation --
    pt_tebd_params = oqupy.PtTebdParameters(
        dt=pt.dt,
        order=tebd_order,
        epsrel=tebd_epsrel)

    pt_tebd = oqupy.PtTebd(
        initial_augmented_mps=initial_augmented_mps,
        system_chain=system_chain,
        process_tensors=[pt]*K+[None]*(N-K),
        parameters=pt_tebd_params,
        dynamics_sites=list(range(N)))
    
""""
 # Parameter set intended to check convergence with PTs epsrel.
parameters_A1 = [
    ["spinBoson_alpha0.16_zeta1.0_T0.0_cutoff1.0exp_tcut4.0_dt04_steps06_epsrel15",
     "spinBoson_alpha0.16_zeta1.0_T0.0_cutoff1.0exp_tcut4.0_dt04_steps06_epsrel16",
     "spinBoson_alpha0.16_zeta1.0_T0.0_cutoff1.0exp_tcut4.0_dt04_steps06_epsrel17"], # process_tensor_name
    [7],                             # number_of_sites
    [2],                             # number_of_pts
    ["XY"],                          # model
    [2**(-15), 2**(-16), 2**(-17)],  # tebd_epsrel
"""

    start_time = time.time()
    result = pt_tebd.compute(len(pt))
    result['walltime'] = time.time()-start_time
    result['N'] = N
    result['K'] = K
    result['model'] = model
    result['tebd_esprel'] = tebd_epsrel

    return result
