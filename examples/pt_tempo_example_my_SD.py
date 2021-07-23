import os
from time import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import time_evolving_mpo as tempo
import numpy as np
import matplotlib.pyplot as plt

from scipy import constants

from solve_IBM import coherent_I_B
coherent_IB_vec = np.vectorize(coherent_I_B)

Omega = 0.0 # 1.0
# omega_cutoff = 3.0
# alpha = 0.1
omega_cutoff = 2.0e-3* constants.electron_volt * 1e-12 / constants.hbar
alpha = constants.hbar * 11.2e-3 * omega_cutoff**2 * 1e12/ \
        (2 * constants.pi * constants.Boltzmann)

system = tempo.System(0.0 * Omega * tempo.operators.sigma("x"))
initial_state = tempo.operators.spin_dm("y+")
def generate_pt(temp):
    temp_new_units = temp * constants.Boltzmann * 1e-12 / constants.hbar
    start_time = 0.0
    end_time = 10.0
    
    correlations = tempo.PowerLawSD(alpha=alpha, 
                                    zeta=3, 
                                    cutoff=omega_cutoff, 
                                    cutoff_type='gaussian',
                                    temperature=temp_new_units)
    bath = tempo.Bath(0.5 * tempo.operators.sigma("z"), correlations)

    dt = 0.1 # 0.01
    dkmax = 20 # 200
    epsrel = 1.0e-6# 1.0e-7


    tempo_parameters = tempo.TempoParameters(dt=dt, dkmax=dkmax, epsrel=epsrel)

    pt_tempo_parameters = tempo.PtTempoParameters(dt=dt, dkmax=dkmax, epsrel=epsrel)

    pt = tempo.pt_tempo_compute(bath=bath,
                                start_time=start_time,
                                end_time=end_time,
                                parameters=pt_tempo_parameters,
                                progress_type='bar')
    pt.export("details_pt_tempo_{}K.processTensor".format(temp),overwrite=True)

def generate_lots_of_PTs(temp_list):
    for i in temp_list:
        generate_pt(i)



def plot_lots_of_PTs(temp_list):
    colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    for i,temp in enumerate(temp_list):

        pt = tempo.import_process_tensor("details_pt_tempo_{}K.processTensor".format(temp))

        st = time()
        dyn = pt.compute_dynamics_from_system(system=system, initial_state=initial_state)
        et = time()
        print('Time = {}'.format(et-st))
        print(dyn)

        t2, s2_y = dyn.expectations(tempo.operators.sigma("y"), real=True)

        plt.plot(t2, s2_y, label=r'{}K'.format(temp),color=colours[i],zorder=6-i)
        k_B_new = constants.Boltzmann * 10**(-12) / (constants.hbar)

        s3_y_analytic = coherent_IB_vec(t2,1/(temp*k_B_new),0,0,0)

        plt.plot(t2,s3_y_analytic,linestyle='dashed',zorder=6-i)
        
    def analytic_solution(t):
        x = (t*omega_cutoff)**2
        phi = 2 * alpha * (1 + (x-1)/(x+1)**2)
        return np.exp(-phi)


    sy_exact = analytic_solution(t2)
    #plt.plot(t2, sy_exact, label=r'gerlalds exact', linestyle="dotted")


    plt.xlabel(r'$t(ps)$')
    plt.ylabel(r'$\langle \sigma_y\rangle$')
    plt.legend(loc=4)
    plt.tight_layout()


temp_list  = np.array([0.0001,5,20])
# temp_list = np.array([0.0001,0.01,1,3,5])

# generate_lots_of_PTs(temp_list)
plot_lots_of_PTs(temp_list)


plt.show()