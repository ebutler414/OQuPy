import oqupy
import oqupy.operators as op
import numpy as np
from oqupy.iTEBD_TEMPO_useoqupybath import iTEBD_TEMPO_oqupy
from oqupy.process_tensor import TTInvariantProcessTensor
from oqupy.tti_tempo import TTITempo
from oqupy.tti_tempo import TTITempoCounting
import matplotlib.pyplot as plt
import dill
import os
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize,Bounds

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--param', type=float, nargs=3, help='Protocol time (int), dt (float), epsrel power (int) and tcut (int)')
args = parser.parse_args()

#t_arg = int(args.param[0])  # protocol time (int)
dt = args.param[0]  # dt
epsrel_pow = args.param[1] # epsrel (int)
epsrel = 10**(-epsrel_pow)  # convert to float
tcut=int(args.param[2])  # tcut (int)

#folder="OQuPy/results/processtensors"
folder="results/processtensors"
os.makedirs(folder,exist_ok=True)

# fetch process tensors
file_name1=os.path.join(folder,'processtensor_dt={0}_ps={1}_tcut={2}'.format(dt,np.round(np.log10(epsrel),1),tcut))
with open(file_name1,'rb') as f:
    pt=dill.load(f)

file_name1=os.path.join(folder,'processtensorCF_dt={0}_ps={1}_tcut={2}'.format(dt,np.round(np.log10(epsrel),1),tcut))
with open(file_name1,'rb') as f:
    ptcf=dill.load(f)


print(f'Running with dt : {dt} ps, epsrel power: -{epsrel_pow}, tcut: {tcut} ps')

pt_parameters = {'epsrel':epsrel,
                 'alpha':0.1,
                 'omega_cutoff':1,                 
                 'temp':0.131,
                 'dt':dt,
                 'tcut':tcut}


omega_cutoff = pt_parameters['omega_cutoff']
alpha = pt_parameters['alpha']
temperature = pt_parameters['temp']

opt_dict={}

protocol_times=[500,1000,5000,8000]
for t_prot in protocol_times:
# processtensor_simplemodel
    file_name1='OQuPy/results/zero_control/{0}ps/optimization_simplemodel_{0}ps'.format(t_prot)
    with open(file_name1,'rb') as f:
        opt_dict[t_prot]=dill.load(f)

# System setup

Rho_0=oqupy.operators.spin_dm('x+')

num_params=1

for t_prot in protocol_times:
    opt_control=opt_dict[t_prot]['result'].x
    delta_t = interp1d(np.linspace(0,t_prot,len(opt_control)),opt_control)
    def hamiltonian_t(t):
        return delta_t(t)*oqupy.operators.sigma("x")/2
    
    system = oqupy.TimeDependentSystem(hamiltonian_t)
    
    num_steps=len(opt_control)
    dynamics=oqupy.compute_dynamics(
    process_tensor=pt,        
    system=system,
    initial_state=Rho_0,
    start_time=0,
    num_steps=num_steps)
    t, s_x = dynamics.expectations(op.sigma('x'), real=True)

    # compute heats

    dynamicscf = oqupy.compute_dynamics(
    process_tensor=ptcf,        
    system=system,
    initial_state=Rho_0,
    start_time=0,
    num_steps=num_steps)


    folder="OQuPy/results/optimised_longer/dt={0}ps_eps={1}_tcut={2}/".format(dt,np.round(np.log10(epsrel),1),tcut)

    os.makedirs(folder,exist_ok=True)

    long_dict={"dynamics":dynamics,
            "dynamicscf":dynamicscf,
            "parameters":pt_parameters}

    # optimization result
    file_name1=os.path.join(folder,'{0}ps'.format(t_prot))
    with open(file_name1,'wb') as f:
        dill.dump(long_dict,f)

