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
parser.add_argument('--param', type=float, nargs=4, help='Protocol time (int), dt (float), epsrel power (int) and tcut (int)')
args = parser.parse_args()

#t_arg = int(args.param[0])  # protocol time (int)
dt = args.param[0]  # dt
epsrel_pow = args.param[1] # epsrel (int)
epsrel = 10**(-epsrel_pow)  # convert to float
tcut=int(args.param[2])  # tcut (int)
u=args.param[3]


print(f'Running with dt : {dt} ps, epsrel power: -{epsrel_pow}, tcut: {tcut} ps')

pt_parameters = {'epsrel':epsrel,
                 'alpha':0.05,
                 'omega_cutoff':1,                 
                 'temp':0.131,
                 'dt':dt,
                 'tcut':tcut}


omega_cutoff = pt_parameters['omega_cutoff']
alpha = pt_parameters['alpha']
temperature = pt_parameters['temp']
#epsrel = pt_parameters['epsrel']
# dt = 1./omega_cutoff/np.sqrt(3)
#dt=pt_parameters['dt']

tcut=pt_parameters['tcut']

Rho_0=oqupy.operators.spin_dm('x+')

# spectral density (without cutoff)
def j(w):
    return 2*alpha*w

correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=1,
                                cutoff=omega_cutoff,
                                cutoff_type='exponential',
                                temperature=temperature)
bath = oqupy.Bath(op.sigma("z")/2.0, correlations)
parameters=oqupy.TempoParameters(dt=dt,epsrel=epsrel,dkmax=int(tcut/dt)+1)

correlationscf=oqupy.bath_correlations.CustomCountingSD(j_function=j,cutoff=omega_cutoff,u=u,
                                                 cutoff_type='exponential',temperature=temperature)

bathcf = oqupy.Bath(op.sigma("z")/2.0, correlationscf)

#num_steps=int(t_arg/dt)

pt= TTITempo(bath,start_time=0.0,parameters=parameters)
processtensor= pt.get_process_tensor()

pt_cf=TTITempoCounting(bathcf,start_time=0.0,parameters=parameters)
processtensorcf = pt_cf.get_process_tensor()

folder="results/processtensors_lowercoupling"
os.makedirs(folder,exist_ok=True)

# optimization result
file_name1=os.path.join(folder,'processtensor_dt={0}_ps={1}_tcut={2}_u={3}_alpha=0.05'.format(dt,np.round(np.log10(epsrel),1),tcut,u))
with open(file_name1,'wb') as f:
    dill.dump(processtensor,f)

file_name1=os.path.join(folder,'processtensorCF_dt={0}_ps={1}_tcut={2}_u={3}_alpha=0.05'.format(dt,np.round(np.log10(epsrel),1),tcut,u))
with open(file_name1,'wb') as f:
    dill.dump(processtensorcf,f)


