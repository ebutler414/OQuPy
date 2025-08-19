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
parser.add_argument('--param', type=float, nargs=1, help='Protocol time (int), dt (float), epsrel power (int) and tcut (int)')
args = parser.parse_args()

t_prot=args.param[0]

import oqupy
import oqupy.operators as op
import numpy as np
from oqupy.iTEBD_TEMPO_useoqupybath import iTEBD_TEMPO_oqupy
from oqupy.process_tensor import TTInvariantProcessTensor
from oqupy.tti_tempo import TTITempo
from oqupy.tti_tempo import TTITempoCounting
import matplotlib.pyplot as plt


from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize,Bounds

pt_parameters = {'tmax':50,
                 'dkmax':50,
                 'epsrel':10**(-6),
                 'alpha':0.1,
                 'omega_cutoff':1,                 
                 'temp':0.131}

omega_cutoff = pt_parameters['omega_cutoff']
alpha = pt_parameters['alpha']
temperature = pt_parameters['temp']
epsrel = pt_parameters['epsrel']
t_max = pt_parameters['tmax']
# dt = 1./omega_cutoff/np.sqrt(3)
dkmax = pt_parameters['dkmax']

tcut=50

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

correlationscf=oqupy.bath_correlations.CustomCountingSD(j_function=j,cutoff=omega_cutoff,u=0.01,
                                                 cutoff_type='exponential',temperature=temperature)

bathcf = oqupy.Bath(op.sigma("z")/2.0, correlationscf)

with open('OQuPy/results/processtensors/processtensorCF_dt=0.25_ps=-7.0_tcut=50', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    processtensor = dill.load(f)

with open('OQuPy/results/processtensors/processtensorCF_dt=0.25_ps=-7.0_tcut=50', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    processtensorCF = dill.load(f)

file_name1='OQuPy/results/zero_control/500ps/optimization_simplemodel_500ps'.format(t_prot)
with open(file_name1, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    opt_dict= dill.load(f)

from scipy.interpolate import interp1d

opt_control=opt_dict['result'].x
delta_t = interp1d(np.linspace(0,t_prot,len(opt_control)),opt_control)
def hamiltonian_t(t):
    return delta_t(t)*oqupy.operators.sigma("x")/2

system = oqupy.TimeDependentSystem(hamiltonian_t)

num_steps=int(t_prot/processtensor.dt)
processtensor.length=num_steps


bath_corr = oqupy.bath_dynamics.TwoTimeBathCorrelations(system, bath, processtensor, Rho_0)

tlist,occ=bath_corr.occupation(1,0.1,change_only=True)

dw=0.01
freq_list = np.arange(dw, 1, dw)
occ_freq_t=np.zeros((len(tlist),len(freq_list)))
heat_freq_t=np.zeros((len(tlist),len(freq_list)))

folder="OQuPy/results"


for i,w in enumerate(freq_list):
    tlist,occ=bath_corr.occupation(w,dw,change_only=True)
    for j,t in enumerate(tlist):
        occ_freq_t[j][i]=occ[j]
        heat_freq_t[j][i]=w*occ[j]

heatmap_dict={'occupations_map':occ_freq_t,
              'heat_map':occ_freq_t,
              'bath_corr':bath_corr}

file_name1=os.path.join(folder,'heatmaps/{0}ps'.format(t_prot))
with open(file_name1,'wb') as f:
    dill.dump(heatmap_dict,f)

    