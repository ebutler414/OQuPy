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
parser.add_argument('--param', type=float, nargs=4,help='Parameter ID')
args = parser.parse_args()

dt = args.param[0]  # dt
epsrel_pow = args.param[1] # epsrel (int)
epsrel = 10**(-epsrel_pow)  # convert to float
tcut=int(args.param[2])  # tcut (int)
tprot=int(args.param[3]) # protocol time
# Use param_id to select initial guess, config file, etc.
print(f'Running optimization with protocol time: {args.param} ps')

pt_parameters = {'epsrel':epsrel,
                 'alpha':0.05,
                 'omega_cutoff':1,                 
                 'temp':0.131,
                 'dt':dt,
                 'tcut':tcut}


omega_cutoff = pt_parameters['omega_cutoff']
alpha = pt_parameters['alpha']
temperature = pt_parameters['temp']
epsrel = pt_parameters['epsrel']
# dt = 1./omega_cutoff/np.sqrt(3)
dt=pt_parameters['dt']

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

u=0.01
correlationscf=oqupy.bath_correlations.CustomCountingSD(j_function=j,cutoff=omega_cutoff,u=u,
                                                 cutoff_type='exponential',temperature=temperature)

bathcf = oqupy.Bath(op.sigma("z")/2.0, correlationscf)

# load pre-computed process tensors


with open('results/processtensors_lowercoupling/processtensor_dt={0}_ps=-{1}_tcut={2}_alpha=0.05'.format(dt,epsrel_pow,tcut), 'rb') as f:

    processtensor = dill.load(f)

with open('results/processtensors_lowercoupling/processtensorCF_dt={0}_ps=-{1}_tcut={2}_alpha=0.05'.format(dt,epsrel_pow,tcut), 'rb') as f:

    processtensorcf = dill.load(f)

def discrete_hamiltonian(hx):
    return hx*op.sigma('x')/2

system = oqupy.ParameterizedSystem(discrete_hamiltonian)

target_derivative = -1j*np.identity(2)/u

# Cost Function
num_params=1
opt_log=[]

def heat(paras,process_tensor,num_steps):
    """""
    Take a numpy array [hx0, hz0, hx1, hz1, ...] over full timesteps and
    return the fidelity and gradient of the fidelity to the global target_derivative
    """
    control_times = np.linspace(0, t_prot, num_steps, endpoint=False)

    delta_t = interp1d(control_times, paras, kind='previous', fill_value="extrapolate")

    def hamiltonian_t(t):
        return delta_t(t)*oqupy.operators.sigma("x")/2

    system = oqupy.TimeDependentSystem(hamiltonian_t)

    num_steps_fine = int(np.round(t_prot / dt))

    dynamicscf = oqupy.compute_dynamics(
        process_tensor=process_tensor,
        system=system,
        initial_state=Rho_0,
        start_time=0,
        num_steps=num_steps_fine,
        progress_type='silent')
    
    return dynamicscf.states.trace(axis1=1,axis2=2).imag/u

import time

min_heats=[]
opt_runtimes=[
]

for t_prot in [tprot]:
    num_steps=int(t_prot/processtensor.dt)
    hx=np.ones(num_steps)*0.05
    #hx=control_dict['result'].x
    parameter_list=[item for pair in zip(hx) for item in pair]
    start = time.time()
    
    optimization_result = minimize(
                            fun=heat,
                            x0=parameter_list,
                            args=(processtensorcf,num_steps),
                            method='l-bfgs-b',
                            jac=True,
                            options = {'disp':True, 'gtol': 7e-04}
    )
    end = time.time()
    opt_runtimes.append(end-start)

    print("The minimal heat was found to be : ",optimization_result.fun)

    print("The Jacobian was found to be : ",optimization_result.jac)

opt_parameters = reshapedparas = [i for i in (optimization_result.x.reshape((-1,num_params))).tolist() for j in range(2)]
opt_parameters=np.array(opt_parameters)

grad_res_opt = oqupy.state_gradient(
    system=system,
    initial_state=Rho_0,
    target_derivative=op.spin_dm('mixed').T,
    process_tensors=[processtensor],
    num_steps=num_steps,
    parameters=opt_parameters,
    progress_type='silent')

folder="results/zero_control_lowercoupling/"
os.makedirs(folder,exist_ok=True)

subfolder = os.path.join(folder, 'dt={0}_ps={1}_tcut={2}_alpha=0.05_nonzero'.format(dt, np.round(np.log10(epsrel), 1), tcut))
os.makedirs(subfolder, exist_ok=True)

file_name1 = os.path.join(subfolder, '{0}ps'.format(tprot))

optimization_dict={"result":optimization_result,
                   "log":opt_log,
                   "fidelitygrad":grad_res_opt,
                   "opttime":opt_runtimes,
                   "parameters":pt_parameters
                   }

with open(file_name1,'wb') as f:
    dill.dump(optimization_dict,f)


