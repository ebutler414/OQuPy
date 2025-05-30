import oqupy
import oqupy.operators as op
import numpy as np
from oqupy.iTEBD_TEMPO_useoqupybath import iTEBD_TEMPO_oqupy
from oqupy.process_tensor import TTInvariantProcessTensor
from oqupy.tti_tempo import TTITempo
from oqupy.tti_tempo import TTITempoCounting
import matplotlib.pyplot as plt
import dill

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize,Bounds

pt_parameters = {'epsrel':10**(-7),
                 'alpha':0.1,
                 'omega_cutoff':1,                 
                 'temp':0.131,
                 'dt':0.25}

omega_cutoff = pt_parameters['omega_cutoff']
alpha = pt_parameters['alpha']
temperature = pt_parameters['temp']
epsrel = pt_parameters['epsrel']
# dt = 1./omega_cutoff/np.sqrt(3)
dt=pt_parameters['dt']

tcut=27

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

# converged parameters for different protocol times 
protocol_times=[10,50,70,100,200]

with open('processtensor_simplemodel', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    processtensor = dill.load(f)

with open('processtensorCF_simplemodel', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    processtensorcf = dill.load(f)

def discrete_hamiltonian(hx):
    return hx*op.sigma('x')/2

system = oqupy.ParameterizedSystem(discrete_hamiltonian)

target_derivative = -1j*np.identity(2)/u

# Cost Function
num_params=1

def heatandgrad(paras,process_tensor,num_steps):
    """""
    Take a numpy array [hx0, hz0, hx1, hz1, ...] over full timesteps and
    return the fidelity and gradient of the fidelity to the global target_derivative
    """

    # Reshape flat parameter list to form accepted by state_gradient: [[hx0,hz0],[hx1,hz1,]...]
    reshapedparas = [i for i in (paras.reshape((-1,num_params))).tolist() for j in range(2)]
    #parameter_list=np.array([itemopt_steps=[9] for pair in zip(paras[:num_steps],paras[num_steps:2*num_steps]) for item in pair])
    #reshapedparas = [i for i in (parameter_list.reshape((-1,num_params))).tolist() for j in range(2)]
    reshapedparas = np.array(reshapedparas)

    gradient_dict = oqupy.state_gradient(
        system=system,
        initial_state=Rho_0,
        target_derivative=target_derivative,
        process_tensors=[process_tensor],
        parameters=reshapedparas,
        num_steps=num_steps,
        progress_type='silent')
    
    fs=gradient_dict['final_state']
    gps=gradient_dict['gradient']

    heat=np.trace(fs).imag/u

    # Adding adjacent elements
    for i in range(0,gps.shape[0],2): 
        gps[i,:]=gps[i,:]+gps[i+1,:]
        
    gps=gps[0::2]

    x=[]
    for i in range(0,gps.shape[1]): 
        x.append(gps[:,i])
    
    gps=np.array(x)

    # Return the minus the gradient as infidelity is being minimized 
    return heat,(1.0*gps.reshape((-1)).real).tolist()

import time

opt_dict={protocol_times[0]:[],
                 protocol_times[1]:[], 
                 protocol_times[2]:[],
                 protocol_times[3]:[],
                 protocol_times[4]:[]}

min_heats=[]
opt_runtimes=[
]

for t_prot in [50]:
    num_steps=int(t_prot/processtensor.dt)
    hx=np.zeros(num_steps)
    parameter_list=[item for pair in zip(hx) for item in pair]
    start = time.time()
    
    optimization_result = minimize(
                            fun=heatandgrad,
                            x0=parameter_list,
                            args=(processtensorcf,num_steps),
                            method='l-bfgs-b',
                            jac=True,
                            options = {'disp':True, 'gtol': 7e-04}
    )
    end = time.time()
    opt_runtimes.append(end-start)

    opt_dict[t_prot]=optimization_result

    min_heats.append(optimization_result.fun)

    print("The minimal heat was found to be : ",optimization_result.fun)

    print("The Jacobian was found to be : ",optimization_result.jac)

# processtensor_simplemodel
file_name1='optimization_simplemodel'
with open(file_name1,'wb') as f:
    dill.dump(optimization_result,f)