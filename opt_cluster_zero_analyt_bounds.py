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
parser.add_argument('--param', type=float, nargs=7,help='Parameter ID')
args = parser.parse_args()

dt = args.param[0]  # dt
epsrel_pow = args.param[1] # epsrel (int)
epsrel = 10**(-epsrel_pow)  # convert to float
tcut=int(args.param[2])  # tcut (int)
tprot=int(args.param[3]) # protocol time

b_x=args.param[4] # symmetric parameter bounds for x,y,z
b_y=args.param[5]
b_z=args.param[6]


# Use param_id to select initial guess, config file, etc.
print(f'Running optimization with protocol time: {args.param} ps')

pt_parameters = {'epsrel':epsrel,
                 'alpha':0.1,
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

with open('OQuPy/results/processtensors/processtensor_dt={0}_ps=-{1}_tcut={2}'.format(dt,epsrel_pow,tcut), 'rb') as f:
    processtensor = dill.load(f)

with open('OQuPy/results/processtensors/processtensorcf_dt={0}_ps=-{1}_tcut={2}'.format(dt,epsrel_pow,tcut), 'rb') as f:
    processtensorcf = dill.load(f)

# for ParameterisedSystem2ls (NEED 3 PARAMS + no factors of 1/2)
def discrete_hamiltonian(hx,hy,hz):
    return hx*oqupy.operators.sigma("x") + hy*oqupy.operators.sigma("y") + hz*oqupy.operators.sigma("z")

system = oqupy.ParameterizedSystem2ls(discrete_hamiltonian)

target_derivative = -1j*np.identity(2)/u

# Cost Function
num_params=3
opt_log=[]

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
    fs_times=gradient_dict['dynamics']

    heat=np.trace(fs).imag/u

    heat_times=fs_times.states.trace(axis1=1,axis2=2).imag/u

    opt_log.append({
        "x":reshapedparas.copy(),
        "final heat": heat.copy(),
        "full heats":heat_times.copy()
    })

    # Adding adjacent elements
    for i in range(0,gps.shape[0],2): 
        gps[i,:]=gps[i,:]+gps[i+1,:]
        
    gps=gps[0::2]

    # Return the minus the gradient as infidelity is being minimized 
    return heat,(1.0*gps.reshape((-1)).real).tolist()



import time

min_heats=[]
opt_runtimes=[
]


for t_prot in [tprot]:
    num_steps=int(t_prot/processtensor.dt)
    hx=-np.ones(num_steps)*0.0000000005
    hy=np.zeros(num_steps)
    hz=np.zeros(num_steps)

        # Set upper and lower bounds on control parameters
    x_bound = [-b_x,b_x]
    y_bound = [-b_y,b_y] 
    z_bound = [-b_z,b_z]

    bounds = np.zeros((num_steps*num_params,2))

    for i in range(0, num_params*num_steps,num_params):
            bounds[i] = x_bound
            bounds[i+1] = y_bound
            bounds[i+2] = z_bound
        
    parameter_list=[item for pair in zip(hx,hy,hz) for item in pair]
    start = time.time()
    
    """
    def grad(z):
        return heatandgrad(z,processtensorcf,num_steps)[1]
    
    def func(z):
        return heatandgrad(z,processtensorcf,num_steps)[0]
    
    print(check_grad(func, grad, parameter_list))
    """
    optimization_result = minimize(
                            fun=heatandgrad,
                            x0=parameter_list,
                            args=(processtensorcf,num_steps),
                            method='l-bfgs-b',
                            jac=True,
                            bounds=bounds,
                            options = {'disp':True, 'gtol': 1e-10}
    )
    end = time.time()
    opt_runtimes.append(end-start)

    print("The minimal heat was found to be : ",optimization_result.fun)

    #print("The Jacobian was found to be : ",optimization_result.jac) (jac removed from scipy?)

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

folder='OQuPy/results/zero_control/conv_checks/'
os.makedirs(folder,exist_ok=True)

subfolder = os.path.join(folder, 'dt={0}_ps={1}_tcut={2}_x={3}_y={4}_z={5}'.format(dt, np.round(np.log10(epsrel), 1), tcut,b_x,b_z,b_y))
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


