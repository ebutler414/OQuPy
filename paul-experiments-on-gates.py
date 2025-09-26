import oqupy
import oqupy.operators as op
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize,Bounds

from oqupy.iTEBD_TEMPO_useoqupybath import iTEBD_TEMPO_oqupy
from oqupy.process_tensor import TTInvariantProcessTensor
from oqupy.tti_tempo import TTITempo

from oqupy.gate_gradient import compute_dynamical_map,compute_dynamical_map_and_grad
from oqupy.gradient import state_gradient

pt_parameters = {'tmax':50,
                 'steps':100, 
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
num_steps = pt_parameters['steps']
dt = t_max/num_steps
dkmax = pt_parameters['dkmax']

initial_state = op.spin_dm('x+')
target_state = op.spin_dm('x-')
target_derivative = target_state.T

t_list = np.linspace(0,t_max,num_steps+1)

correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=1,
                                cutoff=omega_cutoff,
                                cutoff_type='exponential',
                                temperature=temperature)
bath = oqupy.Bath(op.sigma("z")/2.0, correlations)
parameters=oqupy.TempoParameters(dt=dt,epsrel=epsrel,dkmax=dkmax)

pt=TTITempo(bath,start_time=0.0,parameters=parameters)
process_tensor_tebd = pt.get_process_tensor()

def discrete_hamiltonian(hx,hz):
    return hx*op.sigma('x')/2 + hz*op.sigma('z')/2
system = oqupy.ParameterizedSystem(discrete_hamiltonian)

#%%
def fidelity_from_kraus(unitary,kraus_ops):
    d=unitary.shape[0]
    first_term=np.zeros((d,d),dtype=complex)
    second_term=0
    for kraus in kraus_ops:
        M=unitary.conj().T @ kraus

        first_term+=M.conj().T @ M
        second_term+=np.abs(np.trace(M))**2

    first_term=np.trace(first_term)

    fidelity=(first_term+second_term)/(d*(d+1))
    return fidelity

def choi_to_kraus(choi):
    vals,vecs=np.linalg.eigh(choi)
    kraus_ops=[]

    for val,vec in zip(vals,vecs.T):
        if val>1e-20:
            K=np.sqrt(val)*vec.reshape((2,2),order='C')
            kraus_ops.append(K)
    return kraus_ops

def superop_to_choi(map):
    d=2
    choi_matrix=np.zeros((d*d,d*d),dtype=complex)
    for i in range(d):
        for j in range(d):
            e_ij=np.zeros((d,d),dtype=complex)
            e_ij[i,j]=1.0

            phi_e_ij= map @ e_ij.flatten(order='C')
            phi_e_ij_mat = phi_e_ij.reshape((d,d), order='C')
            choi_matrix+=np.kron(e_ij,phi_e_ij_mat)
    return choi_matrix

def fidelity_from_superop(unitary,map):
    choi=superop_to_choi(map)
    kraus_ops=choi_to_kraus(choi)
    fidelity=fidelity_from_kraus(unitary,kraus_ops)
    return fidelity
#%%

def fidelities(hzval,hxval):
    h_z = np.ones(2*num_steps)*hzval
    h_x = np.ones(2*num_steps)*hxval

    parameters = np.vstack((h_x,h_z)).T
    #gradient=state_gradient(system=system,
    #                        initial_state=initial_state,
    #                        target_derivative=op.spin_dm('x-').T,
    #                        process_tensors=[process_tensor_tebd],
    #                        parameters=parameters,
    #                        num_steps=num_steps)

    map_list,grad_list=compute_dynamical_map_and_grad(system,
            [process_tensor_tebd],
            parameters,
            dt,
            start_time=0,
            num_steps=num_steps)

    states=[]
    for i in range(len(map_list)):
        stateati=np.matmul(initial_state.flatten(),map_list[i]).reshape((2,2))
        states.append(stateati)

    fidelities=[]
    for i in map_list:
        fid=fidelity_from_superop(np.identity(2),i)
        fidelities.append(fid)

    return fidelities

#%%
# Code to compute fidelity averaged over input states
#plt.plot(fidelities(0.0,omega_cutoff))
fig,ax=plt.subplots()
ax.plot(fidelities(0.0,omega_cutoff),label=r'$h_z=0.0, h_x=\omega_c$')
ax.plot(fidelities(0.0,0.5*omega_cutoff),label=r'$h_z=0.0, h_x=0.5\omega_c$')
ax.plot(fidelities(0.0,0.0*omega_cutoff),label=r'$h_z=0.0, h_x=0.0$')
ax.legend()
fig.show()

# %%
