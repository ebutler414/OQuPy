# %%
import oqupy
import oqupy.operators as op
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize,Bounds
from oqupy.iTEBD_TEMPO_useoqupybath import iTEBD_TEMPO_oqupy
from oqupy.process_tensor import TTInvariantProcessTensor
from oqupy.tti_tempo import TTITempo
from oqupy.gate_gradient import compute_dynamical_map,compute_dynamical_map_and_grad
from oqupy.gate_gradient import gate_chain_rule

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

h_z = np.ones(2*num_steps)*0
h_x = np.ones(2*num_steps)*omega_cutoff

parameters = np.vstack((h_x,h_z)).T


# %%
def map_and_grad(system,process_tensor_tebd,parameters,dt,num_steps):
        map_list,grad_list=compute_dynamical_map_and_grad(system,
                [process_tensor_tebd],
                parameters,
                dt,
                start_time=0,
                num_steps=num_steps)

        num_parameters = parameters.shape[1]

        propagators=system.get_propagators(dt,parameters)
        prop_derivs=system.get_propagator_derivatives(dt,parameters)

        dyn_map_derivatives=gate_chain_rule(adjoint_tensor=grad_list,propagators=propagators,dprop_dparam=prop_derivs,num_steps=num_steps,num_parameters=num_parameters)

        return map_list,dyn_map_derivatives


map_list,dyn_map_derivs=map_and_grad(system,process_tensor_tebd,parameters,dt,num_steps)
map_list[0]=np.array([map_list[0]]) # for some reason map_list[0] is missing an outer [] compared with the others
map_list=[i[0] for i in map_list] # make map_list a more sensible arrangement


# %%
def outstate(dynmap,instate):
    # return the final state for a given input state
    return np.matmul(instate.flatten(),dynmap)

finalstates=np.array([outstate(i,oqupy.operators.spin_dm('z+')) for i in map_list]) # dyn_map[0] has a different structure

def objective(dynmap,instate,target):
    # a simple objective corresponding to the fidelity between the output state from input state, and a target state 
    outputstate=np.matmul(instate.flatten(),dynmap)
    return np.dot(target.T.flatten(),outputstate)

obfunvals=np.array([objective(i,oqupy.operators.spin_dm('z+'),oqupy.operators.spin_dm('y+')) for i in map_list])
sx=np.array([np.trace(np.matmul(oqupy.operators.sigma('x'),np.reshape(i,(2,2)))) for i in finalstates])
sy=np.array([np.trace(np.matmul(oqupy.operators.sigma('y'),np.reshape(i,(2,2)))) for i in finalstates])
sz=np.array([np.trace(np.matmul(oqupy.operators.sigma('z'),np.reshape(i,(2,2)))) for i in finalstates])
plt.plot(sx.real,label='x')
plt.plot(sy.real,label='y')
plt.plot(sz.real,label='z')
plt.plot(obfunvals.real,label='Fidelity to y+')#
plt.legend()



# %%
#%%%

# %%
fidelity=rotation_fidelity(target_unitary,kraus_ops).real

fidelity_grads=[]
for map in map_list:
    fidelity_grads.append(fidelity_map_deriv(map,target_unitary))

map_grads=dyn_map_derivatives
# Adding adjacent elements
for i in range(0,map_grads.shape[0],2): 
    map_grads[i,:]=map_grads[i,:]+ map_grads[i+1,:]
        
map_grads=map_grads[0::2]

grads = np.sum(map_grads.sum(axis=1) * fidelity_grads, axis=(1,2))
plt.plot(grads)

# %%

import scipy as sp
def infidandgrad(paras):
    """""
    Take a numpy array [hx0, hz0, hx1, hz1, ...] over full timesteps and
    return the fidelity and gradient of the fidelity to the global target_derivative
    """

    # Reshape flat parameter list to form accepted by state_gradient: [[hx0,hz0],[hx1,hz1,]...]
    reshapedparas = [i for i in (paras.reshape((-1,num_parameters))).tolist() for j in range(2)]
    reshapedparas = np.array(reshapedparas)

    maps,map_grads=map_and_grad(system,process_tensor_tebd,reshapedparas,dt,num_steps)

    # Adding adjacent elements
    for i in range(0,map_grads.shape[0],2): 
        map_grads[i,:]=map_grads[i,:]+ map_grads[i+1,:]
            
    map_grads=map_grads[0::2]
    
    choi_matrix=superop_to_choi(maps[-1])
    kraus_ops=choi_to_kraus(choi_matrix)

    #target_unitary = np.array(1j*op.sigma("x"), dtype=complex)  # pi rotation around x-axis
    #target_unitary = sp.linalg.expm(-1j*omega_cutoff*((num_steps-1)+1)*dt*op.sigma('x')/2)
    target_unitary=np.identity(2)
    # target unitary is rotation around x from t=0 to t=t_final

    fidelity=rotation_fidelity(target_unitary,kraus_ops).real

    fidelity_grads=[]
    for map in maps:
        fidelity_grads.append(fidelity_map_deriv(map,target_unitary))

    # Sum over the second axis of map_grads to match fidelity_grads shape
    grads = np.sum(map_grads.sum(axis=1) * fidelity_grads, axis=(1,2))
   # grads=np.sum(map_grads*fidelity_grads)

    grad_norm = np.linalg.norm(grads, ord=np.inf)  # same criterion L-BFGS-B uses

    print(f"Infidelity = {1-fidelity:.6f}, ||grad|| = {grad_norm:.6f}")


    # Return the minus the gradient as infidelity is being minimized 
    return 1-fidelity.real,(-1.0*grads.reshape((-1)).real).tolist()

# %%
# Set upper and lower bounds on control parameters
x_bound = [-5*np.pi,5*np.pi]
z_bound = [-np.pi,np.pi]
num_params=2
bounds = np.zeros((num_steps*num_params,2))

for i in range(0, num_params*num_steps,num_params):
        bounds[i] = x_bound
        bounds[i+1] = z_bound

z0 = np.zeros(num_steps)
x0 = np.ones(num_steps)*omega_cutoff

# Flatten list for input to optimizer
parameter_list=[item for pair in zip(x0, z0) for item in pair]

optimization_result = minimize(
                        fun=infidandgrad,
                        x0=parameter_list,
                        method='L-BFGS-B',
                        jac=True,
                        bounds=bounds,
                        options = {'disp':True, 'gtol': 1e-05}

                        
)

print("The maximal fidelity was found to be : ",1-optimization_result.fun)

# %%
optimized_params = optimization_result.x
reshapedparas=np.array([i for i in (optimized_params.reshape((-1,num_params))).tolist() for j in range(2)])

plt.plot(reshapedparas[:,0],label='x')
plt.plot(reshapedparas[:,1],label='z')
plt.legend()
plt.show()

# %%
fig, axs = plt.subplots(nrows=4, ncols=1,figsize=(8,10)) 
fig.suptitle("Optimisation results")
fig.subplots_adjust(hspace=0.4)  # more vertical space between rows

field_labels = ["x","y","z"]
for i in range(0,num_params):
        axs[0].plot(t[:-1],optimization_result['x'][i::num_params],label=field_labels[i])
        axs[0].set_ylabel(r"$h_i$",rotation=0,fontsize=16)
        axs[0].set_xlabel("t")
        axs[0].legend()

basis_states=[op.SPIN_DM["x+"],op.SPIN_DM["y+"],op.SPIN_DM["z+"]] 
basis_labels=["|x+><x+|","|y+><y+|","|z+><z+|"]

for i,state in enumerate(basis_states):
        j=i+1
        # Input optimized controls into state_gradient to show dynamics of system under optimized fields
        optimized_dynamics = state_gradient(
                system=system,
                initial_state=np.array(state),
                target_derivative=target_derivative,
                process_tensors=[process_tensor_tebd],
                parameters=reshapedparas,
                num_steps=num_steps)

        dynamics = optimized_dynamics['dynamics']

        t, bloch_x =optimized_dynamics['dynamics'].expectations(op.sigma("x"))
        t, bloch_y = optimized_dynamics['dynamics'].expectations(op.sigma("y"))
        t, bloch_z = optimized_dynamics['dynamics'].expectations(op.sigma("z"))

        axs[j].plot(t,bloch_x,label='x')
        axs[j].plot(t,bloch_y,label='y')
        axs[j].plot(t,bloch_z,label='z')
        bloch_length = np.sqrt(bloch_x**2 +bloch_y**2 + bloch_z**2)
        axs[j].legend()
        axs[j].plot(t,bloch_length,label=r'$|\mathbf{\sigma}|$')
        axs[j].set_title(r"$\rho(0)={0}$".format(basis_labels[i]))
        axs[j].set_ylabel(r"$\langle \sigma \rangle$",rotation=0,fontsize=16)
        axs[j].set_xlabel("t")


# %%



