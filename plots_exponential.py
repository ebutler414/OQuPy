import sys
sys.path.insert(0,'..')
import os 
import pickle 
import dill

import oqupy
import oqupy.operators as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

#plt.style.use('physrev') 
plt.rcParams['figure.dpi'] = "300"

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize,Bounds


# ----------------- Parameters --------------

# --- Bath ----

omega_cutoff = 5.0*2*np.pi
wq0 = 5.0*2*np.pi

alpha = 0.03
temperature = 0.0 
t_max = 11.0
dt = 1./omega_cutoff/np.sqrt(3)/2
num_steps = int(t_max/dt)
t_list = np.linspace(0,t_max,num_steps)
dt_half = dt/2 # Accounts for the fact that we use half-time propagators.
num_steps_half = int(t_max/dt_half)
t_list_half = np.linspace(0,t_max,num_steps_half)

N=6000
c = 3*omega_cutoff/N

def omega_k(k):
    return c*np.abs(k)
def g_k(k):
    return np.sqrt(2*alpha*c*omega_k(k))*np.sqrt(np.exp(-omega_k(k)/omega_cutoff))

wk_vec = np.array([omega_k(k) for k in range(0,N)])
gk_vec = np.array([g_k(k) for k in range(0,N)])

# TDVP functions

def fw(f_vec,wq):
    return np.exp(1j*dt_half*(wq+wk_vec))*f_vec+gk_vec/2/(wq+wk_vec)*(np.exp(1j*dt_half*(wq+wk_vec))-1)

def cost(wq_vec):
    
    f_t_vec= np.zeros([N,num_steps_half],dtype=complex)
    f_t_vec[:,0] = -0.5*gk_vec/(wq_vec[0]+wk_vec)
    
    for t in np.arange(0,num_steps_half-1):
        
        f_t_vec[:,t+1] = fw(f_t_vec[:,t],wq_vec[t])

    return -np.exp(-2*np.linalg.norm(f_t_vec[:,-1])**2) 

def cost_vec(wq_vec):

    f_t_vec= np.zeros([N,num_steps_half],dtype=complex)
    f_t_vec[:,0] = -0.5*gk_vec/(wq_vec[0]+wk_vec)
    
    for t in np.arange(0,num_steps_half-1):
        
        f_t_vec[:,t+1] = fw(f_t_vec[:,t],wq_vec[t])

    return [-np.exp(-2*np.linalg.norm(f_t_vec[:,t])**2) for t in np.arange(0,num_steps_half)]

def tdvp_sol(wq_vec):

    f_t_vec= np.zeros([N,len(wq_vec)],dtype=complex)
    f_t_vec[:,0] = -0.5*gk_vec/(wq_vec[0]+wk_vec)
    
    for t in np.arange(0,len(wq_vec)-1):
        
        f_t_vec[:,t+1] = fw(f_t_vec[:,t],wq_vec[t])

    return f_t_vec


# TDVP calculation


name = 'alpha{}tmax{}wc{}wq0{}exponential'.format(alpha,round(t_max,3),round(omega_cutoff/2/np.pi,2),round(wq0/2/np.pi,2))
name_replaced = name.replace('.','-')
name_path = os.path.dirname(__file__)+'/opt/'+name_replaced  


opt_file = open(name_path, 'rb')    
dict_run = pickle.load(opt_file)
opt_file.close()

x0 = dict_run['optimization_result'].x

resTDVP = cost_vec(2*x0+0.0001) # factor of 2 due to the fact that we removed the 1/2 from the Hamiltonian.

# Oqupy calculation

from oqupy.iTEBD_TEMPO_useoqupybath import iTEBD_TEMPO_oqupy
from oqupy.process_tensor import TTInvariantProcessTensor
from oqupy.tti_tempo import TTITempo

name = 'alpha{}wc{}wq0{}exponential'.format(alpha,round(omega_cutoff/2/np.pi,2),round(wq0/2/np.pi,2))
name_replaced = name.replace('.','-')
name_path = os.path.dirname(__file__)+'/pt/'+name_replaced+".processTensor"       
pt_file = open(name_path,'rb')
process_tensor_tebd = dill.load(pt_file)
pt_file.close()


def discrete_hamiltonian(hx):
        return hx*op.sigma('x')
system = oqupy.ParameterizedSystem(discrete_hamiltonian)

h_x_opt = np.expand_dims(dict_run['optimization_result'].x,1)

initial_state = op.spin_dm('mixed')
target_state = op.spin_dm('x-')
target_derivative = target_state.T

grad_res_opt = oqupy.state_gradient(
    system=system,
    initial_state=initial_state,
    target_derivative=target_derivative,
    process_tensors=[process_tensor_tebd],
    num_steps=num_steps,
    parameters=h_x_opt)

dynamics_opt = grad_res_opt['dynamics']
t, s_x_opt = dynamics_opt.expectations(op.sigma('x'), real=True)


# --------------- Plots  ------------------

import matplotlib.gridspec as gridspec

# Magnetization and optimal protocol, with TDVP comparison, and protocol PSD

a=0.2
b=0.03
w = 0.7
h = 0.25

fig, axs = plt.subplots(3, figsize=(3.0,4))
axs[0].plot(t_list_half, np.log10((1+np.array(resTDVP))/2), c = 'tab:gray',label='TDVP', linestyle = 'solid')
axs[0].plot(t,np.log10((1+s_x_opt)/2),'k', label = 'TEMPO',linestyle = 'solid')
axs[0].get_xaxis().set_visible(False)
axs[0].set_ylabel(r'$\log(P_+)$')
axs[0].set_yticks([0,-1,-2,-3])
axs[0].set_ylim([-3.2,0.2])

axs[1].plot(t_list_half[0:num_steps_half],h_x_opt/2/np.pi*2)
axs[1].set_xlabel(r'$t/ns$')
axs[1].set_ylabel(r'$\omega_q/2\pi GHz$')

axs[0].set_position([a,1-b-h,w,h])
axs[1].set_position([a,1-b-2*h,w,h])

data = x0 - np.mean(x0)
ps = np.abs(np.fft.fft(data))**2
time_step = dt_half
freqs = np.fft.fftfreq(data.size, time_step)
idx = np.argsort(freqs)


axs[2].plot(freqs[idx], ps[idx]/100000,label = r'$\omega_q^0/2\pi = 5 GHz$')


name = 'alpha{}tmax{}wc{}wq0{}exponential'.format(alpha,round(t_max,3),round(omega_cutoff/2/np.pi,2),round(6.0,2))
name_replaced = name.replace('.','-')
name_path = os.path.dirname(__file__)+'/opt/'+name_replaced  
opt_file = open(name_path, 'rb')    
dict_run = pickle.load(opt_file)
opt_file.close()

x1 = dict_run['optimization_result'].x

data = x1 - np.mean(x1)
ps = np.abs(np.fft.fft(data))**2
time_step = dt_half
freqs = np.fft.fftfreq(data.size, time_step)
idx = np.argsort(freqs)


axs[2].plot(freqs[idx], ps[idx]/100000,label = r'$\omega_q^0/2\pi = 6  GHz$')
axs[2].set_xlim([0,10])
axs[2].set_ylabel(r'PSD/$10^5$')
axs[2].set_xlabel(r'$\omega/2\pi$')

for ax in axs:
    ax.legend(fontsize='5')

axs[0].set_position([a,1-b-h,w,h])
axs[1].set_position([a,1-b-2*h,w,h])
axs[2].set_position([a,1-b-3*h-0.13,w,h])

for ax, label in zip(axs,['(a)','(b)','(c)']):
    ax.text(
    0.2, 0.7, label, transform=(
        ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
    fontsize='small', va='bottom', fontfamily='serif')
# plt.tight_layout(h_pad = 0.0)
plt.savefig("exponential1.pdf",bbox_inches='tight')

plt.show()

small =10
plt.rc('font', size = small )
plt.rc('axes', titlesize = small )
plt.rc('axes', labelsize = small )
plt.rc('xtick', labelsize = small )
plt.rc('ytick', labelsize = small )
plt.rc('legend', fontsize = 7 )

wq_prot = 2*x0+0*np.pi
x = t_list_half
y = wk_vec/2/np.pi
X, Y = np.meshgrid(x,y)
f_0_vec = -0.5*gk_vec/(wq_prot[0]+wk_vec)
f_sol = tdvp_sol(wq_prot+0.0001)
y_sol = np.array([(f_sol[:,j]-f_0_vec) for j in range(0,num_steps_half)]).T

fig = plt.figure(figsize=(8,3))
ax1 = plt.subplot(222)
ax1 = plt.gca()
pc1 = ax1.pcolormesh(X, Y, np.angle(y_sol)/np.pi, rasterized = True)
# plt.title(r'$arg(q_k)/\pi $')
plt.xlabel(r'$t/ns$')
plt.ylabel(r'$\omega_k/2\pi GHz $')
plt.ylim([0,5])
cbar = fig.colorbar(pc1, ticks=[-0.99,0,0.99])
cbar.ax.set_yticklabels(['-1','0','1'])
cbar.set_label(r'$arg(q_k)/\pi $', rotation=270, labelpad=8)


k= int(N/6)
ax2 = plt.subplot(121)
ax2.plot(np.real(f_sol[k])*10**3,np.imag(f_sol[k])*10**4,'k', rasterized = True)
ax2.plot(np.real(f_sol[k][1700:1750]*10**3),np.imag(f_sol[k][1700:1750]*10**4),'r-')
ax2 = plt.gca()
ax2.set_aspect('equal', adjustable='box')
ax2.axhline(y=0, color='k',linewidth = 0.5)
ax2.axvline(x=0, color='k',linewidth = 0.5)
ax2.set_aspect('auto')
ax2.set_xlabel(r'$Re{f_k} \times 10^3$')
ax2.set_ylabel(r'$Im{f_k} \times 10^4$')
ax2.legend()



ax3 = plt.subplot(224)

wq_prot = 2*x0
f_0_vec = -0.5*gk_vec/(wq_prot[0]+wk_vec)
f_sol = tdvp_sol(wq_prot+0.0001)
y_sol = np.array([(f_sol[0:,j]-f_0_vec) for j in range(0,num_steps_half)]).T # plot up to wc . 
angles = np.angle(y_sol)[0:int(N/3),1153*2]/np.pi # 1153 corresponds to second to last minima. 
angles[np.abs(np.diff(angles, append=0)) > 0.5] = np.nan


ax3.plot(wk_vec[0:int(N/3)]/2/np.pi, angles, label = r'$\omega_q^0 = 5GHz$')


wq_prot = 2*x0-3*2*np.pi
f_0_vec = -0.5*gk_vec/(wq_prot[0]+wk_vec)
f_sol = tdvp_sol(wq_prot+0.0001)
y_sol = np.array([(f_sol[:,j]-f_0_vec) for j in range(0,num_steps_half)]).T
angles =np.angle(y_sol)[0:int(N/3),1153*2]/np.pi
angles[np.abs(np.diff(angles, append=0)) > 0.5] = np.nan


ax3.plot(wk_vec[0:int(N/3)]/2/np.pi, angles, label = r'$\omega_q^0 = 2GHz$')
ax3.set_xlabel(r'$\omega_k/2\pi GHz$')
ax3.set_ylabel(r'$arg(q_k(t_f))/\pi$')
ax3.set_ylim([-1,1])
ax3.legend(loc = 'upper right', borderpad = 0.0)


for ax, label in zip([ax1, ax2, ax3],['(b)','(a)','(c)']):
    ax.text(
    0.0, 1.0, label, transform=(
        ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
    fontsize='medium', va='bottom', fontfamily='serif')

plt.tight_layout(h_pad = -1.0, w_pad = 2.5)

plt.savefig("oscillator_exp.pdf",bbox_inches='tight', dpi = 600)
plt.show()





# # single oscillator


# k= 500

# plt.figure()
# plt.plot(np.real(f_sol[k]),np.imag(f_sol[k]),'k')
# plt.plot(np.real(f_sol[k][8000:8300]),np.imag(f_sol[k][8000:8300]),'r')
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# ax.axhline(y=0, color='k',linewidth = 0.5)
# ax.axvline(x=0, color='k',linewidth = 0.5)
# # plt.plot(np.real(f_sol[500,:]),np.imag(f_sol[500,:]))
# plt.xlabel(r'$\Re{q}$')
# plt.ylabel(r'$\Im{q}$')
# plt.legend()

# # phase plot
# plt.figure()
# wq_prot = 2*x0+0*np.pi
# x = t_list_half
# y = wk_vec/2/np.pi
# X, Y = np.meshgrid(x,y)
# f_0_vec = -0.5*gk_vec/(wq_prot[0]+wk_vec)
# f_sol = tdvp_sol(wq_prot+0.0001)
# y_sol = np.array([(f_sol[:,j]-f_0_vec) for j in range(0,num_steps_half)]).T

# plt.figure(figsize=(2,2))
# fig = plt.gcf()
# ax = plt.gca()
# pc1 = ax.pcolormesh(X, Y, np.angle(y_sol))
# plt.title(r'$arg(q_k)/\pi $')
# plt.xlabel(r'$t/ns$')
# plt.ylabel(r'$\omega_k/2\pi GHz $')
# plt.ylim([0,5])
# fig.colorbar(pc1)






# # final phase of oscillators

# plt.figure()

# wq_prot = 2*x0
# f_0_vec = -0.5*gk_vec/(wq_prot[0]+wk_vec)
# f_sol = tdvp_sol(wq_prot+0.0001)
# y_sol = np.array([(f_sol[0:,j]-f_0_vec) for j in range(0,num_steps_half)]).T # plot up to wc . 
# angles = np.angle(y_sol)[0:int(N/3),1153*2]/np.pi # 1153 corresponds to second to last minima. 
# angles[np.abs(np.diff(angles, append=0)) > 0.5] = np.nan


# plt.plot(wk_vec[0:int(N/3)]/2/np.pi, angles, label = r'$\omega_q^0 = 5GHz$')


# wq_prot = 2*x0-3*2*np.pi
# f_0_vec = -0.5*gk_vec/(wq_prot[0]+wk_vec)
# f_sol = tdvp_sol(wq_prot+0.0001)
# y_sol = np.array([(f_sol[:,j]-f_0_vec) for j in range(0,num_steps_half)]).T
# angles =np.angle(y_sol)[0:int(N/3),1153*2]/np.pi
# angles[np.abs(np.diff(angles, append=0)) > 0.5] = np.nan


# plt.plot(wk_vec[0:int(N/3)]/2/np.pi, angles, label = r'$\omega_q^0 = 2GHz$')



# plt.xlim([0,10])
# plt.ylim([-1,1])
# plt.xlabel(r'$\omega_k/GHz$')
# plt.ylabel(r'$arg(q_k)/\pi $')
# plt.legend()
# plt.tight_layout()
# # plt.savefig("phase_exponential.pdf",bbox_inches='tight')
# # plt.show()

# # convergence plot
# plt.figure(1)
# plt.plot(np.arange(0,np.size(dict_run['optimization_convergence'])),np.log10(dict_run['optimization_convergence'])   )
# plt.show()