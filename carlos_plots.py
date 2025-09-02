#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
#plt.rcParams['figure.dpi'] = "75"

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize,Bounds
from endpointcorrections import trapezoidal_fft_integral


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

#name = 'alpha{}tmax{}wc{}wq0{}exponential'.format(alpha,round(t_max,3),round(omega_cutoff/2/np.pi,2),round(wq0/2/np.pi,2))
#name_replaced = name.replace('.','-')
#name_path = os.path.dirname(__file__)+'/opt/'+name_replaced  
#name_path = os.getcwd()+'/opt/'+name_replaced  


#opt_file = open(name_path, 'rb')    
#dict_run = pickle.load(opt_file)
#opt_file.close()

#x0 = dict_run['optimization_result'].x


# Oqupy calculation

from oqupy.iTEBD_TEMPO_useoqupybath import iTEBD_TEMPO_oqupy
from oqupy.process_tensor import TTInvariantProcessTensor
from oqupy.tti_tempo import TTITempo

name = 'alpha{}wc{}wq0{}exponential'.format(alpha,round(omega_cutoff/2/np.pi,2),round(wq0/2/np.pi,2))
name_replaced = name.replace('.','-')
name_path = os.path.dirname(__file__)+'/pt/'+name_replaced+".processTensor"       
name_path = os.getcwd() + '/pt/'+name_replaced+".processTensor"
pt_file = open(name_path,'rb')
process_tensor_tebd = dill.load(pt_file)
pt_file.close()


#h_x_opt = np.expand_dims(dict_run['optimization_result'].x,1)

#%%


pt=process_tensor_tebd
hx=omega_cutoff/2
system=oqupy.System(hx*op.sigma('x'))
pt.set_length(2200) 
spin_down = oqupy.operators.spin_dm("down")
s_z = 0.5*oqupy.operators.sigma("z")
s_x = 0.5*oqupy.operators.sigma("x")
corr = oqupy.PowerLawSD(alpha, 1, omega_cutoff, temperature = 0.0)
bath = oqupy.Bath(s_z, corr)
w = omega_cutoff
delta = 0.1 * omega_cutoff

initial_state = op.spin_dm('mixed')
dynamics=oqupy.compute_dynamics(system=system,initial_state=initial_state,process_tensor=pt,num_steps=2200)
times,sx=dynamics.expectations(s_x)

#%%

plt.figure()
plt.plot(times,sx.real)
plt.xlabel('Time')
plt.ylabel('sigma_x')

# In[4]:


if os.path.isfile('bath_corr.pkl'):
    print('loading correlation from file')
    with open('bath_corr.pkl', 'rb') as f:
        bath_corr=dill.load(f)
else:
    print('generating correlations and saving to file')
    bath_corr = oqupy.bath_dynamics.TwoTimeBathCorrelations(system, bath, pt, initial_state)
    tlist, occ = bath_corr.occupation(w, delta, change_only = True)
    with open('bath_corr.pkl', 'wb') as f:
        dill.dump(bath_corr,f) 
    energy = w * occ


#plt.plot(tlist[1:],energy)
#plt.show()




cfarr=np.array(bath_corr._system_correlations)
cfarr[np.isnan(cfarr)] = 0
plt.clf()
tsforplot=108
tforplot=tsforplot*dt
tprimes=dt*np.arange(0,tsforplot+1)
plt.plot(tprimes,cfarr[:(tsforplot+1),tsforplot].real)
plt.plot(tprimes,cfarr[:(tsforplot+1),tsforplot].imag)
plt.show()

#%%
def plotdispl(time):
    tsforplot=int(time//dt)
    tindx=tsforplot
    n=np.shape(cfarr[:,tindx])[0]#6000
    pstep=1
    tdat=cfarr[:,tindx]
    allomega,ftcfarr,ftcarrdumb=trapezoidal_fft_integral(tdat, 0.0, dt, n)
    
    omega=allomega[0:n//2]
    disps=ftcfarr[0:n//2]
    dispsdumb=ftcfarr[0:n//2]
    
    # multiply by the spectral density.
    
    disps=disps*corr.spectral_density(omega)
    
    disps=-2.0j*np.exp(-1.0j*tindx*dt)*disps
    
    
    dispsdumb=dispsdumb*corr.spectral_density(omega)
    
    dispsdumb=-2.0j*np.exp(-1.0j*tindx*dt)*dispsdumb
    
    
    # compare with displacements in the polaron state
    wq=2*hx
    plt.plot(omega,corr.spectral_density(omega)/(2*(wq+omega)),label='Polaron Ansatz')
    plt.plot(omega[::pstep],np.abs(disps)[::pstep],label='OQuPy')
    plt.plot(omega[::pstep],np.abs(dispsdumb)[::pstep],label='OQuPy-Simple FFT')
    plt.xlim(right=150)
    plt.xlabel(r'$\omega$ (ns$^{-1}$)')
    plt.ylabel(r'$|f(\omega)|^2$ (ns)')
    plt.text(0.8,0.5,r'$\alpha$=0.03',transform=plt.gca().transAxes)
    t=tindx*dt
    plt.text(0.8,0.4,rf't={t:.1f}',transform=plt.gca().transAxes)
    plt.legend()
    print(np.max(np.abs(disps-dispsdumb)))

#%%
# refactored to do the ft slightly differently
def plotdispl2(time,natten):
    tsforplot=int(time//dt)
    tindx=tsforplot
    n=np.shape(cfarr[:,tindx])[0]#6000
    pstep=1
    tdat=cfarr[:(tindx+1),tindx]
    #natten=tindx/natten
    attenfactor=np.exp(np.arange(-tindx,1)/natten)
    tdat=tdat*attenfactor
    allomega,ftcfarr,ftcarrdumb=trapezoidal_fft_integral(tdat, 0, dt, n)
    
    omega=allomega[0:n//2]
    disps=ftcfarr[0:n//2]
    dispsdumb=ftcarrdumb[0:n//2]
    
    # multiply by the spectral density.
    
    disps=-2.0j*disps*corr.spectral_density(omega)

    dispsdumb=dispsdumb*corr.spectral_density(omega)
    
    dispsdumb=-2.0j*np.exp(-1.0j*tindx*dt)*dispsdumb
    
    # compare with displacements in the polaron state
    wq=2*hx
    plt.plot(omega,corr.spectral_density(omega)/(2*(wq+omega)),label='Polaron Ansatz')
    plt.plot(omega[::pstep],np.abs(disps)[::pstep],label='OQuPy')
    plt.plot(omega[::pstep],np.abs(dispsdumb)[::pstep],label='OQuPy-Simple FFT')
    plt.xlim(right=150)
    plt.xlabel(r'$\omega$ (ns$^{-1}$)')
    plt.ylabel(r'$|f(\omega)|^2$ (ns)')
    plt.text(0.8,0.5,r'$\alpha$=0.03',transform=plt.gca().transAxes)
    t=tindx*dt
    plt.text(0.8,0.4,rf't={t:.1f}',transform=plt.gca().transAxes)
    plt.legend()
    print(np.max(np.abs(disps-dispsdumb)))


fig,ax=plt.subplots(nrows=3,ncols=1)

plt.sca(ax[0])
plotdispl2(2.0,200)
plt.sca(ax[1])
plotdispl2(5.0,200)
plt.sca(ax[2])
plotdispl2(10.0,200)
plt.show()




# %%
