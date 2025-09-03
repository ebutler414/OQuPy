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

plt.style.use('physrev.mplstyle')

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
pt=process_tensor_tebd
hx=omega_cutoff/2
system=oqupy.System(hx*op.sigma('x'))
pt.set_length(1100) 
s_z = 0.5*oqupy.operators.sigma("z")
s_x = 0.5*oqupy.operators.sigma("x")
corr = oqupy.PowerLawSD(alpha, 1, omega_cutoff, temperature = 0.0)
bath = oqupy.Bath(s_z, corr)
w = omega_cutoff
delta = 0.1 * omega_cutoff
initial_state = op.spin_dm('mixed')
corrfile='bath_corr_mixedic.pkl'

#%%
#dynamics=oqupy.compute_dynamics(system=system,initial_state=initial_state,process_tensor=pt,num_steps=2200)
#times,sx=dynamics.expectations(s_x)
#plt.figure()
#plt.plot(times,sx.real)
#plt.xlabel('Time')
#plt.ylabel('sigma_x')
#plt.show()

if os.path.isfile(corrfile):
    print('loading correlation from file')
    with open(corrfile, 'rb') as f:
        bath_corr=dill.load(f)
else:
    print('generating correlations and saving to file')
    bath_corr = oqupy.bath_dynamics.TwoTimeBathCorrelations(system, bath, pt, initial_state)
    tlist, occ = bath_corr.occupation(w, delta, change_only = True)
    with open(corrfile, 'wb') as f:
        dill.dump(bath_corr,f) 
    energy = w * occ
cfarr=np.array(bath_corr._system_correlations)
cfarr[np.isnan(cfarr)] = 0

#%%
# function to do ft and return frequencies and displacements 
def displacementdensity(time,natten):
    tsforplot=int(time//dt)
    tindx=tsforplot
    n=np.shape(cfarr[:,tindx])[0]#6000 
    tdat=cfarr[:(tindx+1),tindx]
    attenfactor=np.exp(np.arange(-tindx,1)/natten)
    tdat=tdat*attenfactor
    allomega,ftcfarr,ftcarrdumb=trapezoidal_fft_integral(tdat, 0, dt, n)
    omega=allomega[0:n//2]
    disps=ftcfarr[0:n//2]   
    # multiply by the spectral density.    
    disps=-2.0j*disps*corr.spectral_density(omega)
    disps=disps*np.exp(-1.0j*omega*tindx*dt)
    return omega,disps

omega,disps=displacementdensity(10.0,54)
wq=2*hx
every=10
plt.figure()
plt.plot(omega,1000*corr.spectral_density(omega)/(2*(wq+omega)),label='Polaron Ansatz')
plt.plot(omega,1000*np.abs(disps))
plt.plot(omega[::every],1000*np.abs(disps)[::every],'o',markersize=3)
plt.xlim(0,100)
plt.xlabel(r'$\omega$ (ns$^{-1}$)')
plt.ylabel(r'$|gf(\omega)|\times 10^3$ (ns)')
plt.show()





# %%
