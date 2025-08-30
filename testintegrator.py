#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 07:36:23 2025

@author: paul
"""

import matplotlib.pyplot as plt
import numpy as np
from endpointcorrections import trapezoidal_fft_integral

numt=101
dt=0.01
times=np.arange(numt)*dt
tdat=np.cos(12.0*times)

omegas,fftres,dumb=trapezoidal_fft_integral(tdat,0,dt,1001)    

# analytic result for integral from 0 to b, with y=12 the prefactor in the cosine.
def analyticsft(omega, b, y):
    numerator = -1j * omega + np.exp(1j * b * omega) * (1j * omega * np.cos(b * y) + y * np.sin(b * y))
    denominator = (y - omega) * (y + omega)
    return numerator / denominator

omegas=np.fft.fftshift(omegas)
fftres=np.fft.fftshift(fftres)
dumb=np.fft.fftshift(dumb)

analytic=np.vectorize(lambda x: analyticsft(x,1.0,12.0))(omegas)

# plot them, along with the naive fourier transform result.
plt.figure()
plt.plot(omegas[::2],np.abs(fftres)[::2],'x', label='Integrator')
plt.plot(omegas,np.abs(dumb),label='Simple FFT')
plt.plot(omegas,np.abs(analytic),label='Analytic')
plt.xlabel(r'$\omega$')
plt.legend()
plt.minorticks_on()
plt.grid(which='both')

plt.figure()
plt.plot(omegas,np.abs(analytic-fftres),label='Error')
plt.xlabel(r'$\omega$')
plt.ylabel('Error')

