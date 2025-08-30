#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 14:33:47 2025

@author: paul

An attempt to implement the endpoint corrections, as in Numerical Recipes, for a Fourier integral with lower limit 0
and upper limit corresponding to the highest time in the data provided.

"""


import numpy as np

def w(theta):
    if np.abs(theta)<0.1:
        return 1.0-theta**2/12+theta**4/360-theta**6/20160
    else:
        return 2*(1-np.cos(theta))/(theta**2)
    
def alpha0(theta):
    if np.abs(theta)<0.1:
        return -0.5+theta**2/24-theta**4/720+theta**6/40320 + \
            1.0j*theta*(1.0/6.0-theta**2/120+theta**4/5040-theta**6/362880)
    else:
        return -(1-np.cos(theta))/(theta**2)+1.0j*(theta-np.sin(theta))/theta**2

def trapezoidal_fft_integral(samples,a,delta,n):
    r'''
    Numerically computes the Fourier integral $\int_a^b e^{i\omega t} f(t)$, using the FFT with endpoint corrections.
    Function f(t) is given as the array samples, where samples[k] is the value of the function at the time $a+k*\delta$.
    $a$ is the lower limit of integral
    $\Delta$ is the time spacing.
    n is the number of points at which to compute the Fourier transform, which is adjusted to an even number.
    It should be chosen so as to oversample the function f(t) by a sufficient factor. 
    Note that the upper limit $b=a+m\Delta$, where m=samples.shape[0]-1, i.e. the maximum time in the array.
    Returns omegas,dftres, where omegas are the angular frequencies at which the result is computed, and dftres the result.
    These are in the scrambled FFT order and can be unscrambled into the usual order with np.fft.fftshift.
    '''
    m=samples.shape[0]-1 # largest index of the data
    assert (n>=(m+1))
    if (n%2==1):
        n=n+1
    print(n)
    thetas=np.fft.fftfreq(n)*2*np.pi
    omegas=thetas/delta    
    ws=np.vectorize(w)(thetas)
    a0s=np.vectorize(alpha0)(thetas)
    dftres=np.fft.ifft(samples,n=n,norm="forward")
    #dumbway=dftres
    dftres=delta*np.exp(1.0j*omegas*a)*(dftres*ws+a0s*samples[0]+
                                        +np.conjugate(a0s)*samples[m]*np.exp(1.0j*omegas*(m*delta-a)))
    return omegas,dftres#,delta*dumbway

