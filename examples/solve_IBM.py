import numpy as np
import matplotlib.pyplot as plt 

from scipy.integrate import quad, quad_explain

from scipy.constants import Boltzmann, hbar, electron_volt, pi

import time_evolving_mpo as tempo

omega_cutoff = 2.0e-3* electron_volt * 1e-12 / hbar
alpha = hbar * 11.2e-3 * omega_cutoff**2 * 1e12/ \
        (2 * pi * Boltzmann)

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True
'''
This code integrates my solution to the IBM for finding the Sy+ state

TODO: Make this code cleaner!

'''

# np.seterr(all='raise') # this makes sure any exceptions are thrown and dumps me into the except block


w0 = 0

k_B_new = Boltzmann * 10**(-12) / (hbar)

def get_the_SD(omega): # gets the spectral density
    correlations = tempo.PowerLawSD(alpha=alpha, 
                                zeta=3, 
                                cutoff=omega_cutoff, 
                                cutoff_type='gaussian',
                                temperature=0)
    return correlations.spectral_density(omega)

def coherent_integrand(w,beta,t):
    try:
        spectral_density_part = -get_the_SD(w)/(w**2)*(1-np.cos(w*t)) 
    except FloatingPointError:
        if w < 10**(-6):
            return - 0.0206855 * t**4*w**4 / beta - 0.00172379 * (-1.29973 * t**4 + beta**2 * t**4) * w**6 / beta

        spectral_density_part = 0

    try:
        bose_part = (2/(-1+np.exp(w*beta))+1)
    except FloatingPointError:
        bose_part = 1
        if w*beta < 16:
            print('Help me Jeebus') #this corrisponds to a bose function value of 10^(-7), should be safe enough
    return spectral_density_part * bose_part

coherent_integrand_vec = np.vectorize(coherent_integrand)


# (not sure if this is fully correct, check below line WRT to transfer report)
def coherent_sy(t,beta,w0,p1,p2): # solves for sy 
    np.seterr(over='raise') # this makes sure any exceptions are thrown and dumps me into the except block
    integral = quad(coherent_integrand,0,np.inf,args=(beta,t,))[0]
    np.seterr(over='warn')
    return np.exp(integral)/(2j) * (np.conj(p1)*p2 * np.exp(1j*w0*t) - p1*np.conj(p2)* np.exp(-1j*w0*t))

# this is def correct
def coherent_I_B(t,beta,w0,p1,p2):
    np.seterr(over='raise')
    integral = quad(coherent_integrand,0,np.inf,args=(beta,t,))[0]
    np.seterr(over='warn')
    return np.exp(integral)



def save_I_B(temp,show=False):
    tempo_x = np.linspace(0,5,100)
    coherent_I_B_vec = np.vectorize(coherent_I_B)
    tempo_ibm = coherent_I_B_vec(tempo_x,1/(k_B_new*temp),0,0,0)
    np.save('analytical_ibm_20_x',tempo_x)
    np.save('analytical_ibm_20_y',tempo_ibm)

    if show:
        fig8 = plt.figure()
        plt.plot(tempo_x,tempo_ibm)
        plt.title('this is what your saving')
        plt.show()




def plot_I_B_loads(beta_list):
    fig6 = plt.figure()
    x_t = np.linspace(0,5,100)
    for i in range(beta_list.size):


        y_result = coherent_I_B_vec(x_t,beta_list[i],1,1,1j) #computes the integral


        plt.plot(x_t,y_result,label=r'$T={}$ K'.format(temp_list[i])) # plot y values

    
    plt.legend(loc=1,prop={'size': 12})
    #plt.title(r'$I_B$ vs Temperature')
    plt.xlabel(r'Time ($ps$)')
    plt.ylabel(r'$I_B$')
    plt.tight_layout()
    plt.grid()
    plt.savefig('i_b_loads.pdf')



def plot_twinx_graph():

    fig5, ax51 = plt.subplots()

    colour = 'C1'

    ax51.plot(x_t_long,np.real(y_sy_coherent),colour,label=r'$<S_y(t)>$')
    #plt.plot(x_t,np.imag(y_sy_coherent),label='imag')
    ax51.set_ylabel(r'$<S_y(t)>$')
    ax51.set_xlabel('Time (ps)')
    ax51.set_ylim(-0.2,0.5)
    ax51.tick_params(axis='y', labelcolor = colour)

    ax52 = ax51.twinx()


    colour = 'C2'
    ax52.set_ylabel(r'$I_B$')
    ax52.plot(x_t_long,y_I_B,colour,label=r'$I_B$')
    ax52.set_ylim(-0.4,1)
    ax52.tick_params(axis='y', labelcolor = colour)



    plt.title(r'coherent IBM @ {}K $\omega_0=4\pi$'.format(temp_I_B))
    ax51.grid()
    fig5.legend()
    fig5.tight_layout()
