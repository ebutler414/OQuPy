import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from scipy.interpolate import interp1d
from oqupy.process_tensor import BaseProcessTensor
# function that returns the times that the density matrix is calculated at
from oqupy.helpers import get_full_timesteps

from oqupy.helpers import get_half_timesteps

import oqupy




alpha = 0.126
omega_cutoff = 3.04
temperature = 5 * 0.1309 # 1K = 0.1309/ps in natural units

dt = 0.05
dkmax = 60
esprel = 10**(-7)
max_time = 5


correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=3,
                                cutoff=omega_cutoff,
                                cutoff_type='gaussian',
                                temperature=temperature)
bath = oqupy.Bath(oqupy.operators.sigma("z")/2.0, correlations)

tempo_params = oqupy.TempoParameters(dt=dt, dkmax=dkmax, epsrel=esprel)

generate_process_tensor = False

if generate_process_tensor:
    process_tensor = oqupy.pt_tempo_compute(bath=bath,
                                            start_time=0,
                                            end_time=max_time,
                                            parameters=tempo_params,
                                            # progress_type='silent'
                                            )

    process_tensor.export('optimisation_pt.processTensor',overwrite=True)

else:
    process_tensor = oqupy.import_process_tensor(
                'optimisation_pt.processTensor','simple')


def get_hamiltonian(hx:np.ndarray,hz:np.ndarray,pt:BaseProcessTensor):
    """
    Returns a callable which takes a single parameter, t, and returns the
    hamiltonian of the two level system at that time. This function takes a
    process tensor, and the magnitude of $h_z$ and $h_x$ at each of those
    timesteps
    """

    # expval times including endtime, to generate the last "pixel"
    expval_times_p1 = get_full_timesteps(pt,0,inc_endtime=True)
    assert hx.size == expval_times_p1.size-1, \
        'hx must be same length as number of timesteps, without endtime'
    assert hz.size == expval_times_p1.size-1, \
        'hz must be same length as number of timesteps, without endtime'

    # duplicate last element so any time between t_f-dt and t_f falls within
    # this 'pixel' otherwise scipy interp1d doesn't like extrapolating so calls
    # it out of bounds
    hx_p1 = np.concatenate((hx,np.array([hx[-1]])))
    hz_p1 = np.concatenate((hz,np.array([hz[-1]])))

    hx_interp = interp1d(expval_times_p1,hx_p1,kind='zero')
    hz_interp = interp1d(expval_times_p1,hz_p1,kind='zero')

    def hamiltonian_t(t):
        _hx = hx_interp(t)
        _hz = hz_interp(t)

        hx_sx = 0.5 * oqupy.operators.sigma('x') * _hx
        hz_sz = 0.5 * oqupy.operators.sigma('z') * _hz
        hamiltonian = hz_sz + hx_sx
        return hamiltonian

    return hamiltonian_t

def dpropagator(hamiltonian,
    t, # expectation value times
    dt,
    op,
    h):
    """
    deriv of propagator wrt either a pre node or a post node
    """

    liouvillian_plus_h=-1j * oqupy.operators.commutator(hamiltonian(t)+h*op)
    liouvillian_minus_h=-1j * oqupy.operators.commutator(hamiltonian(t)-h*op)

    propagator_plus_h=expm(liouvillian_plus_h*dt/2.0).T
    propagator_minus_h=expm(liouvillian_minus_h*dt/2.0).T

    deriv=(propagator_plus_h-propagator_minus_h)/(2.0*h)
    return deriv

times = get_full_timesteps(process_tensor,start_time=0)

# pi pulse conjugate to s_x
h_x = np.ones(times.size) * np.pi / max_time
h_z = np.zeros(times.size)
hamiltonian_t = get_hamiltonian(hx=h_x,hz=h_z,pt=process_tensor)
system = oqupy.TimeDependentSystem(hamiltonian_t)


# all propagator half timesteps
dprop_dpram_times = get_half_timesteps(process_tensor,0)

# because my control parameters cover all timesteps need derivatives w.r.t.
# every half propagator.
dprop_dpram_derivs_x = []
for i in range(dprop_dpram_times.size):
    deriv = dpropagator(
                        hamiltonian_t,
                        dprop_dpram_times[i],
                        process_tensor.dt,
                        op=0.5*oqupy.operators.sigma('x'),
                        h = 10**(-6))
    dprop_dpram_derivs_x.append(deriv)

# print(dprop_dpram_derivs_x[0])
# import sys
# sys.exit()
gradient = oqupy.gradient(
                system=system,
                initial_state=oqupy.operators.spin_dm('z-'),
                target_state=oqupy.operators.spin_dm('z+'),
                process_tensor=process_tensor,
                # dprop_dparam_list=dprop_dpram_derivs_x
                )

def finite_difference(hx,hz,pt):
    def hamiltonian_fd(hx,hz,operator:np.ndarray):
        # expval times including endtime, to generate the last "pixel"
        expval_times_p1 = get_full_timesteps(pt,0,inc_endtime=True)
        assert hx.size == expval_times_p1.size-1, \
            'hx must be same length as number of timesteps, without endtime'
        assert hz.size == expval_times_p1.size-1, \
            'hz must be same length as number of timesteps, without endtime'

        # duplicate last element so any time between t_f-dt and t_f falls within
        # this 'pixel' otherwise scipy interp1d doesn't like extrapolating so calls
        # it out of bounds
        hx_p1 = np.concatenate((hx,np.array([hx[-1]])))
        hz_p1 = np.concatenate((hz,np.array([hz[-1]])))

        hx_interp = interp1d(expval_times_p1,hx_p1,kind='zero')
        hz_interp = interp1d(expval_times_p1,hz_p1,kind='zero')

        def hamiltonian_t(t):
            _hx = hx_interp(t)
            _hz = hz_interp(t)

            hx_sx = 0.5 * oqupy.operators.sigma('x') * _hx
            hz_sz = 0.5 * oqupy.operators.sigma('z') * _hz
            hamiltonian = hz_sz + hx_sx
            return hamiltonian

        return hamiltonian_t

deriv_list = gradient.deriv_list

print('deriv_list = ')
np.set_printoptions(suppress=True,precision=3)
print(deriv_list[1])