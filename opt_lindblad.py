
import oqupy.operators as opr
from scipy.linalg import expm,expm_frechet
import numpy as np 
import oqupy.operators as op
from scipy.optimize import minimize, Bounds, check_grad

# code to calculate the Markovian heat statistics and gradient
# using the same rates as in the non-Markovian case, 
# but with a time-discretized version of the tilted dissipator. 
# Used for a comparison with the optimal control results obtained using the non-Markovian heat stats and gradient.

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--param', type=float, nargs=5,help='Parameter ID')
args = parser.parse_args()

dt = args.param[0]  # dt
tprot=int(args.param[1]) # protocol time
b_x=args.param[2] # symmetric parameter bounds for x

temp=args.param[3]
alpha=args.param[4]

rho0 = op.spin_dm("x+") 
hs_dim = 2
omega_c=1 
sx = np.array([[0, 1], [1, 0]]) / 2 
sy = np.array([[0, -1j], [1j, 0]]) / 2 
sz = np.array([[1, 0], [0, -1]]) / 2
results = {}

u=0.01 # counting field parameter

rplus = -0.5 * (sz + 2 * sz @ sx) 
rminus = -0.5 * (sz - 2 * sz @ sx) 

def omega_t_discrete(hx):
    return -2*hx-1e-7

def make_rates(alpha, T):
    def J(w):
        return 2 * alpha * w * np.exp(-w / omega_c)

    def n(w):
        return 1.0 / (np.exp(w / T) - 1.0)

    def gammaA(hx):
        w = omega_t_discrete(hx)
        return 2 * np.pi * J(w) * n(w)

    def gammaE(hx):
        w = omega_t_discrete(hx)
        return 2 * np.pi * J(w) * (n(w) + 1)

    def DgammaA(hx):
        w = omega_t_discrete(hx)
        nw = n(w)
        Jw = J(w)
        dJ_dw = 2 * alpha * np.exp(-w / omega_c) * (1 - w / omega_c)
        dn_dw = -nw * (1 + nw) / T
        return 2 * np.pi * (dJ_dw * nw + Jw * dn_dw) * (-2.0)

    def DgammaE(hx):
        w = omega_t_discrete(hx)
        nw = n(w)
        Jw = J(w)
        dJ_dw = 2 * alpha * np.exp(-w / omega_c) * (1 - w / omega_c)
        dn_dw = -nw * (1 + nw) / T
        return 2 * np.pi * (dJ_dw * (nw + 1) + Jw * dn_dw) * (-2.0)

    return gammaA, gammaE, DgammaA, DgammaE

def tilted_dissipator(L_ops, rates, omegas, u, dim):
    D = np.zeros((dim*dim, dim*dim), dtype=complex)
    for L, gamma, omega in zip(L_ops, rates, omegas):
        Ld = L.conjugate().T
        D += gamma * np.exp(u*omega) * opr.left_right_super(L, Ld)
        D += -0.5 * gamma * opr.acommutator(Ld @ L)
    return D

def d_tilted_dissipator_dhx(L_ops, gammas, dgammas, omegas, u, dim):
    dD = np.zeros((dim*dim, dim*dim), dtype=complex)

    domegas = [-2.0, 2.0]

    for L, g, dg, w, dw in zip(L_ops, gammas, dgammas, omegas, domegas):
        Ld = L.conjugate().T
        dD += (dg + u*g*dw) * np.exp(u*w) * opr.left_right_super(L, Ld)
        dD += -0.5 * dg * opr.acommutator(Ld @ L)

    return dD

opt_log=[]
num_params=1

opt_runtimes=[
]

def heatandgrad_markovian(paras, num_steps, alpha, T,dt):
    gammaA, gammaE, DgammaA, DgammaE = make_rates(alpha, T)

    reshaped = np.repeat(paras.reshape(-1, num_params), 2, axis=0)

    rhos = [rho0.copy()]
    Es, dEs = [], []

    for k in range(num_steps):
        hx= reshaped[2*k]

        rates  = [gammaE(hx), gammaA(hx)]
        drates = [DgammaE(hx), DgammaA(hx)]
        omegas = [omega_t_discrete(hx), -omega_t_discrete(hx)]
        L_ops  = [rminus, rplus]


        D  = tilted_dissipator(L_ops, rates, omegas, u, hs_dim)
        dD = d_tilted_dissipator_dhx(L_ops, rates, drates, omegas, u, hs_dim)

        H=hx*sx*2
        L_vec=-1j*op.commutator(H)
        dL_vec=-1j*op.commutator(2*sx)

        U = expm(dt/2 * L_vec)
        
        dU= expm_frechet(dt/2 * L_vec, dt/2 * dL_vec, compute_expm=False)

        E_diss = expm(dt * D)
        dE_diss = expm_frechet(dt * D, dt * dD, compute_expm=False)

        E=U@E_diss@U

        dE = dU @ (E_diss @ U) + U @ (dE_diss @ U) + U @ (E_diss @ dU)

        Es.append(E)
        dEs.append(dE)

        rho_vec = E @ rhos[-1].reshape(-1)
        rhos.append(rho_vec.reshape(hs_dim, hs_dim))

    Zt = np.array([np.trace(rho) for rho in rhos])
    Qt = (np.log(Zt) - np.log(Zt[0])) / u
    heats = Qt

    opt_log.append(Qt[-1])

    lambdas = [None] * (num_steps + 1)
    lambdas[-1] = np.identity(hs_dim) / (u * Zt[-1])

    for k in reversed(range(num_steps)):
        lambdas[k] = (Es[k].conj().T @ lambdas[k+1].reshape(-1)).reshape(hs_dim, hs_dim)

    grads = np.zeros((num_steps, num_params))
    for k in range(num_steps):
        contrib = np.trace(
            lambdas[k+1].conj().T @
            (dEs[k] @ rhos[k].reshape(-1)).reshape(hs_dim, hs_dim)
        )
        grads[k, 0] = contrib.real

    return heats, grads.reshape(-1)

import time

def run_optimization(alpha, T,num_steps,dt,x_max):
    print(f"\nRunning optimisation for alpha={alpha}, T={T}")
    
    # reasonable guess for initial hamiltonian -> should be atleast kB T and around the peak of the spectral density
    hx0 =np.linspace(-omega_c/2-T/2, 0, num_steps)

    x0 = hx0

    bounds_list = []
    for _ in range(num_steps):
        bounds_list += [(-x_max, 0)]
    bounds = Bounds(*zip(*bounds_list))

    def f(z):
        Q, _ = heatandgrad_markovian(z, num_steps, alpha, T, dt)
        return Q[-1]

    def g(z):
        _, grad = heatandgrad_markovian(z, num_steps, alpha, T, dt)
        return np.asarray(grad, dtype=float)

    def obj(z):
        Q, g = heatandgrad_markovian(z, num_steps, alpha, T,dt)
        return Q[-1], np.asarray(g, dtype=float)

    start=time.time()
    res = minimize(
        fun=obj,
        x0=x0,
        jac=True,
        bounds=bounds,
        method="L-BFGS-B",
        options={"disp": True, "gtol":1e-5}
    )

    end = time.time()
    opt_runtimes.append(end-start)

    x_opt = res.x

    Q_opt, _ = heatandgrad_markovian(x_opt, num_steps, alpha, T,dt)

    return {
        "result": res,
        "x_opt": x_opt,
        "heats": Q_opt,
        "opttime":opt_runtimes,
        "optlog":opt_log
    }

num_steps=int(tprot/dt)
h_max=b_x

results_lindblad = run_optimization(alpha, temp,num_steps,dt,h_max)

import os, dill

folder="results/lindblad_control/"
os.makedirs(folder,exist_ok=True)

subfolder = os.path.join(folder, 'a={0},T={1}/dt={2}_x={3}'.format(alpha, int(temp/0.131),dt,b_x))
os.makedirs(subfolder, exist_ok=True)

file_name1 = os.path.join(subfolder, '{0}ps'.format(tprot))

with open(file_name1,'wb') as f:
    dill.dump(results_lindblad,f)