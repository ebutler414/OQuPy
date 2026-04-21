# file for checking convergence of Q vs. protocol time plot
# optimal controls of particular dt,epsrel -> a coarse graining over finer dt and smaller epsrel
# the effect of removing the final jumps in controls is also tested

import argparse
from scipy.interpolate import interp1d
import numpy as np

import oqupy
from oqupy.system import TimeDependentSystem
import oqupy.operators as op
import dill
import os

parser = argparse.ArgumentParser()
parser.add_argument('--param', type=float, nargs=7,help='Parameter ID')
args = parser.parse_args()

dt_original = args.param[0]  # original dt
dt_finer= args.param[1] # finer dt
epsrel_original = args.param[2] # epsrel (int)
epsrel_smaller=args.param[3] # smaller epsrel
epsrel = 10**(-epsrel_smaller)  # convert to float
tcut=int(args.param[4])  # tcut (int)
tprot=int(args.param[5]) # protocol time

b_x=args.param[6] # symmetric parameter bounds for x,y,z

def interpolate_control(opt_control, t_total, dt_control,dt_finer):

    hx = opt_control

    t_L = np.arange(len(hx)) * dt_control

    # TEMPO simulation times
    t_T = np.arange(int(t_total/dt_finer)+1) * dt_finer

    h_t = interp1d(
        t_L,
        hx,
        kind="cubic",     
        bounds_error=False,
        fill_value=(hx[0], hx[-1])
    )
    return h_t

rho0=op.spin_dm("x+")

# finer process tensor for heats
with open(f'OQuPy/results/processtensors/processtensorCF_a=0.05_T=10_dt={dt_finer}_ps=-{epsrel_smaller}_tcut={tcut}', 'rb') as f:
    process_tensorcf=dill.load(f)

# finer process tensor for dynamics 
with open(f'OQuPy/results/processtensors/processtensor_a=0.05_T=10_dt={dt_finer}_ps=-{epsrel_smaller}_tcut={tcut}', 'rb') as f:
    process_tensor=dill.load(f)

# original control
with open(f'OQuPy/results/zero_control_lowercoupling_10K/conv_checks/dt={dt_original}_ps=-{epsrel_original}_tcut={tcut}_x={b_x}_y=0.0_z=0.0/{tprot}ps', 'rb') as f:
    opt_control_dict=dill.load(f)

opt_control=opt_control_dict["result"].x[0::3]
h_t=interpolate_control(opt_control,tprot,dt_original,dt_finer)

sys= TimeDependentSystem(lambda t : op.sigma("x")*h_t(t))

dynamicscf = oqupy.compute_dynamics(
process_tensor=process_tensorcf, 
system=sys,
initial_state=rho0,
start_time=0,
num_steps=int(tprot/dt_finer))

dynamics= oqupy.compute_dynamics(
process_tensor=process_tensorcf, 
system=sys,
initial_state=rho0,
start_time=0,
num_steps=int(tprot/dt_finer))

final_Q = dynamicscf.states.trace(axis1=1, axis2=2).imag / 0.01

folder="OQuPy/results/convergence_checks/a=0.05_T=10/"
os.makedirs(folder,exist_ok=True)

subfolder = os.path.join(folder, 'dt={0}_ps={1}_tcut={2}_x={3}'.format(dt_finer, np.round(np.log10(epsrel), 1), tcut,b_x))
os.makedirs(subfolder, exist_ok=True)

file_name1 = os.path.join(subfolder, '{0}ps'.format(tprot))
   
conv_dict={"heats":final_Q,
            "dynamics":dynamics.states
                   }

with open(file_name1,'wb') as f:
    dill.dump(conv_dict,f)