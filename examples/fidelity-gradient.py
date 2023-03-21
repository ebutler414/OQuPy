#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.insert(0,'.')
from IPython import embed

import numpy as np
import matplotlib.pyplot as plt

import oqupy
import oqupy.operators as op

def hamiltonian(x, y, z):
    h = np.zeros((2,2),dtype='complex128')
    for var, var_name in zip([x,y,z], ["x", "y", "z"]):
        h += var * op.sigma(var_name)
    return h

param_sys = oqupy.ParametrizedSystem(hamiltonian)


embed()
