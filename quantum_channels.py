import numpy as np

from scipy.linalg import eig

from quantum_channel_transformations import transform_choi_to_kraus

d = 2

# start with a density matrix
density_matrix = np.arange(d**2).reshape(d,d)
# row vectorisation (default in C)
dm_vec = density_matrix.reshape(d**2)
# column vectorisation (default in Fortran)
dm_colvec = density_matrix.reshape(d**2,order='F')

# random superoperator
superoperator = np.arange(d**4).reshape(d,d,d,d)

# no vectorisation here,
dm_final = np.einsum('ijkl,kl->ij',superoperator,density_matrix)
print(dm_final)

# ok now do vectorisation:
# superoperator as in tempo (row vectorisation)
liouville_superoperator = superoperator.reshape(d**2,d**2)
# were tempo to be written in a column vectorisation language
liouville_superoperator_colvec = superoperator.reshape(d**2,d**2,order='F')


dm_final_liouville = np.dot(liouville_superoperator,dm_vec)
dm_final_colvec_liouville = np.dot(liouville_superoperator_colvec,dm_colvec)

print(dm_final_liouville)
print(dm_final_colvec_liouville)

density_matrix_recovered = dm_final_colvec_liouville.reshape(d,d,order='F')
print(density_matrix_recovered)
# success
# assuming superoperator is generated by a row vectorisation
choi_matrix = np.swapaxes(superoperator,0,2)
choi_matrix = np.swapaxes(choi_matrix,2,3)
choi_matrix = np.swapaxes(choi_matrix,1,3)

print('choi_matrix')
print(choi_matrix)

# obtain final density matrix by applying quantum channel in choi matrix form
# should be same density matrix as above
print(np.einsum('kilj,kl->ij',choi_matrix,density_matrix)) # correct
print(np.einsum('ijkl,ik->jl',choi_matrix,density_matrix))

# assuming superoperator is column vectorised
superoperator_colvec = np.swapaxes(superoperator,2,3)

choi_matrix_v2 = np.swapaxes(superoperator_colvec,0,3)
print(np.einsum('ijkl,ik->jl',choi_matrix_v2,density_matrix))

right_kraus,left_kraus = transform_choi_to_kraus(choi_matrix,
        return_left_operators=True)

# print(right_kraus)
# print(left_kraus)

rho_f = np.zeros((d,d),dtype='complex128')
for i in range(right_kraus.shape[0]):
    right = right_kraus[i,:,:]
    left = left_kraus[i,:,:]
    increment = right @ density_matrix @ left
    rho_f += increment

eigvals = eig(choi_matrix.reshape(d**2,d**2))[0]
print(rho_f)
print(eigvals)
