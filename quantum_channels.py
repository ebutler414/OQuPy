import numpy as np

d = 2

density_matrix = np.arange(d**2).reshape(d,d)
# row vectorisation (default in C)
dm_vec = density_matrix.reshape(d**2)
# column vectorisation (default in Fortran)
dm_colvec = density_matrix.reshape(d**2,order='F')

superoperator = np.arange(d**4).reshape(d,d,d,d)
print(superoperator)
superoperator_colvec = np.swapaxes(superoperator,2,3)
print(superoperator_colvec)

# superoperator as in tempo
liouville_superoperator = superoperator.reshape(d**2,d**2)
# were tempo to be written in a column vectorisation language
liouville_superoperator_colvec = superoperator.reshape(d**2,d**2,order='F')

# TODO: how is superoperator_colvec related to superoperator_colvec?
# are they the same?

dm_final = np.einsum('ijkl,kl->ij',superoperator,density_matrix)

dm_final_Ftype = np.einsum('ijkl,lk->ij',superoperator_colvec,density_matrix)

print(dm_final)
print(dm_final_Ftype)

dm_final_liouville = np.dot(liouville_superoperator,dm_vec)
dm_final_colvec_liouville = np.dot(liouville_superoperator_colvec,dm_colvec)

print(dm_final_liouville)
print(dm_final_colvec_liouville)

density_matrix_recovered = dm_final_colvec_liouville.reshape(d,d,order='F')
# success
print(density_matrix_recovered)

choi_matrix_intermediate = superoperator.swapaxes(0,2)
choi_matrix_intermediate_2 = choi_matrix_intermediate.swapaxes(2,3)
choi_matrix = choi_matrix_intermediate_2.swapaxes(1,3)

print(np.einsum('ijkl,kl->ij',choi_matrix,density_matrix))