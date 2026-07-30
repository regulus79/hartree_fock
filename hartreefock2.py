from integral_matricies import *
import math
import numpy as np

sto_3g_exponents_H_1s = np.array([
	0.3425250914E+01,
	0.6239137298E+00,
	0.1688554040E+00
]) / bohr_radius**2
sto_3g_coefficients_H_1s = np.array([
	0.1543289673E+00,
	0.5353281423E+00,
	0.4446345422E+00
])
sto_3g_angular_momenta_H_1s = [(0,0,0), (0,0,0), (0,0,0)]


sto_3g_exponents_O_1s = np.array([
	0.1307093214E+03,
	0.2380886605E+02,
	0.6443608313E+01,
]) / bohr_radius**2
sto_3g_exponents_O_2sp = np.array([
	0.5033151319E+01,
	0.1169596125E+01,
	0.3803889600E+00,
]) / bohr_radius**2

sto_3g_coefficients_O_1s = np.array([
	0.1543289673E+00,
	0.5353281423E+00,
	0.4446345422E+00,
])
sto_3g_coefficients_O_2s = np.array([
	-0.9996722919E-01,
	0.3995128261E+00,
	0.7001154689E+00,
])
sto_3g_coefficients_O_2p = np.array([
	0.1559162750E+00,
	0.6076837186E+00,
	0.3919573931E+00,
])

sto_3g_angular_momenta_O_1s = [(0,0,0), (0,0,0), (0,0,0)]
sto_3g_angular_momenta_O_2s = [(0,0,0), (0,0,0), (0,0,0)]
sto_3g_angular_momenta_O_2px = [(1,0,0), (1,0,0), (1,0,0)]
sto_3g_angular_momenta_O_2py = [(0,1,0), (0,1,0), (0,1,0)]
sto_3g_angular_momenta_O_2pz = [(0,0,1), (0,0,1), (0,0,1)]


offset = 0.7414 * 1e-10

nuclei = [
	Nucleus((0,0,0), charge_e*8),
	Nucleus((-offset*math.sqrt(2)/2,0,offset*math.sqrt(2)/2), charge_e),
	Nucleus((offset*math.sqrt(2)/2,0,offset*math.sqrt(2)/2), charge_e),
]

basisSet = [
	ContractedGaussian((0,0,0), sto_3g_exponents_O_1s, sto_3g_angular_momenta_O_1s, sto_3g_coefficients_O_1s),
	ContractedGaussian((0,0,0), sto_3g_exponents_O_2sp, sto_3g_angular_momenta_O_2s, sto_3g_coefficients_O_2s),
	ContractedGaussian((0,0,0), sto_3g_exponents_O_2sp, sto_3g_angular_momenta_O_2px, sto_3g_coefficients_O_2p),
	ContractedGaussian((0,0,0), sto_3g_exponents_O_2sp, sto_3g_angular_momenta_O_2py, sto_3g_coefficients_O_2p),
	ContractedGaussian((0,0,0), sto_3g_exponents_O_2sp, sto_3g_angular_momenta_O_2pz, sto_3g_coefficients_O_2p),
	ContractedGaussian((-offset*math.sqrt(2)/2,0,offset*math.sqrt(2)/2), sto_3g_exponents_H_1s, sto_3g_angular_momenta_H_1s, sto_3g_coefficients_H_1s),
	ContractedGaussian((offset*math.sqrt(2)/2,0,offset*math.sqrt(2)/2), sto_3g_exponents_H_1s, sto_3g_angular_momenta_H_1s, sto_3g_coefficients_H_1s),
]


num_electrons = 8 + 1 + 1
num_occupied_orbitals = num_electrons // 2


#S = overlapMatrix(basisSet)
#T = kineticMatrix(basisSet)
#V = potentialMatrix(basisSet, nuclei)
#W = electronRepulsionMatrix(basisSet)

S = np.load("overlap.npy")
T = np.load("kinetic.npy")
V = np.load("potential.npy")
W = np.load("electronRepulstion.npy")



# Returns the energy of the antisymmetrized wavefunction, given the density matrices of the spin up (alpha) and spin down (beta) electrons
def HF_energy(density_alpha, density_beta):
	#global S, T, V, W, nuclei

	# Kinetic + Potential Energy
	core_hamiltonian = T + V

	# Electron repulsion energy
	# Coulomb energy is between the terms in the antisymmetrized wavefunction where the order of the two indices match
	coulomb_from_alpha = np.trace(density_alpha @ W, axis1 = 2, axis2 = 3)
	coulomb_from_beta = np.trace(density_beta @ W, axis1 = 2, axis2 = 3)
	# Exchange energy is from the terms where the order of the two electrons is swapped
	exchange_from_alpha = np.trace(density_alpha @ W.transpose((0,3,2,1)), axis1 = 2, axis2 = 3)
	exchange_from_beta = np.trace(density_beta @ W.transpose((0,3,2,1)), axis1 = 2, axis2 = 3)
	# All other permutations of the other indices of the wavefunction cancel out to 0, since we assume the individual molecular orbitals used to construct it are orthonormal

	single_electron_energy = np.trace(density_alpha @ core_hamiltonian) + np.trace(density_alpha @ core_hamiltonian)

	coulomb_energy = \
		np.trace(density_alpha @ coulomb_from_alpha) + \
		np.trace(density_alpha @ coulomb_from_beta) + \
		np.trace(density_beta @ coulomb_from_alpha) + \
		np.trace(density_beta @ coulomb_from_beta)
	# We have to divide by 2 because otherwise we are double-counting (including interactions between electron 1 and electron 2, and between electron 2 and electron 1)
	coulomb_energy /= 2
	
	# Only exchange between same-spin electrons occurs; Exchanging two opposite spin electrons results in 0, since the two spin indices are orthogonal (and not affected by W)
	exchange_energy = \
		np.trace(density_alpha @ exchange_from_alpha) + \
		0 * np.trace(density_alpha @ exchange_from_beta) + \
		0 * np.trace(density_beta @ exchange_from_alpha) + \
		np.trace(density_beta @ exchange_from_beta)
	# Similarly divide by 2 to prevent double-counting
	exchange_energy /= 2
	
	# Exchange energy is subtracted because the wavefunction is anti-symmetrized, so terms with odd permutations are negated
	two_electron_energy = coulomb_energy - exchange_energy

	nuclei_repulsion_energy = nucleiRepulsionEnergy(nuclei)

	total_energy = single_electron_energy + two_electron_energy + nuclei_repulsion_energy

	return total_energy, single_electron_energy, two_electron_energy, nuclei_repulsion_energy



# Create an orthonormal basis from the existing orbital basis functions
# An easy way to do this is by using the eigenvectors of the overlap matrix. Because S is hermitian, its eigenvectors are orthogonal.
# This gives coefficients of the original orbitals which are orthogonal.

overlap_eigenvalues, overlap_eigenvectors = np.linalg.eigh(S)

# However, by default the normalized eigenvectors will not represent normalized orbitals. Instead, they will have a magnitude squared equal to their eigenvalue
# To fix this, we can just divide each one by the square root of its overlap, which will normalize them

orthonormal_basis_coefficients = overlap_eigenvectors @ np.diag(overlap_eigenvalues**-0.5)


# The initial coefficients for the self-consistent-field iterations can be chosen as the eigenvectors of the core hamiltonian

core_hamiltonian = T + V

# By default, the hamiltonian matrix is expressed in terms of the original non-orthonormal basis functions
# If H is the (infinite dimensional) hamiltonian operator, we wish to find
# Hx = Ex
# Where x is the energy eigenvector, and E is the energy eigenvalue
# However, we want the solution expressed as a linear combination of the basis functions. Let V be the matrix of basis functions, and c be a coefficient vector
# HVc = EVc
# Left-multiplying both sides by V^t gives:
# V^tHVc = EV^tVc
# V^tHV is the hamiltonian matrix evaluated with the basis functions, and V^tV is the overlap matrix, S
# Let's introduce the basis of coefficients for orthonormal orbitals X, which we created from the eigenvectors of S, divided by the square root of their eigenvalues. This means X commutes with S, and moreover X^tX = S^-1
# Replacing c with Xc':
# V^tHVXc' = EV^tVXc'
# Left-multiplying by X^t:
# X^tV^tHVXc' = EX^tV^tVXc'
# The overlap matrix cancels out, and we are left with a normal eigenvalue equation, where the matrix is now a transformed version of the hamiltonian, X^tV^tHVX:
# X^tV^tHVXc' = Ec'

# We can construct this transformed hamiltonian here:
core_hamiltonian_in_orthonormal_basis = orthonormal_basis_coefficients.T @ core_hamiltonian @ orthonormal_basis_coefficients

# And find its eigenvalues and eigenvectors:
core_eigenvalues, core_eigenvectors_in_orthonormal_basis = np.linalg.eigh(core_hamiltonian_in_orthonormal_basis)

# Its eigenvectors in the original basis can be found by transforming back:
core_eigenvectors_in_original_basis = orthonormal_basis_coefficients @ core_eigenvectors_in_orthonormal_basis

# We can use these as our initial guess for the orbital coefficients
# Make sure to only use the first n eigenvectors, corresponding to occupied orbitals
initial_coefficients = core_eigenvectors_in_original_basis[np.argsort(core_eigenvalues)][:, :num_occupied_orbitals]

# And for convenience, we can create the initial density matrix from them
initial_density = initial_coefficients @ initial_coefficients.T

# For fun, print the initial energy
initial_energy, single_electron_energy, two_electron_energy, nuclei_repulsion_energy = HF_energy(initial_density, initial_density)
print(f"Initial Energy: {initial_energy}")


# Now for the SCF iterations!

# First we must construct the Fock matrix

def fock_matrix(density_alpha, density_beta):
	# The fock matrix is very similar in form to how the total energy is calculated above (stopping before the final trace with the density), but with some minor differences
	# The core hamiltonian is as usual
	core_hamiltonian = T + V

	# The interaction energy operators are similar
	coulomb_from_alpha = np.trace(density_alpha @ W, axis1 = 2, axis2 = 3)
	coulomb_from_beta = np.trace(density_beta @ W, axis1 = 2, axis2 = 3)
	exchange_from_alpha = np.trace(density_alpha @ W.transpose((0,3,2,1)), axis1 = 2, axis2 = 3)
	exchange_from_beta = np.trace(density_beta @ W.transpose((0,3,2,1)), axis1 = 2, axis2 = 3)

	coulomb_operators = coulomb_from_alpha + coulomb_from_beta
	# Because in RHF the alpha and beta densities are assumed to be identical, it does not matter which one is used for the fock matrix
	# TODO: Is there a more meaningful way to do this, which is more fitting for non-RHF calculations?
	exchange_operators = exchange_from_alpha + 0*exchange_from_beta

	# However, unlike the total energy calculation, we do NOT divide the interaction energy by 2.
	# This is because the interaction matrix/tensor is 4 dimensional (n x n x n x n), and when taking the derivative of it's (not quadratic form, but "quartic form"), a 4 appears, rather than a 2, which cancels out the division by 2 to correct the double-counting.
	# Another way of thinking about this is because the fock matrix is for a single electron within the average potential caused by the current electron configuration, so repulsion from all the electrons should be considered
	interaction_operators = coulomb_operators - exchange_operators

	return core_hamiltonian + interaction_operators


# Now iteratively construct the fock matrix with the current coefficients/density, solve for its eigenvectors as the next set of coefficients/density, and repeat until the energy converges

def HF_iteration(initial_density):
	# Here we assume the alpha and beta densities are equal (restricted hartree fock; does not work for open-shell molecules)
	fock = fock_matrix(initial_density, initial_density)

	# We can solve for the eigenvectors just like we did for the core hamiltonian, by first transforming the fock matrix with the orthonormal coefficient matrix

	fock_matrix_in_orthonormal_basis = orthonormal_basis_coefficients.T @ fock @ orthonormal_basis_coefficients

	fock_eigenvalues, fock_eigenvectors_in_orthonormal_basis = np.linalg.eigh(fock_matrix_in_orthonormal_basis)

	fock_eigenvectors_in_original_basis = orthonormal_basis_coefficients @ fock_eigenvectors_in_orthonormal_basis

	# Only use the occupied orbitals to form the density (assuming only the lowest n orbitals are filled)
	occupied_orbital_coefficients = fock_eigenvectors_in_original_basis[np.argsort(fock_eigenvalues)][:, :num_occupied_orbitals]

	new_density = occupied_orbital_coefficients @ occupied_orbital_coefficients.T

	# Now calculate the new energy
	total_energy, single_electron_energy, two_electron_energy, nuclei_repulsion_energy = HF_energy(new_density, new_density)


	return new_density, total_energy


print("=== SCF Iterations ===")

max_iterations = 50
# When each step decreases the energy by less than this amount, stop
convergence_epsilon = 1e-8

previous_energy = initial_energy
current_density = initial_density
for i in range(max_iterations):
	new_density, total_energy = HF_iteration(current_density)
	energy_difference = total_energy - previous_energy
	print(f"Iteration {i+1}: Energy = {total_energy}, Difference = {total_energy - previous_energy}")
	previous_energy = total_energy
	current_density = new_density
	if abs(energy_difference) < convergence_epsilon:
		print("SCF Converged.")
		break


# Print the final energy
total_energy, single_electron_energy, two_electron_energy, nuclei_repulsion_energy = HF_energy(new_density, new_density)

print("=== Final Energy ===")
print(f"One-Electron Energy: {single_electron_energy}")
print(f"Two-Electron Energy: {two_electron_energy}")
print(f"Nuclei Repulsion Energy: {nuclei_repulsion_energy}")
print(f"Total Energy: {total_energy}")