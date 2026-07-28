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
	Nucleus((offset,0,0), charge_e),
	Nucleus((0,0,offset), charge_e),
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


if False:
	S = overlapMatrix(basisSet)
	np.save("overlap.npy", S)
	#print("S", S)
	T = kineticMatrix(basisSet)
	np.save("kinetic.npy", T)
	#print("T", T)
	V = potentialMatrix(basisSet, nuclei)
	np.save("potential.npy", V)
	#print("V", V)
	W = electronRepulsionMatrix(basisSet)
	np.save("electronRepulstion.npy", W)
	#print("W", W)

	print("Max/min S:", np.max(S), np.min(S))
	print("Max/min T:", np.max(T), np.min(T))
	print("Max/min V:", np.max(V), np.min(V))
	print("Max/min W:", np.max(W), np.min(W))

	exit()



def energy(coeffs, num_occupied_orbitals, nuclei, S, T, V, W):
	occupied_orbitals = coeffs[:num_occupied_orbitals]
	# One-electron operators
	one_electron_energy = 0
	for orbital_coeffs in occupied_orbitals:
		# Electron 1
		one_electron_energy += orbital_coeffs @ (T + V) @ orbital_coeffs / (orbital_coeffs @ S @ orbital_coeffs)
		# Electron 2
		one_electron_energy += orbital_coeffs @ (T + V) @ orbital_coeffs / (orbital_coeffs @ S @ orbital_coeffs)
	print("One electron energy", one_electron_energy)
	
	# Two electron operators
	two_electron_energy = 0
	for orbital_coeffs1 in occupied_orbitals:
		for orbital_coeffs2 in occupied_orbitals:
			# Electron 1 (up), Electron 1 (up)
			# Hartree energy
			two_electron_energy += orbital_coeffs2 @ (orbital_coeffs1 @ W @ orbital_coeffs1) @ orbital_coeffs2 / (orbital_coeffs1 @ S @ orbital_coeffs1) / (orbital_coeffs2 @ S @ orbital_coeffs2)
			# Fock energy
			two_electron_energy += -orbital_coeffs2 @ (orbital_coeffs1 @ W @ orbital_coeffs2) @ orbital_coeffs1 / (orbital_coeffs1 @ S @ orbital_coeffs1) / (orbital_coeffs2 @ S @ orbital_coeffs2)
			
			# Electron 1 (up), Electron 2 (down)
			# Hartree energy
			two_electron_energy += orbital_coeffs2 @ (orbital_coeffs1 @ W @ orbital_coeffs1) @ orbital_coeffs2 / (orbital_coeffs1 @ S @ orbital_coeffs1) / (orbital_coeffs2 @ S @ orbital_coeffs2)
			# Fock energy is 0 since the different spin indicies are orthogonal
			two_electron_energy += 0

			# Electron 2 (down), Electron 1 (up)
			# Hartree energy
			two_electron_energy += orbital_coeffs2 @ (orbital_coeffs1 @ W @ orbital_coeffs1) @ orbital_coeffs2 / (orbital_coeffs1 @ S @ orbital_coeffs1) / (orbital_coeffs2 @ S @ orbital_coeffs2)
			# Fock energy is 0 since the different spin indicies are orthogonal
			two_electron_energy += 0

			# Electron 2 (down), Electron 2 (down)
			# Hartree energy
			two_electron_energy += orbital_coeffs2 @ (orbital_coeffs1 @ W @ orbital_coeffs1) @ orbital_coeffs2 / (orbital_coeffs1 @ S @ orbital_coeffs1) / (orbital_coeffs2 @ S @ orbital_coeffs2)
			# Fock energy
			two_electron_energy += -orbital_coeffs2 @ (orbital_coeffs1 @ W @ orbital_coeffs2) @ orbital_coeffs1 / (orbital_coeffs1 @ S @ orbital_coeffs1) / (orbital_coeffs2 @ S @ orbital_coeffs2)
	two_electron_energy /= 2
	#for orbital_coeffs in occupied_orbitals:
	#	two_electron_energy += 0.5 * orbital_coeffs @ (orbital_coeffs @ W @ orbital_coeffs) @ orbital_coeffs / (orbital_coeffs @ S @ orbital_coeffs) / (orbital_coeffs @ S @ orbital_coeffs)
	print("Two electron energy", two_electron_energy)
	
	nuclei_repulsion_energy = nucleiRepulsionEnergy(nuclei)
	print("Nuclei repulsion energy", nuclei_repulsion_energy)
	print("Non-Nuclei energy:", one_electron_energy + two_electron_energy)
	return one_electron_energy + two_electron_energy + nuclei_repulsion_energy



def fock_operator(coeffs, num_occupied_orbitals, nuclei, S, T, V, W):
	occupied_orbitals = coeffs[:num_occupied_orbitals]

	one_electron_operators = np.zeros((len(coeffs),len(coeffs)))

	one_electron_operators += (T + V)
	
	two_electron_operators = np.zeros((len(coeffs),len(coeffs)), dtype = "complex128")
	for orbital_coeffs1 in occupied_orbitals:
		two_electron_operators += (orbital_coeffs1 @ W @ orbital_coeffs1) / (orbital_coeffs1 @ S @ orbital_coeffs1)
		two_electron_operators += -(orbital_coeffs1 @ W.transpose((2,1,0,3)) @ orbital_coeffs1) / (orbital_coeffs1 @ S @ orbital_coeffs1)

		two_electron_operators += (orbital_coeffs1 @ W @ orbital_coeffs1) / (orbital_coeffs1 @ S @ orbital_coeffs1)
		two_electron_operators += 0

		#two_electron_operators += (orbital_coeffs1 @ W @ orbital_coeffs1) / (orbital_coeffs1 @ S @ orbital_coeffs1)
		#two_electron_operators += 0

		#two_electron_operators += (orbital_coeffs1 @ W @ orbital_coeffs1) / (orbital_coeffs1 @ S @ orbital_coeffs1)
		#two_electron_operators += -(orbital_coeffs1 @ W.transpose((2,1,0,3)) @ orbital_coeffs1) / (orbital_coeffs1 @ S @ orbital_coeffs1)
	two_electron_operators /= 2
	#for orbital_coeffs in occupied_orbitals: # NO
	#	two_electron_operators += 0.5 * (orbital_coeffs @ W @ orbital_coeffs) / (orbital_coeffs @ S @ orbital_coeffs)
	
	return one_electron_operators + two_electron_operators





S = np.load("overlap.npy")
T = np.load("kinetic.npy")
V = np.load("potential.npy")
W = np.load("electronRepulstion.npy")


overlap_eigvals, overlap_eigvecs = np.linalg.eig(S)
orthonormal_basis_coeffs = overlap_eigvecs @ np.diag(overlap_eigvals**-0.5)



print("single electron best energy:", np.sort((np.linalg.eig(np.linalg.inv(S) @ (T+V)))[0]))

num_electrons = 8 + 1 + 1
num_occupied_orbitals = num_electrons // 2

NR = nucleiRepulsionEnergy(nuclei)

initial_coeffs = np.identity(len(basisSet))
#initial_coeffs, _ = np.linalg.qr(np.random.randn(len(basisSet), len(basisSet)))

print("Total Energy:",energy(initial_coeffs, num_occupied_orbitals, nuclei, S, T, V, W))

def scf_iteration(S, T, V, W, NR, initial_coeffs):
	fock = fock_operator(initial_coeffs, num_occupied_orbitals, nuclei, S, T, V, W)
	print("Total energy from fock matrix (real):",sum([2*((orbital_coeffs) @ fock @ (orbital_coeffs)) / ((orbital_coeffs) @ S @ (orbital_coeffs)) for orbital_coeffs in initial_coeffs[:num_occupied_orbitals]]))
	#exit()
	#for orbital_coeffs in initial_coeffs:
		#print((orbital_coeffs @ fock @ orbital_coeffs) / (orbital_coeffs @ S @ orbital_coeffs))
	eigenvals, eigenvecs = np.linalg.eig(orthonormal_basis_coeffs.T @ fock @ orthonormal_basis_coeffs)
	eigenvals2, eigenvecs2 = np.linalg.eig(np.linalg.inv(S) @ fock)
	print(eigenvals[np.argsort(eigenvals)])
	print(eigenvals2[np.argsort(eigenvals2)])
	#print(eigenvecs.T[np.argsort(eigenvals)])
	#print(fock @ np.array([0,0,0,1,0,0,0]))
	new_coeffs_ortho = eigenvecs.T[np.argsort(eigenvals)]
	new_coeffs = (orthonormal_basis_coeffs @ new_coeffs_ortho.T).T
	new_coeffs2 = eigenvecs2.T[np.argsort(eigenvals2)]
	#print("Old in new basis:", new_coeffs.T @ initial_coeffs)
	#print("New in new basis:", new_coeffs_ortho.T @ new_coeffs_ortho)
	print("Old energy from fock matrix (old mat, old vec):",sum([2*(orbital_coeffs @ fock @ orbital_coeffs) / (orbital_coeffs @ S @ orbital_coeffs) for orbital_coeffs in initial_coeffs[:num_occupied_orbitals]]))
	print("New energy from fock matrix (old mat, new vec):",sum([2*(orbital_coeffs @ fock @ orbital_coeffs) / (orbital_coeffs @ S @ orbital_coeffs) for orbital_coeffs in new_coeffs[:num_occupied_orbitals]]))
	print("New energy from ortho fock matrix (old mat, new vec):",sum([2*(orbital_coeffs @ orthonormal_basis_coeffs.T @ fock @ orthonormal_basis_coeffs @ orbital_coeffs) for orbital_coeffs in new_coeffs_ortho[:num_occupied_orbitals]]))
	fock1 = fock_operator(new_coeffs, num_occupied_orbitals, nuclei, S, T, V, W)
	print("New energy from fock matrix (new mat, new vec):",sum([2*(orbital_coeffs @ fock1 @ orbital_coeffs) / (orbital_coeffs @ S @ orbital_coeffs) for orbital_coeffs in new_coeffs[:num_occupied_orbitals]]))
	print("New energy2 from fock matrix (old mat, new vec2):",sum([2*(orbital_coeffs @ fock @ orbital_coeffs) / (orbital_coeffs @ S @ orbital_coeffs) for orbital_coeffs in new_coeffs2[:num_occupied_orbitals]]))
	fock2 = fock_operator(new_coeffs2, num_occupied_orbitals, nuclei, S, T, V, W)
	print("New energy2 from fock matrix (new mat, new vec2):",sum([2*(orbital_coeffs @ fock2 @ orbital_coeffs) / (orbital_coeffs @ S @ orbital_coeffs) for orbital_coeffs in new_coeffs2[:num_occupied_orbitals]]))
	#for orbital_coeffs in new_coeffs:
		#print((orbital_coeffs @ fock @ orbital_coeffs) / (orbital_coeffs @ S @ orbital_coeffs),(orbital_coeffs @ S @ orbital_coeffs))
	print(energy(new_coeffs, num_occupied_orbitals, nuclei, S, T, V, W))
	return new_coeffs

first_coeffs = initial_coeffs
for i in range(5):
	print(f"=== Iteration {i} ===")
	initial_coeffs = scf_iteration(S, T, V, W, NR, initial_coeffs)
print("Compared to initial:",energy(first_coeffs, num_occupied_orbitals, nuclei, S, T, V, W))


exit()

Wslice = np.zeros((len(basisSet), len(basisSet)))
for orbital in range(num_occupied_orbitals):
	Wslice += initial_coeffs[orbital] @ W @ initial_coeffs[orbital] / (initial_coeffs[orbital] @ S @ initial_coeffs[orbital].T)

# psi = (a1 + a2 + a3 + ...) X (a1 + a2 + a3 + ...) X (a1 + a2 + a3 + ...) X (a1 + a2 + a3 + ...)

print("Wslice", Wslice)
print("Initial T:", 2 * initial_coeffs @ (T) @ initial_coeffs / (initial_coeffs @ S @ initial_coeffs))
print("Initial V:", 2 * initial_coeffs @ (V) @ initial_coeffs / (initial_coeffs @ S @ initial_coeffs))
print("Initial T+V:", 2 * initial_coeffs @ (T+V) @ initial_coeffs / (initial_coeffs @ S @ initial_coeffs))
print("Initial W:", initial_coeffs @ (Wslice) @ initial_coeffs / (initial_coeffs @ S @ initial_coeffs))
print("Initial NR:", NR)
print("Initial Energy:", initial_coeffs @ (2*(T + V) + Wslice) @ initial_coeffs / (initial_coeffs @ S @ initial_coeffs) + NR)

exit()

def scf_iteration(S, T, V, W, NR, initial_coeffs):
	Wslice = initial_coeffs @ W @ initial_coeffs / (initial_coeffs @ S @ initial_coeffs.T)
	newMatrix = np.linalg.inv(S) @ (2*(T + V) + Wslice)
	eigenvals, eigenvecs = np.linalg.eig(newMatrix)
	newCoeffs = eigenvecs.T[np.argsort(eigenvals)[0]]
	print("Coeffs:", newCoeffs)
	newWslice = newCoeffs @ W @ newCoeffs / (newCoeffs @ S @ newCoeffs.T)
	print("Energy:", newCoeffs @ (2*(T + V) + newWslice) @ newCoeffs / (newCoeffs @ S @ newCoeffs) + NR)
	return newCoeffs

for i in range(10):
	print(f"=== SCF Iteration {i} ===")
	initial_coeffs = scf_iteration(S, T, V, W, NR, initial_coeffs)



