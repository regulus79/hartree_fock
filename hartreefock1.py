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
	Nucleus((-offset,0,0), charge_e*8),
]

basisSet = [
	ContractedGaussian((0,0,0), sto_3g_exponents_O_1s, sto_3g_angular_momenta_O_1s, sto_3g_coefficients_O_1s),
	ContractedGaussian((0,0,0), sto_3g_exponents_O_2sp, sto_3g_angular_momenta_O_2s, sto_3g_coefficients_O_2s),
	ContractedGaussian((0,0,0), sto_3g_exponents_O_2sp, sto_3g_angular_momenta_O_2pz, sto_3g_coefficients_O_2p),
	ContractedGaussian((0,0,0), sto_3g_exponents_O_2sp, sto_3g_angular_momenta_O_2py, sto_3g_coefficients_O_2p),
	ContractedGaussian((0,0,0), sto_3g_exponents_O_2sp, sto_3g_angular_momenta_O_2px, sto_3g_coefficients_O_2p),
	ContractedGaussian((-offset,0,0), sto_3g_exponents_O_1s, sto_3g_angular_momenta_O_1s, sto_3g_coefficients_O_1s),
	ContractedGaussian((-offset,0,0), sto_3g_exponents_O_2sp, sto_3g_angular_momenta_O_2s, sto_3g_coefficients_O_2s),
	ContractedGaussian((-offset,0,0), sto_3g_exponents_O_2sp, sto_3g_angular_momenta_O_2pz, sto_3g_coefficients_O_2p),
	ContractedGaussian((-offset,0,0), sto_3g_exponents_O_2sp, sto_3g_angular_momenta_O_2py, sto_3g_coefficients_O_2p),
	ContractedGaussian((-offset,0,0), sto_3g_exponents_O_2sp, sto_3g_angular_momenta_O_2px, sto_3g_coefficients_O_2p),
]



S = overlapMatrix(basisSet)
print("S", S)
T = kineticMatrix(basisSet)
print("T", T)
V = potentialMatrix(basisSet, nuclei)
print("V", V)
W = electronRepulsionMatrix(basisSet)
print("W", W)

print("Max/min S:", np.max(S), np.min(S))
print("Max/min T:", np.max(T), np.min(T))
print("Max/min V:", np.max(V), np.min(V))
print("Max/min W:", np.max(W), np.min(W))

exit()


NR = nucleiRepulsionEnergy(nuclei)

initial_coeffs = np.array([1,0])
Wslice = initial_coeffs @ W @ initial_coeffs / (initial_coeffs @ S @ initial_coeffs.T)
print("Wslice", Wslice)
print("Initial T:", 2 * initial_coeffs @ (T) @ initial_coeffs / (initial_coeffs @ S @ initial_coeffs))
print("Initial V:", 2 * initial_coeffs @ (V) @ initial_coeffs / (initial_coeffs @ S @ initial_coeffs))
print("Initial T+V:", 2 * initial_coeffs @ (T+V) @ initial_coeffs / (initial_coeffs @ S @ initial_coeffs))
print("Initial W:", initial_coeffs @ (Wslice) @ initial_coeffs / (initial_coeffs @ S @ initial_coeffs))
print("Initial NR:", NR)
print("Initial Energy:", initial_coeffs @ (2*(T + V) + Wslice) @ initial_coeffs / (initial_coeffs @ S @ initial_coeffs) + NR)


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



