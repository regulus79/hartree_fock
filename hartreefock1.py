from integral_implementations1 import overlap_integral, overlap_integral_derivative, nuclear_attraction_integral, electron_repulsion_integral
import math
import numpy as np

bohr_radius = 5.291772e-11
h_bar = 6.626070e-34 / (2*math.pi)
mass_e = 9.109384e-31
charge_e = 1.602176e-19
epsilon0 = 8.854188e-12
ev_per_hartree = 27.211386245981
hartree_per_joule = 1 / charge_e / ev_per_hartree

class Nucleus():
	def __init__(self, position, charge):
		self.position = np.array(position)
		self.charge = charge

class Gaussian():
	def __init__(self, position, exponent, angular_momentum):
		self.position = np.array(position)
		self.exponent = exponent
		self.angular_momentum = angular_momentum # NOT IMPLEMENTED

class ContractedGaussian():
	def __init__(self, position, exponents, angular_momenta, coefficients):
		self.coefficients = coefficients
		self.gaussians = []
		for i in range(len(self.coefficients)):
			self.gaussians.append(Gaussian(position, exponents[i], angular_momenta[i]))
		self.selfOverlap = 0
		for i, gaussian1 in enumerate(self.gaussians):
			for j, gaussian2 in enumerate(self.gaussians):
				self.selfOverlap += self.coefficients[i] * self.coefficients[j] * overlap_integral(*gaussian1.position, gaussian1.exponent, *gaussian2.position, gaussian2.exponent)
		self.normalizationFactor = 1 / math.sqrt(self.selfOverlap)



def overlapMatrixElement(contractedGaussian1, contractedGaussian2):
	total = 0
	for i, gaussian1 in enumerate(contractedGaussian1.gaussians):
		for j, gaussian2 in enumerate(contractedGaussian2.gaussians):
			print(i,j, contractedGaussian1.coefficients[i], contractedGaussian2.coefficients[j], overlap_integral(*gaussian1.position, gaussian1.exponent, *gaussian2.position, gaussian2.exponent) * contractedGaussian1.normalizationFactor * contractedGaussian2.normalizationFactor)
			total += contractedGaussian1.coefficients[i] * contractedGaussian2.coefficients[j] \
				* overlap_integral(*gaussian1.position, gaussian1.exponent, *gaussian2.position, gaussian2.exponent)
	total *= contractedGaussian1.normalizationFactor
	total *= contractedGaussian2.normalizationFactor
	print(total)
	return total

def kineticMatrixElement(contractedGaussian1, contractedGaussian2):
	total = 0
	for i, gaussian1 in enumerate(contractedGaussian1.gaussians):
		for j, gaussian2 in enumerate(contractedGaussian2.gaussians):
			laplacianX = overlap_integral_derivative(*gaussian1.position, gaussian1.exponent, *gaussian2.position, gaussian2.exponent, 2, 0, 0)
			laplacianY = overlap_integral_derivative(*gaussian1.position, gaussian1.exponent, *gaussian2.position, gaussian2.exponent, 0, 2, 0)
			laplacianZ = overlap_integral_derivative(*gaussian1.position, gaussian1.exponent, *gaussian2.position, gaussian2.exponent, 0, 0, 2)
			total += contractedGaussian1.coefficients[i] * contractedGaussian2.coefficients[j] \
				* -h_bar**2 / (2*mass_e) * (laplacianX + laplacianY + laplacianZ)
	total *= contractedGaussian1.normalizationFactor
	total *= contractedGaussian2.normalizationFactor
	total *= hartree_per_joule
	return total

def potentialMatrixElement(contractedGaussian1, contractedGaussian2, nuclei):
	total = 0
	for i, gaussian1 in enumerate(contractedGaussian1.gaussians):
		for j, gaussian2 in enumerate(contractedGaussian2.gaussians):
			for nucleus in nuclei:
				total += contractedGaussian1.coefficients[i] * contractedGaussian2.coefficients[j] \
					* 1/(4*math.pi*epsilon0) * -charge_e * nucleus.charge * nuclear_attraction_integral(*gaussian1.position, gaussian1.exponent, *gaussian2.position, gaussian2.exponent, *nucleus.position)
	total *= contractedGaussian1.normalizationFactor
	total *= contractedGaussian2.normalizationFactor
	total *= hartree_per_joule
	return total

def electronRepulsionMatrixElement(contractedGaussian1, contractedGaussian2, contractedGaussian3, contractedGaussian4):
	total = 0
	for i, gaussian1 in enumerate(contractedGaussian1.gaussians):
		for j, gaussian2 in enumerate(contractedGaussian2.gaussians):
			for k, gaussian3 in enumerate(contractedGaussian3.gaussians):
				for l, gaussian4 in enumerate(contractedGaussian4.gaussians):
					total += contractedGaussian1.coefficients[i] * contractedGaussian2.coefficients[j] * contractedGaussian3.coefficients[k] * contractedGaussian4.coefficients[l] \
						* 1/(4*math.pi*epsilon0) * -charge_e * -charge_e * electron_repulsion_integral(
							*gaussian1.position, gaussian1.exponent,
							*gaussian3.position, gaussian3.exponent,
							*gaussian2.position, gaussian2.exponent,
							*gaussian4.position, gaussian4.exponent,
						)
	total *= contractedGaussian1.normalizationFactor
	total *= contractedGaussian2.normalizationFactor
	total *= contractedGaussian3.normalizationFactor
	total *= contractedGaussian4.normalizationFactor
	total *= hartree_per_joule
	return total



def overlapMatrix(basisSet):
	S = np.zeros((len(basisSet), len(basisSet)))
	for i in range(len(basisSet)):
		for j in range(len(basisSet)):
			S[i][j] = overlapMatrixElement(basisSet[i], basisSet[j])
	return S

def kineticMatrix(basisSet):
	T = np.zeros((len(basisSet), len(basisSet)))
	for i in range(len(basisSet)):
		for j in range(len(basisSet)):
			T[i][j] = kineticMatrixElement(basisSet[i], basisSet[j])
	return T

def potentialMatrix(basisSet, nuclei):
	V = np.zeros((len(basisSet), len(basisSet)))
	for i in range(len(basisSet)):
		for j in range(len(basisSet)):
			V[i][j] = potentialMatrixElement(basisSet[i], basisSet[j], nuclei)
	return V

def electronRepulsionMatrix(basisSet):
	W = np.zeros((len(basisSet), len(basisSet), len(basisSet), len(basisSet)))
	for i in range(len(basisSet)):
		for j in range(len(basisSet)):
			for k in range(len(basisSet)):
				for l in range(len(basisSet)):
					W[i][j][k][l] = electronRepulsionMatrixElement(basisSet[i], basisSet[j], basisSet[k], basisSet[l])
	return W


def nucleiRepulsionEnergy(nuclei):
	total = 0
	for i, nucleus1 in enumerate(nuclei):
		for j, nucleus2 in enumerate(nuclei):
			if i < j:
				total += 1/(4*math.pi*epsilon0) * nucleus1.charge * nucleus2.charge / (np.sqrt(np.sum((nucleus2.position - nucleus1.position)**2)))
	return total * hartree_per_joule




sto_3g_exponents = np.array([
	0.3425250914E+01,
	0.6239137298E+00,
	0.1688554040E+00
]) / bohr_radius**2
# REMEMBER
# The coefficients in these gaussian basis sets assume the individual gaussians are already normalized, or multiplied by (2*exponent/pi)**(3/4)
# To turn the coeffs into non-normalized gaussian coeffs, multiply by the normalization factors
sto_3g_coefficients = np.array([
	0.1543289673E+00,
	0.5353281423E+00,
	0.4446345422E+00
]) * (2 * sto_3g_exponents / np.pi)**(3/4)



offset = 0.7414 * 1e-10

nuclei = [
	Nucleus((0,0,0), charge_e),
	Nucleus((offset,0,0), charge_e)
]

basisSet = [
	ContractedGaussian((0,0,0), sto_3g_exponents, [(0,0,0), (0,0,0), (0,0,0)], sto_3g_coefficients),
	ContractedGaussian((offset,0,0), sto_3g_exponents, [(0,0,0), (0,0,0), (0,0,0)], sto_3g_coefficients),
]


S = overlapMatrix(basisSet)
T = kineticMatrix(basisSet)
V = potentialMatrix(basisSet, nuclei)
W = electronRepulsionMatrix(basisSet)
print("S", S)
print("T", T)
print("V", V)
print("W", W)

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

for i in range(3):
	print(f"=== SCF Iteration {i} ===")
	initial_coeffs = scf_iteration(S, T, V, W, NR, initial_coeffs)



