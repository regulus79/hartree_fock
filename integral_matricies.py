from integral_implementations1 import *
import math
import numpy as np

bohr_radius = 5.291772e-11
h_bar = 6.626070e-34 / (2*math.pi)
mass_e = 9.1093837139e-31
charge_e = 1.602176634e-19
epsilon0 = 8.8541878188e-12
ev_per_hartree = 27.211386245981
hartree_per_joule = 1 / charge_e / ev_per_hartree

class Nucleus():
	def __init__(self, position, charge):
		self.position = np.array(position)
		self.charge = charge

class HermiteGaussian():
	def __init__(self, position, exponent, angular_momentum):
		self.position = np.array(position)
		self.exponent = exponent
		self.angular_momentum = np.array(angular_momentum)

class CartesianGaussian():
	def __init__(self, position, exponent, angular_momentum):
		self.position = np.array(position)
		self.exponent = exponent
		self.hermite_coefficients = []
		self.hermite_gaussians = []
		hermite_coeffsX = monomial_to_hermite_polynomials(angular_momentum[0], exponent)
		hermite_coeffsY = monomial_to_hermite_polynomials(angular_momentum[1], exponent)
		hermite_coeffsZ = monomial_to_hermite_polynomials(angular_momentum[2], exponent)
		for orderX, coeffX in enumerate(hermite_coeffsX):
			for orderY, coeffY in enumerate(hermite_coeffsY):
				for orderZ, coeffZ in enumerate(hermite_coeffsZ):
					self.hermite_coefficients.append(coeffX * coeffY * coeffZ)
					self.hermite_gaussians.append(HermiteGaussian(position, exponent, (orderX, orderY, orderZ)))
		self.selfOverlap = 0
		for i, gaussian1 in enumerate(self.hermite_gaussians):
			for j, gaussian2 in enumerate(self.hermite_gaussians):
				self.selfOverlap += self.hermite_coefficients[i] * self.hermite_coefficients[j] \
					* overlap_integral_hermite(*gaussian1.position, gaussian1.exponent, *gaussian1.angular_momentum, *gaussian2.position, gaussian2.exponent, *gaussian2.angular_momentum)
		self.normalizationFactor = 1 / math.sqrt(self.selfOverlap)

class ContractedGaussian():
	def __init__(self, position, exponents, angular_momenta, cartesian_coefficients):
		self.cartesian_coefficients = cartesian_coefficients
		self.hermite_coefficients = []
		self.hermite_gaussians = []
		for i in range(len(self.cartesian_coefficients)):
			cartesianGaussian = CartesianGaussian(position, exponents[i], angular_momenta[i])
			self.hermite_coefficients += [coeff*self.cartesian_coefficients[i]*cartesianGaussian.normalizationFactor for coeff in cartesianGaussian.hermite_coefficients if coeff != 0]
			self.hermite_gaussians += [cartesianGaussian.hermite_gaussians[i] for i in range(len(cartesianGaussian.hermite_gaussians)) if cartesianGaussian.hermite_coefficients[i] != 0]
		self.selfOverlap = 0
		for i, gaussian1 in enumerate(self.hermite_gaussians):
			for j, gaussian2 in enumerate(self.hermite_gaussians):
				self.selfOverlap += self.hermite_coefficients[i] * self.hermite_coefficients[j] \
					* overlap_integral_hermite(*gaussian1.position, gaussian1.exponent, *gaussian1.angular_momentum, *gaussian2.position, gaussian2.exponent, *gaussian2.angular_momentum)
		self.normalizationFactor = 1 / math.sqrt(self.selfOverlap)



def overlapMatrixElement(contractedGaussian1, contractedGaussian2):
	total = 0
	for i, gaussian1 in enumerate(contractedGaussian1.hermite_gaussians):
		for j, gaussian2 in enumerate(contractedGaussian2.hermite_gaussians):
			total += contractedGaussian1.hermite_coefficients[i] * contractedGaussian2.hermite_coefficients[j] \
				* overlap_integral_hermite(*gaussian1.position, gaussian1.exponent, *gaussian1.angular_momentum, *gaussian2.position, gaussian2.exponent, *gaussian2.angular_momentum)
	total *= contractedGaussian1.normalizationFactor
	total *= contractedGaussian2.normalizationFactor
	return total

def kineticMatrixElement(contractedGaussian1, contractedGaussian2):
	total = 0
	for i, gaussian1 in enumerate(contractedGaussian1.hermite_gaussians):
		for j, gaussian2 in enumerate(contractedGaussian2.hermite_gaussians):
			laplacianX = overlap_integral_hermite(*gaussian1.position, gaussian1.exponent, *gaussian1.angular_momentum, *gaussian2.position, gaussian2.exponent, *(gaussian2.angular_momentum + np.array([2,0,0])))
			laplacianY = overlap_integral_hermite(*gaussian1.position, gaussian1.exponent, *gaussian1.angular_momentum, *gaussian2.position, gaussian2.exponent, *(gaussian2.angular_momentum + np.array([0,2,0])))
			laplacianZ = overlap_integral_hermite(*gaussian1.position, gaussian1.exponent, *gaussian1.angular_momentum, *gaussian2.position, gaussian2.exponent, *(gaussian2.angular_momentum + np.array([0,0,2])))
			total += contractedGaussian1.hermite_coefficients[i] * contractedGaussian2.hermite_coefficients[j] \
				* -h_bar**2 / (2*mass_e) * (laplacianX + laplacianY + laplacianZ)
	total *= contractedGaussian1.normalizationFactor
	total *= contractedGaussian2.normalizationFactor
	total *= hartree_per_joule
	return total

def potentialMatrixElement(contractedGaussian1, contractedGaussian2, nuclei):
	total = 0
	for i, gaussian1 in enumerate(contractedGaussian1.hermite_gaussians):
		for j, gaussian2 in enumerate(contractedGaussian2.hermite_gaussians):
			for nucleus in nuclei:
				total += contractedGaussian1.hermite_coefficients[i] * contractedGaussian2.hermite_coefficients[j] \
					* 1/(4*math.pi*epsilon0) * -charge_e * nucleus.charge * nuclear_attraction_integral_hermite(*gaussian1.position, gaussian1.exponent, *gaussian1.angular_momentum, *gaussian2.position, gaussian2.exponent, *gaussian2.angular_momentum, *nucleus.position)
	total *= contractedGaussian1.normalizationFactor
	total *= contractedGaussian2.normalizationFactor
	total *= hartree_per_joule
	return total

def electronRepulsionMatrixElement(contractedGaussian1, contractedGaussian2, contractedGaussian3, contractedGaussian4):
	total = 0
	for i, gaussian1 in enumerate(contractedGaussian1.hermite_gaussians):
		for j, gaussian2 in enumerate(contractedGaussian2.hermite_gaussians):
			for k, gaussian3 in enumerate(contractedGaussian3.hermite_gaussians):
				for l, gaussian4 in enumerate(contractedGaussian4.hermite_gaussians):
					total += contractedGaussian1.hermite_coefficients[i] * contractedGaussian2.hermite_coefficients[j] * contractedGaussian3.hermite_coefficients[k] * contractedGaussian4.hermite_coefficients[l] \
						* 1/(4*math.pi*epsilon0) * -charge_e * -charge_e * electron_repulsion_integral_hermite(
							*gaussian1.position, gaussian1.exponent, *gaussian1.angular_momentum,
							*gaussian2.position, gaussian2.exponent, *gaussian2.angular_momentum,
							*gaussian3.position, gaussian3.exponent, *gaussian3.angular_momentum, 
							*gaussian4.position, gaussian4.exponent, *gaussian4.angular_momentum,
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
			print(f"Computing Overlap Integrals... {i*len(basisSet)+j+1}/{len(basisSet)**2}", end="\r")
			if i <= j:
				S[i][j] = overlapMatrixElement(basisSet[i], basisSet[j])
				S[j][i] = S[i][j]
	print("")
	return S

def kineticMatrix(basisSet):
	T = np.zeros((len(basisSet), len(basisSet)))
	for i in range(len(basisSet)):
		for j in range(len(basisSet)):
			print(f"Computing Kinetic Integrals... {i*len(basisSet)+j+1}/{len(basisSet)**2}", end="\r")
			if i <= j:
				T[i][j] = kineticMatrixElement(basisSet[i], basisSet[j])
				T[j][i] = T[i][j]
	print("")
	return T

def potentialMatrix(basisSet, nuclei):
	V = np.zeros((len(basisSet), len(basisSet)))
	for i in range(len(basisSet)):
		for j in range(len(basisSet)):
			print(f"Computing Potential Integrals... {i*len(basisSet)+j+1}/{len(basisSet)**2}", end="\r")
			if i <= j:
				V[i][j] = potentialMatrixElement(basisSet[i], basisSet[j], nuclei)
				V[j][i] = V[i][j]
	print("")
	return V

def electronRepulsionMatrix(basisSet):
	W = np.zeros((len(basisSet), len(basisSet), len(basisSet), len(basisSet)))
	for i in range(len(basisSet)):
		for j in range(len(basisSet)):
			for k in range(len(basisSet)):
				for l in range(len(basisSet)):
					print(f"Computing Electron Repulsion Integrals... {i*len(basisSet)**3+j*len(basisSet)**2+k*len(basisSet)+l+1}/{len(basisSet)**4}", end="\r")
					if i <= j and k <= l:
						# Note the inidices are i,k,j,l instead of i,j,k,l, since when doing coeffs @ W @ coeffs, numpy multiplies down the last two indicies
						# so to make sure one of each vector/covector indices are used, we need to flip them
						W[i][j][k][l] = electronRepulsionMatrixElement(basisSet[i], basisSet[k], basisSet[j], basisSet[l])
						W[i][j][l][k] = W[i][j][k][l]
						W[j][i][k][l] = W[i][j][k][l]
						W[j][i][l][k] = W[i][j][k][l]
	print("")
	return W


def nucleiRepulsionEnergy(nuclei):
	total = 0
	for i, nucleus1 in enumerate(nuclei):
		for j, nucleus2 in enumerate(nuclei):
			if i < j:
				total += 1/(4*math.pi*epsilon0) * nucleus1.charge * nucleus2.charge / (np.sqrt(np.sum((nucleus2.position - nucleus1.position)**2)))
	return total * hartree_per_joule
