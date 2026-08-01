from gaussian_integrals import *
import numpy as np
import math

bohr_radius = 5.291772e-11
h_bar = 6.626070e-34 / (2*math.pi)
mass_e = 9.1093837139e-31
charge_e = 1.602176634e-19
epsilon0 = 8.8541878188e-12
ev_per_hartree = 27.211386245981
hartree_per_joule = 1 / charge_e / ev_per_hartree


class Nucleus():
	def __init__(self, position, charge, symbol):
		self.position = np.array(position)
		self.charge = charge
		self.symbol = symbol

def nucleiRepulsionEnergy(nuclei):
	total = 0
	for i, nucleus1 in enumerate(nuclei):
		for j, nucleus2 in enumerate(nuclei):
			if i < j:
				total += 1/(4*math.pi*epsilon0) * nucleus1.charge * nucleus2.charge / (np.sqrt(np.sum((nucleus2.position - nucleus1.position)**2)))
	return total * hartree_per_joule





class HermiteGaussian():
	def __init__(self, position, exponent, angular_momentum):
		self.position = np.array(position)
		self.exponent = exponent
		self.angular_momentum = np.array(angular_momentum)

class CartesianGaussian():
	def __init__(self, position, exponent, angular_momentum):
		self.position = np.array(position)
		self.exponent = exponent
		self.angular_momentum = angular_momentum
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
		self.position = position
		self.exponents = exponents
		self.angular_momenta = angular_momenta
		self.cartesian_coefficients = cartesian_coefficients
		self.cartesian_gaussians = []
		self.hermite_coefficients = []
		self.hermite_gaussians = []
		for i in range(len(self.cartesian_coefficients)):
			cartesianGaussian = CartesianGaussian(position, exponents[i], angular_momenta[i])
			self.cartesian_gaussians.append(cartesianGaussian)
			self.hermite_coefficients += [coeff*self.cartesian_coefficients[i]*cartesianGaussian.normalizationFactor for coeff in cartesianGaussian.hermite_coefficients if coeff != 0]
			self.hermite_gaussians += [cartesianGaussian.hermite_gaussians[i] for i in range(len(cartesianGaussian.hermite_gaussians)) if cartesianGaussian.hermite_coefficients[i] != 0]
		self.selfOverlap = 0
		for i, gaussian1 in enumerate(self.hermite_gaussians):
			for j, gaussian2 in enumerate(self.hermite_gaussians):
				self.selfOverlap += self.hermite_coefficients[i] * self.hermite_coefficients[j] \
					* overlap_integral_hermite(*gaussian1.position, gaussian1.exponent, *gaussian1.angular_momentum, *gaussian2.position, gaussian2.exponent, *gaussian2.angular_momentum)
		self.normalizationFactor = 1 / math.sqrt(self.selfOverlap)
