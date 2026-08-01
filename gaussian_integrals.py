import math
import numpy as np



# The "Boys" Function, defined as the integral of x^(2n) * e^(-Tx^2) dx from 0 to 1
# "n" is the order of the Boys function. This is useful because the derivative of the Boys function wrt T simply brings
# down an extra factor of -t^2, which is the same as increasing the order n by 1 (and negating)
# The higher order Boys functions can be computed with a recurrance relation, which is used here.
def boys(n, T):
	if n == 0:
		if T == 0:
			return 1
		return math.erf(math.sqrt(T))/math.sqrt(T) * math.sqrt(math.pi)/2
	else:
		if T == 0:
			return 1 / (2*n + 1)
		return (n - 1/2) / T * boys(n - 1, T) - math.exp(-T)/(2*T)

# Computes the n'th hermite polynomial using the standard recurrance relation.
# As the hermite polynomials are formed by taking teh derivatives of a gaussian, it is handy to be able to specify
# the exponent/scaling of the gaussian, which here is given by the variable p
def hermite(n, x, p):
	hermitePrev = 0
	hermiteCurrent = 1
	for i in range(n):
		hermiteNext = -2*p*x*hermiteCurrent - 2*i*p*hermitePrev
		#print(hermitePrev, hermiteCurrent, hermiteNext)
		hermitePrev = hermiteCurrent
		hermiteCurrent = hermiteNext
	return hermiteCurrent


# It turns out one can also define horizontal recurrance relations for the hermite polynomials
# This is particularly useful when trying to express x^n as a sum of hermite polynomials (when trying to express a cartesian gaussian x^n*e^-x^2 as a sum of hermite gaussians)
# Given the coefficient of the x^i term of the n'th hermite polynomial, this function returns the coefficient of the x^(i-2) term of the n'th hermite polynomial (with p being the exponent of e^-(px^2)
def hermite_left_recurrance(i,n,p, current_coeff):
	return i*(i-1) / (2*(i-1) - 2*(n+1)) / p * current_coeff

# Given a cartesian gaussian such as x^3 * e^-x^2, express the x^3 as a sum of hermite polynomials
def monomial_to_hermite_polynomials(n, p):
	# First two terms to ground the recursion
	if n == 0:
		return np.array([1])
	if n == 1:
		return np.array([0, -1/2 / p])
	coeffs = np.zeros(n+1,)
	coeffs[-1] = 1
	# The first hermite polynomial of degree n is the n'th hermite polynomial
	# But it also comtains some lower order terms which we need to cancel out with earlier hermite polynomials to end up with only x^n
	# We can generate the other terms via the leftward recurrance relation
	# The rightmost term always has a coeff of (-2p)^n
	hermite_term_coefficient = (-2*p)**n
	for i in range(n, 0+1, -2): #+1 on lower bound to prevent it from reaching 1 (since i-2 would be -1, which is negative. And i=1 is a bse value already, it returns [0, -2] above)
		hermite_term_coefficient = hermite_left_recurrance(i, n, p, hermite_term_coefficient)
		# We need to cancel out this term by adding some linear combination of the previous hermite polynomials. What linear combination will result in this coeff*x^i term?
		# Of course! That's the whole point of this function, to express monomials like x^i as a sum of hermites. So let's just do some recursion.
		coeffs[0:i-2+1] += -hermite_term_coefficient * monomial_to_hermite_polynomials(i-2, p)
	# Finally, divide by (-2*p)^n so that the x^i term has a coeff of 1
	return coeffs / (-2*p)**n


# Multiplying two gaussians results in another gaussian, so the inner product between two gaussians can be expressed as the area
# under a single gaussian with exponent (p1 + p2) and scaled by a constant e^-(p1p2/(p1+p2) * (position difference squared))
def overlap_integral_primative(x1,y1,z1,p1, x2,y2,z2,p2):
	return math.exp(-(p1*p2) / (p1+p2) * ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)) * math.sqrt(math.pi / (p1+p2))**3

# The overlap integral can easily be defined for derivatives of gaussians
def overlap_integral_hermite(x1,y1,z1,p1,lx1,ly1,lz1, x2,y2,z2,p2,lx2,ly2,lz2):
	B = (p1*p2) / (p1+p2)
	deltax = x2 - x1
	deltay = y2 - y1
	deltaz = z2 - z1
	signx = (-1)**lx1 * (-1)**lx2 * (-1)**lx2 # Since each d/dc is the negative of its d/dx, but also when taking the final derivative, it's the difference (c2-c1), so the sign of one is negative
	signy = (-1)**ly1 * (-1)**ly2 * (-1)**ly2
	signz = (-1)**lz1 * (-1)**lz2 * (-1)**lz2
	return signx * signy * signz * (hermite(lx1+lx2, deltax, B) * hermite(ly1+ly2, deltay, B) * hermite(lz1+lz2, deltaz, B)) * overlap_integral_primative(x1,y1,z1,p1, x2,y2,z2,p2)


def nuclear_attraction_integral_primative(x1,y1,z1,p1, x2,y2,z2,p2, xn, yn, zn):
	x1 -= xn; x2 -= xn
	y1 -= yn; y2 -= yn
	z1 -= zn; z2 -= zn
	A = math.exp(-(p1*p2) / (p1+p2) * ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2))
	B = p1+p2
	Cx = (p1*x1 + p2*x2) / (p1 + p2)
	Cy = (p1*y1 + p2*y2) / (p1 + p2)
	Cz = (p1*z1 + p2*z2) / (p1 + p2)
	C = math.sqrt(Cx**2 + Cy**2 + Cz**2)
	return 2 * math.pi/B * A * boys(0, B*C**2)


def electron_repulsion_integral_primative(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4):
	A1 = math.exp(-(p1*p3) / (p1+p3) * ((x3-x1)**2 + (y3-y1)**2 + (z3-z1)**2))
	B1 = p1+p3
	A2 = math.exp(-(p2*p4) / (p2+p4) * ((x4-x2)**2 + (y4-y2)**2 + (z4-z2)**2))
	B2 = p2+p4
	C1x = (p1*x1 + p3*x3) / (p1 + p3)
	C1y = (p1*y1 + p3*y3) / (p1 + p3)
	C1z = (p1*z1 + p3*z3) / (p1 + p3)
	C2x = (p2*x2 + p4*x4) / (p2 + p4)
	C2y = (p2*y2 + p4*y4) / (p2 + p4)
	C2z = (p2*z2 + p4*z4) / (p2 + p4)
	R = math.sqrt((C2x - C1x)**2 + (C2y - C1y)**2 + (C2z - C1z)**2)
	T = R**2 * (B1*B2)/(B1+B2)
	return 2*math.pi**(5/2) * A1*A2 / (B1*B2*math.sqrt(B1+B2)) * boys(0, T)



# Okay, so this is not awesome.
# I tried to work out my own recurrance relations for the nuclear repulsion integrals
# And they work! But they are very slow and very convoluted.
#
#
# Essentially, we first define a function, R, with some fancy subscript and superscripts: R^0_0,0,0,0,0,0
# R^0_0,0,0,0,0,0 is the default nuclear attraction integral: e^(-p1p2/(p1+p2) * ((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)) * int_0^1 e^(-(p1+p2)*(((p1x1+p2x2)/(p1+p2))^2+((p1y1+p2y2)/(p1+p2))^2+((p1z1+p2z2)/(p1+p2))^2)*t^2) dt
# Adding a t^2n in the boys integral increases the superscript: R^n_0,0,0,0,0,0
# Multiplying it by x1 increments the x1 subscript: R^0_1,0,0,0,0,0, and likewise for y1: R^0_0,1,0,0,0,0, or x2: R^0_0,0,0,1,0,0
# So exentially, x1*R^0_0,0,0,0,0,0 = R^0_1,0,0,0,0,0
#
# The derivative of R^0_0,0,0,0,0,0 wrt x2 is
# -2*(p1p2/(p1+p2))*(x2-x1) * R^0_0,0,0,0,0,0 + -2*(p1+p2)*((p1x1+p2x2)/(p1+p2))*(p2/(p1+p2)) * R^1_0,0,0,0,0,0
# wrt x1 is similar
# -2*(p1p2/(p1+p2))*(x2-x1)*-1 * R^0_0,0,0,0,0,0 + -2*(p1+p2)*((p1x1+p2x2)/(p1+p2))*(p1/(p1+p2)) * R^1_0,0,0,0,0,0
# By grouping the x1,x2 with the R terms, we can take advantage of the subscript notation, so
# -2*(p1p2/(p1+p2))*(x2-x1) * R^0_0,0,0,0,0,0 + -2*(p1+p2)*((p1x1+p2x2)/(p1+p2))*(p2/(p1+p2)) * R^1_0,0,0,0,0,0
# becomes
# -2*(p1p2/(p1+p2))*(R^0_0,0,0,1,0,0 - R^0_1,0,0,0,0,0) + -2*(p1+p2)*(p2/(p1+p2)) * (p1/(p1+p2)*R^1_1,0,0,0,0,0 + p2/(p1+p2)*R^1_0,0,0,1,0,0)
# Now we have expressed the derivative of R^0_0,0,0,0,0,0 in terms of a linear combination of higher R terms
#
# The derivatives of the higher R terms can also be found:
# Forunately, the derivative of R^n_0,0,0,0,0,0 is the same as the derivative of R^0_0,0,0,0,0,0, but with +n in the resulting superscripts
# The derivative of the subscript terms:
# Because R^0_n,0,0,0,0,0 = x1^n*R^0_0,0,0,0,0,0, the derivative of R^0_n,0,0,0,0,0 = nR^0_(n-1),0,0,0,0,0 + the derivative of R^0_0,0,0,0,0,0

# Here, nx1,ny1, etc correspond to the subscripts of R, so the powers of x1,y1,etc in front of the R term
def R_nuclear_attraction(x1,y1,z1,p1, x2,y2,z2,p2, nx1,ny1,nz1,nx2,ny2,nz2, n_boys):
	A = math.exp(-(p1*p2/(p1+p2)) * ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2))
	B = (p1+p2)
	Cx = (p1*x1+p2*x2) / B
	Cy = (p1*y1+p2*y2) / B
	Cz = (p1*z1+p2*z2) / B
	return 2*math.pi/B * A * boys(n_boys, B*(Cx**2 + Cy**2 + Cz**2)) * x1**nx1 * y1**ny1 * z1**nz1 * x2**nx2 * y2**ny2 * z2**nz2



# This function takes derivatives of the R function
# The wrt_list has a list of the remaining indicies which need derivatives taken. It takes the next derivative by expanding it in terms of
# higher order R functions, and recursively calls this function to take the derivatives of those, down the line.
def nuclear_attraction_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, nx1,ny1,nz1,nx2,ny2,nz2, n_boys, wrt_list):
	wrt_list_copy = wrt_list.copy()
	# If there are no more derivatives to take, evalute the R function
	if wrt_list_copy == []:
		return R_nuclear_attraction(x1,y1,z1,p1, x2,y2,z2,p2, nx1,ny1,nz1,nx2,ny2,nz2, n_boys)
	wrt = wrt_list_copy.pop()
	wrt_other = wrt+3 if wrt<3 else wrt-3
	wrt_p = p1 if wrt<3 else p2
	wrt_p_other = p2 if wrt<3 else p1
	subscripts = [nx1,ny1,nz1,nx2,ny2,nz2]
	subscripts_this_incremented = [nx1,ny1,nz1,nx2,ny2,nz2]; subscripts_this_incremented[wrt] += 1
	subscripts_other_incremented = [nx1,ny1,nz1,nx2,ny2,nz2]; subscripts_other_incremented[wrt_other] += 1
	subscripts_this_decremented = [nx1,ny1,nz1,nx2,ny2,nz2]; subscripts_this_decremented[wrt] -= 1
	# -2*(p1p2/(p1+p2))*(R^0_1,0,0,0,0,0 - R^0_0,0,0,1,0,0) + -2*(p1+p2)*(p1/(p1+p2)) * (p1/(p1+p2)*R^1_1,0,0,0,0,0 + p2/(p1+p2)*R^1_0,0,0,1,0,0)
	# along with the term from the x1^n, so + nR^0_(n-1),0,0,0,0,0
	return -2*(p1*p2/(p1+p2)) * (
		nuclear_attraction_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, *subscripts_this_incremented, n_boys, wrt_list_copy)
		- nuclear_attraction_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, *subscripts_other_incremented, n_boys, wrt_list_copy)
	) + -2*(p1+p2)*(wrt_p/(p1+p2)) * (
		wrt_p/(p1+p2) * nuclear_attraction_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, *subscripts_this_incremented, n_boys + 1, wrt_list_copy)
		+ wrt_p_other/(p1+p2) * nuclear_attraction_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, *subscripts_other_incremented, n_boys + 1, wrt_list_copy)
	) + (subscripts[wrt] * nuclear_attraction_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, *subscripts_this_decremented, n_boys, wrt_list_copy) if subscripts[wrt]>0 else 0)



# This function initiates the recursion by setting up a list of all the derivatives which need to be taken.
def nuclear_attraction_integral_hermite(x1,y1,z1,p1,lx1,ly1,lz1, x2,y2,z2,p2,lx2,ly2,lz2, xn, yn, zn):
	x1 -= xn; x2 -= xn
	y1 -= yn; y2 -= yn
	z1 -= zn; z2 -= zn
	wrt_list = []
	wrt_list += [0] * lx1
	wrt_list += [1] * ly1
	wrt_list += [2] * lz1
	wrt_list += [3] * lx2
	wrt_list += [4] * ly2
	wrt_list += [5] * lz2
	return nuclear_attraction_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, 0,0,0, 0,0,0, 0, wrt_list)







# For speed, precompute the Boys integrals and the gaussian overlap scaling constants when computing the ERI's
precomputed_boys_values = []
precomputed_A1 = 0
precomputed_A2 = 0
def precompute_ERI_values(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4, n_max):
	global precomputed_boys_values, precomputed_A1, precomputed_A2
	precomputed_A1 = math.exp(-(p1*p3/(p1+p3)) * ((x3-x1)**2 + (y3-y1)**2 + (z3-z1)**2))
	precomputed_A2 = math.exp(-(p2*p4/(p2+p4)) * ((x4-x2)**2 + (y4-y2)**2 + (z4-z2)**2))
	precomputed_boys_values = []
	Cx1 = (p1*x1+p3*x3) / (p1+p3)
	Cy1 = (p1*y1+p3*y3) / (p1+p3)
	Cz1 = (p1*z1+p3*z3) / (p1+p3)
	Cx2 = (p2*x2+p4*x4) / (p2+p4)
	Cy2 = (p2*y2+p4*y4) / (p2+p4)
	Cz2 = (p2*z2+p4*z4) / (p2+p4)
	B = (p1+p3)*(p2+p4)/(p1+p2+p3+p4)
	for n in range(n_max+1):
		precomputed_boys_values.append(boys(n, B*((Cx2-Cx1)**2 + (Cy2-Cy1)**2 + (Cz2-Cz1)**2)))






# Similar setup as the nuclear attraction integrals for the electron repulsion integrals

# Note: Remember to call precompute_ERI_values() for each ERI integral before calling this method!
def R_electron_repulsion(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4, nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nx4,ny4,nz4, n_boys):
	A1 = precomputed_A1#math.exp(-(p1*p3/(p1+p3)) * ((x3-x1)**2 + (y3-y1)**2 + (z3-z1)**2))
	A2 = precomputed_A2#math.exp(-(p2*p4/(p2+p4)) * ((x4-x2)**2 + (y4-y2)**2 + (z4-z2)**2))
	#Cx1 = (p1*x1+p3*x3) / (p1+p3)
	#Cy1 = (p1*y1+p3*y3) / (p1+p3)
	#Cz1 = (p1*z1+p3*z3) / (p1+p3)
	#Cx2 = (p2*x2+p4*x4) / (p2+p4)
	#Cy2 = (p2*y2+p4*y4) / (p2+p4)
	#Cz2 = (p2*z2+p4*z4) / (p2+p4)
	#B = (p1+p3)*(p2+p4)/(p1+p2+p3+p4)
	#return 2*math.pi**(5/2) / ((p1+p3)*(p2+p4)*(p1+p2+p3+p4)**0.5) * A1*A2 * boys(n_boys, B*((Cx2-Cx1)**2 + (Cy2-Cy1)**2 + (Cz2-Cz1)**2)) \
	return 2*math.pi**(5/2) / ((p1+p3)*(p2+p4)*(p1+p2+p3+p4)**0.5) * A1*A2 * precomputed_boys_values[n_boys] \
		* x1**nx1 * y1**ny1 * z1**nz1 \
		* x2**nx2 * y2**ny2 * z2**nz2 \
		* x3**nx3 * y3**ny3 * z3**nz3 \
		* x4**nx4 * y4**ny4 * z4**nz4


def electron_repulsion_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4, nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nx4,ny4,nz4, n_boys, wrt_list):
	wrt_list_copy = wrt_list.copy()
	# If there are no more derivatives to take, evaluate the R function
	if wrt_list_copy == []:
		return R_electron_repulsion(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4, nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nx4,ny4,nz4, n_boys)
	wrt = wrt_list_copy.pop()
	wrt_other = None
	wrt_opposite = None
	wrt_opposite_other = None
	wrt_p = None
	wrt_p_other = None
	wrt_p_opposite = None
	wrt_p_opposite_other = None
	if wrt < 3:
		wrt_other = wrt+3; wrt_opposite = wrt+6; wrt_opposite_other = wrt_other+6
		wrt_p = p1; wrt_p_other = p2; wrt_p_opposite = p3; wrt_p_opposite_other = p4
	elif wrt < 6:
		wrt_other = wrt-3; wrt_opposite = wrt+6; wrt_opposite_other = wrt_other+6
		wrt_p = p2; wrt_p_other = p1; wrt_p_opposite = p4; wrt_p_opposite_other = p3
	elif wrt < 9:
		wrt_other = wrt+3; wrt_opposite = wrt-6; wrt_opposite_other = wrt_other-6
		wrt_p = p3; wrt_p_other = p4; wrt_p_opposite = p1; wrt_p_opposite_other = p2
	else:
		wrt_other = wrt-3; wrt_opposite = wrt-6; wrt_opposite_other = wrt_other-6
		wrt_p = p4; wrt_p_other = p3; wrt_p_opposite = p2; wrt_p_opposite_other = p1
	subscripts = [nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nx4,ny4,nz4]
	subscripts_this_incremented = [nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nx4,ny4,nz4]; subscripts_this_incremented[wrt] += 1
	subscripts_other_incremented = [nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nx4,ny4,nz4]; subscripts_other_incremented[wrt_other] += 1
	subscripts_opposite_incremented = [nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nx4,ny4,nz4]; subscripts_opposite_incremented[wrt_opposite] += 1
	subscripts_opposite_other_incremented = [nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nx4,ny4,nz4]; subscripts_opposite_other_incremented[wrt_opposite_other] += 1
	subscripts_this_decremented = [nx1,ny1,nz1,nx2,ny2,nz2,nx3,ny3,nz3,nx4,ny4,nz4]; subscripts_this_decremented[wrt] -= 1
	# -2*(p1p3/(p1+p3))*(R^0_1,0,0,0,0,0,0,0,0,0,0,0 - R^0_0,0,0,0,0,0,0,1,0,0,0,0)
	#+ -2*(p1+p3)*(p2+p4)/(p1+p2+p3+p4)*(p1/(p1+p3)) * (p1/(p1+p3)*R^1_1,0,0,0,0,0,0,0,0,0,0,0 + p3/(p1+p3)*R^1_0,0,0,0,0,0,1,0,0,0,0,0 - p2/(p2+p4)*R^1_0,0,0,1,0,0,0,0,0,0,0,0 - p4/(p2+p4)*R^1_0,0,0,0,0,0,0,0,0,1,0,0)
	# along with the term from the x1^n, so + nR^0_(n-1),0,0,0,0,0,0,0,0,0,0,0
	result = -2*(wrt_p*wrt_p_opposite/(wrt_p+wrt_p_opposite)) * (
		electron_repulsion_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4, *subscripts_this_incremented, n_boys, wrt_list_copy)
		- electron_repulsion_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4, *subscripts_opposite_incremented, n_boys, wrt_list_copy)
	) + -2*(p1+p3)*(p2+p4)/(p1+p2+p3+p4)*(wrt_p/(wrt_p+wrt_p_opposite)) * (
		wrt_p/(wrt_p+wrt_p_opposite) \
			* electron_repulsion_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4, *subscripts_this_incremented, n_boys + 1, wrt_list_copy)
		+ wrt_p_opposite/(wrt_p+wrt_p_opposite) \
			* electron_repulsion_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4, *subscripts_opposite_incremented, n_boys + 1, wrt_list_copy)
		- wrt_p_other/(wrt_p_other+wrt_p_opposite_other) \
			* electron_repulsion_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4, *subscripts_other_incremented, n_boys + 1, wrt_list_copy)
		- wrt_p_opposite_other/(wrt_p_other+wrt_p_opposite_other) \
			* electron_repulsion_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4, *subscripts_opposite_other_incremented, n_boys + 1, wrt_list_copy)
	) + (subscripts[wrt] * electron_repulsion_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4, *subscripts_this_decremented, n_boys, wrt_list_copy) if subscripts[wrt]>0 else 0)
	return result



def electron_repulsion_integral_hermite(x1,y1,z1,p1,lx1,ly1,lz1, x2,y2,z2,p2,lx2,ly2,lz2, x3,y3,z3,p3,lx3,ly3,lz3, x4,y4,z4,p4,lx4,ly4,lz4):
	average_position_x = (x1+x2+x3+x4)/4
	average_position_y = (y1+y2+y3+y4)/4
	average_position_z = (z1+z2+z3+z4)/4
	# For some reason, the offset position was causing some ERI's to blow up to like -2.6e84 on contracted p orbitals, but only on the integrals between 3 of one p orbital exponent, and 1 of another. I don't know why.
	# This seems to fix it at least for now, but the recursion relations should probably be reworked sometime.
	x1 -= average_position_x; x2 -= average_position_x; x3 -= average_position_x; x4 -= average_position_x
	y1 -= average_position_y; y2 -= average_position_y; y3 -= average_position_y; y4 -= average_position_y
	z1 -= average_position_z; z2 -= average_position_z; z3 -= average_position_z; z4 -= average_position_z
	wrt_list = []
	wrt_list += [0] * lx1
	wrt_list += [1] * ly1
	wrt_list += [2] * lz1
	wrt_list += [3] * lx2
	wrt_list += [4] * ly2
	wrt_list += [5] * lz2
	wrt_list += [6] * lx3
	wrt_list += [7] * ly3
	wrt_list += [8] * lz3
	wrt_list += [9] * lx4
	wrt_list += [10] * ly4
	wrt_list += [11] * lz4
	precompute_ERI_values(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4, len(wrt_list))
	return electron_repulsion_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0, wrt_list)


