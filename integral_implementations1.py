import math


def boys(n, T):
	if n == 0:
		if T == 0:
			return 1
		return math.erf(math.sqrt(T))/math.sqrt(T) * math.sqrt(math.pi)/2
	else:
		return (n - 1/2) / T * boys(n - 1, T) - math.exp(-T)/(2*T)


def hermite(n, x, p):
	hermitePrev = 0
	hermiteCurrent = 1
	for i in range(n):
		hermiteNext = -2*p*x*hermiteCurrent - 2*i*p*hermitePrev
		#print(hermitePrev, hermiteCurrent, hermiteNext)
		hermitePrev = hermiteCurrent
		hermiteCurrent = hermiteNext
	return hermiteCurrent


def overlap_integral_primative(x1,y1,z1,p1, x2,y2,z2,p2):
	return math.exp(-(p1*p2) / (p1+p2) * ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)) * math.sqrt(math.pi / (p1+p2))**3

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


def nuclear_attraction_integral_primative_hermite(x1,y1,z1,p1,lx1,ly1,lz1, x2,y2,z2,p2,lx2,ly2,lz2, xn, yn, zn):
	delta = max(math.sqrt(p1+p2)**-1, math.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)) * 0.01
	if lx1+ly1+lz1+lx2+ly2+lz2 == 0:
		return nuclear_attraction_integral_primative(x1,y1,z1,p1, x2,y2,z2,p2, xn, yn, zn)
	elif lx1+ly1+lz1 == 1 and lx2+ly2+lz2 == 1:
		return \
			((nuclear_attraction_integral_primative(x1+delta*lx1,y1+delta*ly1,z1+delta*lz1,p1, x2+delta*lx2,y2+delta*ly2,z2+delta*lz2,p2, xn, yn, zn) \
				- nuclear_attraction_integral_primative(x1,y1,z1,p1, x2+delta*lx2,y2+delta*ly2,z2+delta*lz2,p2, xn, yn, zn))/delta \
			- (nuclear_attraction_integral_primative(x1+delta*lx1,y1+delta*ly1,z1+delta*lz1,p1, x2,y2,z2,p2, xn, yn, zn) \
				- nuclear_attraction_integral_primative(x1,y1,z1,p1, x2,y2,z2,p2, xn, yn, zn))/delta)/delta


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


def electron_repulsion_integral_primative_hermite(x1,y1,z1,p1,lx1,ly1,lz1, x2,y2,z2,p2,lx2,ly2,lz2, x3,y3,z3,p3,lx3,ly3,lz3, x4,y4,z4,p4,lx4,ly4,lz4):
	delta = max(math.sqrt(p1+p2)**-1, math.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)) * 0.01
	if lx1+ly1+lz1+lx2+ly2+lz2 == 0:
		return electron_repulsion_integral_primative(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4)
	elif lx1+ly1+lz1 == 1 and lx2+ly2+lz2 == 1:
		x1+delta*lx1,y1+delta*ly1,z1+delta*lz1
		x2+delta*lx2,y2+delta*ly2,z2+delta*lz2
		x3+delta*lx3,y3+delta*ly3,z3+delta*lz3
		x4+delta*lx4,y4+delta*ly4,z4+delta*lz4
		return \
			((((electron_repulsion_integral_primative(x1+delta*lx1,y1+delta*ly1,z1+delta*lz1,p1, x2+delta*lx2,y2+delta*ly2,z2+delta*lz2,p2, x3+delta*lx3,y3+delta*ly3,z3+delta*lz3,p3, x4+delta*lx4,y4+delta*ly4,z4+delta*lz4,p4) \
				- electron_repulsion_integral_primative(x1,y1,z1,p1, x2+delta*lx2,y2+delta*ly2,z2+delta*lz2,p2, x3+delta*lx3,y3+delta*ly3,z3+delta*lz3,p3, x4+delta*lx4,y4+delta*ly4,z4+delta*lz4,p4))/delta \
			- (electron_repulsion_integral_primative(x1+delta*lx1,y1+delta*ly1,z1+delta*lz1,p1, x2,y2,z2,p2, x3+delta*lx3,y3+delta*ly3,z3+delta*lz3,p3, x4+delta*lx4,y4+delta*ly4,z4+delta*lz4,p4) \
				- electron_repulsion_integral_primative(x1,y1,z1,p1, x2,y2,z2,p2, x3+delta*lx3,y3+delta*ly3,z3+delta*lz3,p3, x4+delta*lx4,y4+delta*ly4,z4+delta*lz4,p4))/delta)/delta \
			- ((electron_repulsion_integral_primative(x1+delta*lx1,y1+delta*ly1,z1+delta*lz1,p1, x2+delta*lx2,y2+delta*ly2,z2+delta*lz2,p2, x3,y3,z3,p3, x4+delta*lx4,y4+delta*ly4,z4+delta*lz4,p4) \
				- electron_repulsion_integral_primative(x1,y1,z1,p1, x2+delta*lx2,y2+delta*ly2,z2+delta*lz2,p2, x3,y3,z3,p3, x4+delta*lx4,y4+delta*ly4,z4+delta*lz4,p4))/delta \
			- (electron_repulsion_integral_primative(x1+delta*lx1,y1+delta*ly1,z1+delta*lz1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4+delta*lx4,y4+delta*ly4,z4+delta*lz4,p4) \
				- electron_repulsion_integral_primative(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4+delta*lx4,y4+delta*ly4,z4+delta*lz4,p4))/delta)/delta)/delta \
			- (((electron_repulsion_integral_primative(x1+delta*lx1,y1+delta*ly1,z1+delta*lz1,p1, x2+delta*lx2,y2+delta*ly2,z2+delta*lz2,p2, x3+delta*lx3,y3+delta*ly3,z3+delta*lz3,p3, x4,y4,z4,p4) \
				- electron_repulsion_integral_primative(x1,y1,z1,p1, x2+delta*lx2,y2+delta*ly2,z2+delta*lz2,p2, x3+delta*lx3,y3+delta*ly3,z3+delta*lz3,p3, x4,y4,z4,p4))/delta \
			- (electron_repulsion_integral_primative(x1+delta*lx1,y1+delta*ly1,z1+delta*lz1,p1, x2,y2,z2,p2, x3+delta*lx3,y3+delta*ly3,z3+delta*lz3,p3, x4,y4,z4,p4) \
				- electron_repulsion_integral_primative(x1,y1,z1,p1, x2,y2,z2,p2, x3+delta*lx3,y3+delta*ly3,z3+delta*lz3,p3, x4,y4,z4,p4))/delta)/delta \
			- ((electron_repulsion_integral_primative(x1+delta*lx1,y1+delta*ly1,z1+delta*lz1,p1, x2+delta*lx2,y2+delta*ly2,z2+delta*lz2,p2, x3,y3,z3,p3, x4,y4,z4,p4) \
				- electron_repulsion_integral_primative(x1,y1,z1,p1, x2+delta*lx2,y2+delta*ly2,z2+delta*lz2,p2, x3,y3,z3,p3, x4,y4,z4,p4))/delta \
			- (electron_repulsion_integral_primative(x1+delta*lx1,y1+delta*ly1,z1+delta*lz1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4) \
				- electron_repulsion_integral_primative(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4))/delta)/delta)/delta)/delta



