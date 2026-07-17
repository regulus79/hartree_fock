import math


def boys(T):
	if T == 0:
		return 1
	return math.erf(math.sqrt(T))/math.sqrt(T) * math.sqrt(math.pi)/2

def hermite(n, x, p):
	hermitePrev = 0
	hermiteCurrent = 1
	for i in range(n):
		hermiteNext = -2*p*x*hermiteCurrent - 2*i*p*hermitePrev
		#print(hermitePrev, hermiteCurrent, hermiteNext)
		hermitePrev = hermiteCurrent
		hermiteCurrent = hermiteNext
	return hermiteCurrent


def overlap_integral(x1,y1,z1,p1, x2,y2,z2,p2):
	return math.exp(-(p1*p2) / (p1+p2) * ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)) * math.sqrt(math.pi / (p1+p2))**3

def overlap_integral_derivative(x1,y1,z1,p1, x2,y2,z2,p2, lx,ly,lz):
	B = (p1*p2) / (p1+p2)
	deltax = x2 - x1
	deltay = y2 - y1
	deltaz = z2 - z1
	return (hermite(lx, deltax, B) * hermite(ly, deltay, B) * hermite(lz, deltaz, B)) * overlap_integral(x1,y1,z1,p1, x2,y2,z2,p2)


def nuclear_attraction_integral(x1,y1,z1,p1, x2,y2,z2,p2, xn, yn, zn):
	x1 -= xn; x2 -= xn
	y1 -= yn; y2 -= yn
	z1 -= zn; z2 -= zn
	A = math.exp(-(p1*p2) / (p1+p2) * ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2))
	B = p1+p2
	Cx = (p1*x1 + p2*x2) / (p1 + p2)
	Cy = (p1*y1 + p2*y2) / (p1 + p2)
	Cz = (p1*z1 + p2*z2) / (p1 + p2)
	C = math.sqrt(Cx**2 + Cy**2 + Cz**2)
	return 2 * math.pi/B * A * boys(B*C**2)

def electron_repulsion_integral(x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3, x4,y4,z4,p4):
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
	return 2*math.pi**(5/2) * A1*A2 / (B1*B2*math.sqrt(B1+B2)) * boys(T)


#print(electron_repulsion(0,0,0, 1, 0,0,0, 1, 0,0,0, 1, 0,0,0, 1))



