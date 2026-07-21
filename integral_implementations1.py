import math


def boys(n, T):
	if n == 0:
		if T == 0:
			return 1
		return math.erf(math.sqrt(T))/math.sqrt(T) * math.sqrt(math.pi)/2
	else:
		if T == 0:
			return 1 / (2*n + 1)
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


def nuclear_attraction_integral_hermiteOLD(x1,y1,z1,p1,lx1,ly1,lz1, x2,y2,z2,p2,lx2,ly2,lz2, xn, yn, zn):
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


# Need to take derivatives of
# e^-kc * F(bc^2)
# A*B
# A'*B + A*B'*2bc
# A''*B + A'*B' + A'*B'*2bc + A*B''*2bc + A*B'*2b
# A'''*B + A''*B' + A''*B' + A'*B'' + A''*B'*2bc + A'*B''*2bc + A'*B'*2b + A'*B''*2bc + A*B'''*2bc + A*B''*2b + A'*B'*2b + A*B''*2b

# B
# -B'2bc
# B''4bbcc - B'2b
# -B'''8bbbccc + B''12bbc

initial_term = {
	"scaling": [1],
	"x1": [],
	"y1": [],
	"z1": [],
	"x2": [],
	"y2": [],
	"z2": [],
	"n": 0 # Boys function order
}
# As terms appear, they will look like this:
#{
#	"scaling": [1],
#	"x1": [-2*k],
#	"y1": [-2*B],
#	"z1": [],
#	"x2": [],
#	"y2": [],
#	"z2": [],
#	"n": 1
#}

# Helper function
def copy_term(term):
	return {
		"scaling": term["scaling"].copy(),
		"x1": term["x1"].copy(),
		"y1": term["y1"].copy(),
		"z1": term["z1"].copy(),
		"x2": term["x2"].copy(),
		"y2": term["y2"].copy(),
		"z2": term["z2"].copy(),
		"n": term["n"],
	}

def nuclear_attraction_integral_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, wrt_list = [], terms = [initial_term]):
	if wrt_list == []:
		total = 0
		for term in terms:
			temp = math.prod(term["scaling"])
			for factor in term["x1"]:
				temp *= factor * x1
			for factor in term["x2"]:
				temp *= factor * x2
			for factor in term["y1"]:
				temp *= factor * y1
			for factor in term["y2"]:
				temp *= factor * y2
			for factor in term["z1"]:
				temp *= factor * z1
			for factor in term["z2"]:
				temp *= factor * z2
			temp *= math.exp(-(p1*p2) / (p1+p2) * ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2))
			temp *= boys(term["n"], (p1+p2) * (((p1*x1 + p2*x2) / (p1 + p2))**2 + ((p1*y1 + p2*y2) / (p1 + p2))**2 + ((p1*z1 + p2*z2) / (p1 + p2))**2))
			total += temp
		return 2 * math.pi / (p1+p2) * total

	wrt = wrt_list.pop()
	wrt_other = wrt[0] + ("2" if wrt[-1] == "1" else "1")
	wrt_p = p1 if wrt[-1] == "1" else p2
	wrt_other_p = p2 if wrt[-1] == "1" else p1
	new_terms = []
	for term in terms:
		for i,factor in enumerate(term[wrt]):
			new_term = copy_term(term)
			new_term[wrt].pop(i)
			new_term["scaling"].append(factor)
			new_terms.append(new_term)
		# The scaling term brings down a (-p1p2/(p1+p2) * 2(r2-r1)) term in front
		# which can be split up into (-p1p2/(p1+p2) * 2(r2)) and (-p1p2/(p1+p2) * 2(-r1))
		new_term_scaling1 = copy_term(term)
		new_term_scaling1[wrt].append(-2 * (p1*p2) / (p1+p2))
		new_term_scaling2 = copy_term(term)
		new_term_scaling2[wrt_other].append(-2 * (p1*p2) / (p1+p2) * -1)
		new_terms.append(new_term_scaling1)
		new_terms.append(new_term_scaling2)
		# The boys term brings down a -(p1+p2) * 2*(p1*r1 + p2*r2)/(p1+p2) which by chain rule also times p1/(p1+p2) or vice versa p2/(p1+p2), and increments the boys n order
		# The term simplifies to -2*(p1*r1 + p2*r2) * p1/(p1+p2)
		# Which can be split into -2*(p1*r1) * p1/(p1+p2) and 2*(p2*r2) * p1/(p1+p2)
		new_term_boys1 = copy_term(term)
		new_term_boys1[wrt].append(-2*(wrt_p) * (wrt_p)/(p1+p2))
		new_term_boys2 = copy_term(term)
		new_term_boys2[wrt_other].append(-2*(wrt_other_p) * (wrt_p)/(p1+p2))
		new_term_boys1["n"] += 1
		new_term_boys2["n"] += 1
		new_terms.append(new_term_boys1)
		new_terms.append(new_term_boys2)
	#print("Remaining:", wrt_list)
	#for term in new_terms:
	#	print(term)
	return nuclear_attraction_integral_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, wrt_list, new_terms)


#print(nuclear_attraction_integral_recurrance(0.1,0.2,0.3, 2,  0.6,0.4,0.5, 3,  0,0,0, [], [initial_term]))
#print(nuclear_attraction_integral_recurrance(0.1,0.2,0.3, 2,  0.6,0.4,0.5, 3,  0,0,0, ["x2"], [initial_term]))
#print(nuclear_attraction_integral_recurrance(0.1,0.2,0.3, 2,  0.6,0.4,0.5, 3,  0,0,0, ["x2", "x2"], [initial_term]))

#print(nuclear_attraction_integral_recurrance(3, 2, 0.3, 0.2, 0.1, [], [initial_term]))
#print(nuclear_attraction_integral_recurrance(3, 2, 0.3, 0.2, 0.1, ["x"], [initial_term]))
#print(nuclear_attraction_integral_recurrance(3, 2, 0.3, 0.2, 0.1, ["x","x"], [initial_term]))




def nuclear_attraction_integral_hermite(x1,y1,z1,p1,lx1,ly1,lz1, x2,y2,z2,p2,lx2,ly2,lz2, xn, yn, zn):
	x1 -= xn; x2 -= xn
	y1 -= yn; y2 -= yn
	z1 -= zn; z2 -= zn
	wrt_list = []
	wrt_list += ["x1"] * lx1
	wrt_list += ["y1"] * ly1
	wrt_list += ["z1"] * lz1
	wrt_list += ["x2"] * lx2
	wrt_list += ["y2"] * ly2
	wrt_list += ["z2"] * lz2
	return nuclear_attraction_integral_recurrance(x1,y1,z1,p1, x2,y2,z2,p2, wrt_list, [initial_term])

#print(nuclear_attraction_integral_primative(*(0.1,0.2,0.3), 2, *(0.4,0.5,0.6), 3, *(0,0,0)))
#print("new",nuclear_attraction_integral_hermite(*(0.1,0.2,0.3), 2, *(0,0,0), *(0.4,0.5,0.6), 3, *(0,0,0), *(0,0,0)))
#print("old",nuclear_attraction_integral_hermiteOLD(*(0.1,0.2,0.3), 2, *(0,0,0), *(0.4,0.5,0.6), 3, *(0,0,0), *(0,0,0)))
#print("new",nuclear_attraction_integral_hermite(*(0.1,0.2,0.3), 2, *(1,0,0), *(0.4,0.5,0.6), 3, *(0,0,0), *(0,0,0)))
#print("old",nuclear_attraction_integral_hermiteOLD(*(0.1,0.2,0.3), 2, *(1,0,0), *(0.4,0.5,0.6), 3, *(0,0,0), *(0,0,0)))
#print("new",nuclear_attraction_integral_hermite(*(0.1,0.2,0.3), 2, *(0,0,0), *(0.4,0.5,0.6), 3, *(1,0,0), *(0,0,0)))
#print("old",nuclear_attraction_integral_hermiteOLD(*(0.1,0.2,0.3), 2, *(0,0,0), *(0.4,0.5,0.6), 3, *(1,0,0), *(0,0,0)))
#print("new",nuclear_attraction_integral_hermite(*(0.1,0.2,0.3), 2, *(0,0,0), *(0.4,0.5,0.6), 3, *(2,0,0), *(0,0,0)))
#print("new",nuclear_attraction_integral_hermite(*(0.1,0.2,0.3), 2, *(0,0,0), *(0.4,0.5,0.6), 3, *(0,0,0), *(0,0,0)))
#print("new",nuclear_attraction_integral_hermite(*(0.1,0.2,0.3), 2, *(0,0,0), *(0.4,0.5,0.6), 3, *(1,0,0), *(0,0,0)))
#print("new",nuclear_attraction_integral_hermite(*(0.1,0.2,0.3), 2, *(1,0,0), *(0.4,0.5,0.6), 3, *(0,0,0), *(0,0,0)))
#print("new",nuclear_attraction_integral_hermite(*(0.1,0.2,0.3), 2, *(1,0,0), *(0.4,0.5,0.6), 3, *(1,0,0), *(0,0,0)))
#print("old",nuclear_attraction_integral_hermiteOLD(*(0.1,0.2,0.3), 2, *(1,0,0), *(0.4,0.5,0.6), 3, *(1,0,0), *(0,0,0)))
#print("new",nuclear_attraction_integral_hermite(*(0.1,0.2,0.3), 2, *(1,0,0), *(0.4,0.5,0.6), 3, *(0,1,0), *(0,0,0)))
#print("old",nuclear_attraction_integral_hermiteOLD(*(0.1,0.2,0.3), 2, *(1,0,0), *(0.4,0.5,0.6), 3, *(0,1,0), *(0,0,0)))
#print("new",nuclear_attraction_integral_hermite(*(0.1,0.2,0.3), 2, *(0,0,1), *(0.4,0.5,0.6), 3, *(1,0,0), *(0,0,0)))
#print("old",nuclear_attraction_integral_hermiteOLD(*(0.1,0.2,0.3), 2, *(0,0,1), *(0.4,0.5,0.6), 3, *(1,0,0), *(0,0,0)))
#print("new",nuclear_attraction_integral_hermite(*(0.1,0.2,0.3), 2, *(0,1,0), *(0.4,0.5,0.6), 3, *(0,1,0), *(0,0,0)))
#print("old",nuclear_attraction_integral_hermiteOLD(*(0.1,0.2,0.3), 2, *(0,1,0), *(0.4,0.5,0.6), 3, *(0,1,0), *(0,0,0)))