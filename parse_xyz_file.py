from integral_matricies import Nucleus, charge_e

atom_charges = {
	"H": 1,
	"He": 2,
	"Li": 3,
	"Be": 4,
	"B": 5,
	"C": 6,
	"N": 7,
	"O": 8,
	"F" : 9,
	"Ne" : 10,
	"Na" : 11,
	"Mg" : 12,
	"Al" : 13,
	"Si" : 14,
	"P" : 15,
	"S" : 16,
	"Cl" : 17,
	"Ar" : 18,
	"K" : 19,
	"Ca" : 20,
}


def parse_xyz_file(filename):
	nuclei = []
	total_charge = 0
	with open(filename, "r") as file:
		for line in file.readlines():
			tokens = line.split()
			if len(tokens) < 4:
				continue
			symbol = tokens[0]
			if not symbol in atom_charges:
				print(f"Error: Atom symbol {symbol} not recognized")
				continue
			x = float(tokens[1]) * 1e-10
			y = float(tokens[2]) * 1e-10
			z = float(tokens[3]) * 1e-10
			nuclei.append(Nucleus((x,y,z), charge_e * atom_charges[symbol], symbol))
			total_charge += atom_charges[symbol]
	return nuclei, total_charge

