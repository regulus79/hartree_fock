from integral_matricies import *


S_angular_momenta = [(0,0,0)]
P_angular_momenta = [(1,0,0), (0,1,0), (0,0,1)]
D_angular_momenta = [(2,0,0), (0,2,0), (0,0,2), (1,1,0), (1,0,1), (0,1,1)]


def parse_gaussian_basis_set(filename):
	basis_set = {}
	with open(filename, "r") as file:
		current_element = None
		current_element_basis_set = []
		current_shell_parameters = None
		for line in file.readlines():
			split_line = line.split()
			if line[0] == "!": # Skip Comments
				continue
			if len(split_line) == 0: # Skip empty lines
				continue

			if line[0] != " " and line[0] != "\t" and current_element != None and current_shell_parameters != None: # Either new element or new shell; either case, wrap up the current shell and add it to the list
				if "S" in current_shell_parameters["type"]:
					for angular_momentum in S_angular_momenta:
						current_element_basis_set.append(ContractedGaussian(
							(0,0,0),
							current_shell_parameters["contraction_exponents"],
							[angular_momentum] * current_shell_parameters["num_contraction_coeffs"],
							current_shell_parameters["S_contraction_coeffs"]
						))
				if "P" in current_shell_parameters["type"]:
					for angular_momentum in P_angular_momenta:
						current_element_basis_set.append(ContractedGaussian(
							(0,0,0),
							current_shell_parameters["contraction_exponents"],
							[angular_momentum] * current_shell_parameters["num_contraction_coeffs"],
							current_shell_parameters["P_contraction_coeffs"]
						))
				if "D" in current_shell_parameters["type"]:
					for angular_momentum in D_angular_momenta:
						current_element_basis_set.append(ContractedGaussian(
							(0,0,0),
							current_shell_parameters["contraction_exponents"],
							[angular_momentum] * current_shell_parameters["num_contraction_coeffs"],
							current_shell_parameters["D_contraction_coeffs"]
						))
				current_shell_parameters = None

			if split_line[0] == "****" and current_element != None: # New element, pack up the current element's basis and add it to the basis_set dictionary
				basis_set[current_element] = current_element_basis_set
				current_element = None
				current_element_basis_set = []
				continue
				
			if current_element == None: # Initialize the new element
				current_element = split_line[0]
				continue

			if current_element != None and (line[0] != " " and line[0] != "\t"): # Initialize shell
				current_shell_parameters = {
					"type": split_line[0],
					"num_contraction_coeffs": int(split_line[1]),
					"contraction_exponents": [],
					"S_contraction_coeffs": [],
					"P_contraction_coeffs": [],
					"D_contraction_coeffs": [],
				}
				continue
			
			if current_element != None and (line[0] == " " or line[0] == "\t"): # Parse cartesian gaussians
				current_shell_parameters["contraction_exponents"].append(float(split_line[0].replace("D","E")) / bohr_radius ** 2)
				contraction_coeffs = [float(value.replace("D","E")) for value in split_line[1:]]
				for i, shell_type in enumerate(current_shell_parameters["type"]):
					if shell_type == "S":
						current_shell_parameters["S_contraction_coeffs"].append(contraction_coeffs[i])
					if shell_type == "P":
						current_shell_parameters["P_contraction_coeffs"].append(contraction_coeffs[i])
					if shell_type == "D":
						current_shell_parameters["D_contraction_coeffs"].append(contraction_coeffs[i])
				continue

	return basis_set
