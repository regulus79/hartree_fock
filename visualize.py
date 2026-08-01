import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from helper_classes import *
from parse_gaussian_basis_set import parse_gaussian_basis_set
from parse_xyz_file import parse_xyz_file
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("data_file", default = "output.npz")
parser.add_argument("--quantile", type = float, default = 0.50)
parser.add_argument("--lattice_size", type = int, default = 40)
parser.add_argument("--buffer", type = float, default = 1.0)
parser.add_argument("--plot", type = str, choices = ["occupied", "homo_lumo"], default = "homo_lumo")
parser.add_argument("--extra_unoccipied", type = int, default = 0)
parser.add_argument("--num_frontier", type = int, default = 4)
args = parser.parse_args()



# For some reason, matplotlib does not like plotting 1e-10 scale meshes, so first we scale it up by 1e10 to make the numbers more reasonable
scaleup = 1e10

def plot_orbital(contracted_gaussians, coeffs, ax, quantile = 0.5, lattice_shape = (40, 40, 40), buffer = 1.0):
	allOrbitalPoses = []
	allOrbitalExponents = []
	for contracted_gaussian in contracted_gaussians:
		allOrbitalPoses += [hermite_gaussian.position for hermite_gaussian in contracted_gaussian.hermite_gaussians]
		allOrbitalExponents += [hermite_gaussian.exponent for hermite_gaussian in contracted_gaussian.hermite_gaussians]
	minBound = np.min(allOrbitalPoses, axis = 0)
	maxBound = np.max(allOrbitalPoses, axis = 0)
	lattice_buffer = np.ones(3) * 1/np.min(allOrbitalExponents)**0.5 * buffer
	minBound -= lattice_buffer
	maxBound += lattice_buffer

	psi = np.zeros(lattice_shape, dtype = "complex")
	coords_x = np.linspace(minBound[0], maxBound[0], lattice_shape[0])
	coords_y = np.linspace(minBound[1], maxBound[1], lattice_shape[1])
	coords_z = np.linspace(minBound[2], maxBound[2], lattice_shape[2])
	# Why is x and y flipped in the plot?
	xv, yv, zv = np.meshgrid(coords_x, coords_y, coords_z)
	for j, contracted_gaussian in enumerate(contracted_gaussians):
		contracted_coeff = coeffs[j]
		for i, cartesian_gaussian in enumerate(contracted_gaussian.cartesian_gaussians):
			cartesian_coeff = contracted_gaussian.cartesian_coefficients[i]
			exponent = cartesian_gaussian.exponent
			xPower = cartesian_gaussian.angular_momentum[0]
			yPower = cartesian_gaussian.angular_momentum[1]
			zPower = cartesian_gaussian.angular_momentum[2]
			psi += contracted_coeff * cartesian_coeff \
				* ((xv - cartesian_gaussian.position[0])**xPower) \
				* ((yv - cartesian_gaussian.position[1])**yPower) \
				* ((zv - cartesian_gaussian.position[2])**zPower) \
				* np.exp(-exponent * ((xv - cartesian_gaussian.position[0])**2 + (yv - cartesian_gaussian.position[1])**2 + (zv - cartesian_gaussian.position[2])**2))
				
	scale = maxBound - minBound

	ax.set_xlim3d(min(minBound[0] * scaleup, ax.get_xlim3d()[0]), max(maxBound[0] * scaleup, ax.get_xlim3d()[1]))
	ax.set_ylim3d(min(minBound[1] * scaleup, ax.get_ylim3d()[0]), max(maxBound[1] * scaleup, ax.get_ylim3d()[1]))
	ax.set_zlim3d(min(minBound[2] * scaleup, ax.get_zlim3d()[0]), max(maxBound[2] * scaleup, ax.get_zlim3d()[1]))
	ax.set_aspect("equal")
	plot_wavefunction(psi, scale, minBound, ax, quantile)


def plot_wavefunction(psi, scale, pos, ax, quantile = 0.5):
	prob_density = np.absolute(psi)**2
	prob_density /= np.sum(prob_density)
	level = np.quantile(prob_density, 1 - quantile, weights = prob_density, method = "inverted_cdf")
	verts, faces, normals, values = measure.marching_cubes(prob_density, level)
	facecolors = [colors.hsv_to_rgb((np.angle(psi[tuple(np.round(np.mean(verts[face], axis = 0)).astype(int))]) / (2*math.pi) + 0.5, 1, 1)) for face in faces]
	# Why must the verts be rearranged?
	verts = verts[:, [1, 0, 2]]
	verts *= scale / psi.shape[0] * scaleup
	verts += pos * scaleup
	mesh = Poly3DCollection(verts[faces], shade=True, facecolors = facecolors)
	ax.add_collection3d(mesh)

def setup_axes(title, plot_radius, rows = 1, cols = 1, index = 1):
	ax = plt.subplot(rows, cols, index, projection = "3d")
	ax.title.set_text(title)
	ax.set_xlim3d(-plot_radius, plot_radius)
	ax.set_ylim3d(-plot_radius, plot_radius)
	ax.set_zlim3d(-plot_radius, plot_radius)
	ax.set_aspect("equal")
	return ax


def plot_atom_positions(ax, nuclei):
	for nucleus in nuclei:
		ax.scatter(nucleus.position[0] * scaleup, nucleus.position[1] * scaleup, nucleus.position[2] * scaleup, color = "black")




data = np.load(args.data_file)

eigenvalues = data["orbital_energies"]
eigenvectors = data["orbital_basis_coefficients"]

nuclei, num_electrons = parse_xyz_file(str(data["xyz_file"]))
num_occupied_orbitals = num_electrons // 2

basis_set = parse_gaussian_basis_set(str(data["basis_set_file"]))

gaussians = []
for nucleus in nuclei:
	if not nucleus.symbol in basis_set:
		print(f"Error: Could not find element symbol {symbol} in basis set.")
	contracted_gaussians = basis_set[nucleus.symbol]
	for contracted_gaussian in contracted_gaussians:
		gaussians.append(ContractedGaussian(nucleus.position, contracted_gaussian.exponents, contracted_gaussian.angular_momenta, contracted_gaussian.cartesian_coefficients))


fig = plt.figure()
num_plots = eigenvalues.shape[0]
num_cols = math.ceil(math.sqrt(num_plots))
num_rows = math.ceil(num_plots / num_cols)
plot_index = 1
for i in np.argsort(eigenvalues):
	ax = setup_axes(f"Energy: {eigenvalues[i]}", 0.0001, num_rows, num_cols, plot_index)
	plot_orbital(gaussians, eigenvectors[:, i], ax, args.quantile, (args.lattice_size, args.lattice_size, args.lattice_size), args.buffer)
	plot_atom_positions(ax, nuclei)
	plot_index += 1
plt.show()
