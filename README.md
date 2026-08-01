
# Hartree-Fock

Python script to compute the energy of a molecule using the Hartree-Fock method.

## How to use

Run `hartreefock.py` and provide as arguments:
- Path to molecular structure `.xyz` file
- Path to gaussian basis set `.gbs` file
- Path to output numpy file to save results (example: `output.npz`)

Once the output data (including energy, basis set coefficients, density matrices, etc.) is saved, the orbitals can be visualized by running `visualize.py` with the path to the output file.

### Example
```
python hartreefock.py example_xyz_files/water.xyz basis_sets/sto-3g.gbs output.npz 
```
Output:
```
Computing Overlap Integrals... 49/49
Computing Kinetic Integrals... 49/49
Computing Potential Integrals... 49/49
Computing Electron Repulsion Integrals... 2401/2401
Initial Energy: -73.23210275415009
=== SCF Iterations ===
Iteration 1: Energy = -74.94541956563081, Difference = -1.7133168114807233
Iteration 2: Energy = -74.9623269971965, Difference = -0.016907431565684305
Iteration 3: Energy = -74.96307960645184, Difference = -0.0007526092553433728
Iteration 4: Energy = -74.96317626474594, Difference = -9.665829409755133e-05
Iteration 5: Energy = -74.96319279323644, Difference = -1.652849050515215e-05
Iteration 6: Energy = -74.96319580889174, Difference = -3.015655295257602e-06
Iteration 7: Energy = -74.9631963703214, Difference = -5.614296583189571e-07
Iteration 8: Energy = -74.96319647549835, Difference = -1.0517695159251161e-07
Iteration 9: Energy = -74.96319649524008, Difference = -1.974173358121334e-08
Iteration 10: Energy = -74.96319649894775, Difference = -3.7076688386150636e-09
SCF Converged.
=== Final Energy ===
One-Electron Energy: -122.35635180737198
Two-Electron Energy: 38.204580048226575
Nuclei Repulsion Energy: 9.188575260197648
Total Energy: -74.96319649894775
=== Orbital Energies (Fock Matrix Eigenvalues) ===
[-20.24255683  -1.26863342  -0.61666678  -0.45415009  -0.39158874
   0.60581441   0.73994082]
```

Visualizing the orbitals:
```
python visualize.py output.npz
```

