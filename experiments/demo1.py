from data_loader import suzuki, aryl_amination
from edbo.utils import Data
from edbo.bro import BO

# Build search spaces for reactions 1 and 2 with DFT encoded components

reaction1 = Data(suzuki(electrophile='dft',
                        nucleophile='dft',
                        base='dft',
                        ligand='boltzmann-dft',
                        solvent='dft'))

reaction2a = Data(aryl_amination(aryl_halide='dft',
                                 additive='dft',
                                 base='dft',
                                 ligand='Pd(0)-dft',
                                 subset=1))

reaction2b = Data(aryl_amination(aryl_halide='dft',
                                 additive='dft',
                                 base='dft',
                                 ligand='Pd(0)-dft',
                                 subset=2))

reaction2c = Data(aryl_amination(aryl_halide='dft',
                                 additive='dft',
                                 base='dft',
                                 ligand='Pd(0)-dft',
                                 subset=3))

reaction2d = Data(aryl_amination(aryl_halide='dft',
                                 additive='dft',
                                 base='dft',
                                 ligand='Pd(0)-dft',
                                 subset=4))

reaction2e = Data(aryl_amination(aryl_halide='dft',
                                 additive='dft',
                                 base='dft',
                                 ligand='Pd(0)-dft',
                                 subset=5))

# Preprocess data

for reaction in [reaction1, reaction2a, reaction2b, reaction2c, reaction2d, reaction2e]:
    # Remove nun-numeric and singular columns
    reaction.clean()

    # Drop columns with unimportant information
    reaction.drop(['entry', 'vibration', 'correlation', 'Rydberg', 'correction',
                   'atom_number', 'E-M_angle', 'MEAN', 'STDEV'])

    # Standardize
    reaction.standardize(scaler='minmax')

    # Drop highly correlated features
    reaction.uncorrelated(threshold=0.95)

    # Encoded reaction
    reaction.data.head()

def main():


    # Instantiate edbo.BO
    bo = BO(exindex=reaction2b.data,  # Experiment index to look up results from
            domain=reaction2b.data.drop('yield', axis=1),  # Reaction space
            batch_size=3,  # Choose 3 experiments on each iteraiton
            acquisition_function='MACE',  # Use expected improvement
            fast_comp=True)  # Speed up the simulations using gpytorch's fast computation features

    # Run simulation
    bo.simulate(iterations=9, seed=0)

    # Plot convergence
    bo.plot_convergence()