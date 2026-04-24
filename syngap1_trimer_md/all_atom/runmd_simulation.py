from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

# 1. Load the AlphaFold PDB
print("Loading PDB...")
pdb = PDBFile('./data/pdb/syngap1_trimer.pdb')
cif_file = PDBxFile('./data/pdb/syngap1_trimer.cif')

# 2. Specify the CHARMM36 force field and water model
print("Loading CHARMM force field...")
forcefield = ForceField('charmm36.xml', 'charmm36/water.xml')

# 3. Solvate the system using Modeller
print("Adding solvent and neutralizing ions...")
#modeller = Modeller(pdb.topology, pdb.positions)
modeller = Modeller(cif_file.topology, cif_file.positions)

# Add missing hydrogen atoms
modeller.addHydrogens(forcefield)

# Adds a TIP3P water box with a 1 nm padding around the protein and 0.15 M NaCl
modeller.addSolvent(forcefield, padding=1.0*nanometer, model='tip3p', ionicStrength=0.15*molar)

# 4. Create the system
print("Creating system...")
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME,
        nonbondedCutoff=1*nanometer, constraints=HBonds)

# 5. Add a Barostat for constant pressure (NPT ensemble)
# This is crucial for solvated systems so the water box can equilibrate to the correct density
system.addForce(MonteCarloBarostat(1*bar, 300*kelvin))

# 6. Set up the Integrator
# We use a 2 fs (0.002 ps) time step which is standard for simulations with HBonds constrained
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)

# 7. Initialize the Simulation with CUDA
print("Setting up CUDA platform...")

# Request the CUDA platform
platform = Platform.getPlatformByName('CUDA')

# Optional but highly recommended: Set precision to 'mixed' 
# This gives you the speed of single-precision with the accuracy of double-precision where it matters.
# You can also specify a specific GPU if you have multiple (e.g., 'DeviceIndex': '0')
properties = {'CudaPrecision': 'mixed'}

# Create the simulation object, passing in the platform and properties
simulation = Simulation(modeller.topology, system, integrator, platform, properties)
simulation.context.setPositions(modeller.positions)

# 8. Minimize the energy to resolve steric clashes from the AlphaFold model and added water
print("Minimizing energy...")
simulation.minimizeEnergy()

# 9. Assign initial velocities based on temperature
simulation.context.setVelocitiesToTemperature(300*kelvin)

# 10. Set up Reporters
# 100 ns = 100,000 ps. With a 0.002 ps step, that is 50,000,000 steps.
# We will save a frame every 50,000 steps (every 100 ps) to yield 1,000 total frames in the DCD.
report_interval = 50000

simulation.reporters.append(DCDReporter('output.dcd', report_interval))
simulation.reporters.append(StateDataReporter(stdout, report_interval, step=True,
        potentialEnergy=True, temperature=True, volume=True, speed=True))

# 11. Run the Production Simulation (100 ns)
total_steps = 50000000
print(f"Starting simulation for {total_steps} steps (100 ns)...")
simulation.step(total_steps)
print("Simulation complete!")
