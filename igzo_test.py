from pathlib import Path
import numpy as np
import math
import matplotlib.pyplot as plt
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import SubCircuitFactory
from PySpice.Spice.Parser import SpiceParser
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from PySpice.Spice.Netlist import Circuit





directory_path = Path(__file__).resolve().parent
netlist_path = directory_path.joinpath('sibal.cir')
parser = SpiceParser(path=str(netlist_path))
circuit = parser.build_circuit()

print("netlist: \n\n", circuit)


simulator = circuit.simulator(temperature=25, nominal_temperature=25, simulator='ngspice-subprocess')

#analysis = simulator.transient(step_time=1@u_ps, end_time=100@u_us)
analysis = simulator.dc(Vgate=slice(-3, 3, .1))

ax.plot(analysis['gate'], u_mA(-analysis.VVd))


ax.legend('NMOS characteristic')
ax.grid()
ax.set_xlabel('Vgs [V]')
ax.set_ylabel('Id [mA]')

plt.tight_layout()
plt.savefig('test')



#vml=analysis.Vml1
#vml=-analysis.Vml1

#print('vml=',float(vml))