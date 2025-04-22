# from qcodes_contrib_drivers.drivers.QuTech.IVVI import IVVI
# from qcodes.instrument_drivers.QuTech.IVVI import IVVI
from qcodes.instrument_drivers.tektronix.Keithley_6500 import Keithley_6500
# from Kei213_2 import K213
from qcodes.instrument_drivers.stanford_research.SR860 import SR860
# from qcodes.instrument_drivers.rohde_schwarz.SGS100A import RohdeSchwarzSGS100A
# from qcodes.instrument_drivers.stanford_research.SR830 import SR830
# from qcodes_contrib_drivers.drivers.BlueFors.BlueFors import BlueFors
from qcodes.instrument_drivers.Keithley.Keithley_2400 import Keithley2400
from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430, AMI430_3D
from qcodes.math_utils.field_vector import FieldVector
# from qcodes.instrument_drivers.yokogawa.GS200 import GS200

# keithley_1 = Keithley_6500('keithley_1', 'GPIB0::6::INSTR')
# station.add_component(keithley_1)

keithley_1 = Keithley_6500('keithley_1', 'GPIB0::6::INSTR')
station.add_component(keithley_1)

# K213 = K213('k213', 'GPIB0::9::INSTR')
# station.add_component(K213)

# keithley_24 = Keithley2400('keithley_24', 'GPIB0::22::INSTR')
# station.add_component(keithley_24)

# keithley_19 = Keithley2400('keithley_19', 'GPIB0::19::INSTR')
# station.add_component(keithley_19)

lockin_1 = SR860('lockin_1', 'GPIB0::5::INSTR')
station.add_component(lockin_1)

# lockin_1 = SR830('lockin_1', 'GPIB0::5::INSTR')
# station.add_component(lockin_1)

# SGS = RohdeSchwarzSGS100A('SGS', 'GPIB0::28::INSTR')
# station.add_component(SGS)

# folder_path = 'C:\\Users\\admin\\SynologyDrive\\09 Data\\Fridge log'
# bf = BlueFors('bf_fridge',
#               folder_path=folder_path,
#               channel_vacuum_can=1,
#               channel_pumping_line=2,
#               channel_compressor_outlet=3,
#               channel_compressor_inlet=4,
#               channel_mixture_tank=5,
#               channel_venting_line=6,
#               channel_50k_plate=1,
#               channel_4k_plate=2,
#               channel_magnet=3,
#               channel_still=6,
#               channel_mixing_chamber=5)

# lakeshore = Model_340("lakeshore", "GPIB0::12::INSTR")
# station.add_component(lakeshore)
'''
magnet_z = AMI430("z", address='169.254.68.235', port=7180)
magnet_y = AMI430("y", address='169.254.69.86', port=7180)
magnet_x = AMI430("x", address='169.254.200.32', port=7180)

field_limit = [  # If any of the field limit functions are satisfied we are in the safe zone.
   lambda x, y, z: x < 1 and y < 1 and z < 9
]

i3d = AMI430_3D(
   "AMI430-3D",
    magnet_x,
    magnet_y,
    magnet_z,
    field_limit=field_limit
)

station.add_component(magnet_x)
station.add_component(magnet_y)
station.add_component(magnet_z)
station.add_component(i3d)

# ivvi = IVVI('ivvi', 'ASRL4::INSTR')
# station.add_component(ivvi)

# yoko = GS200('yoko', 'USB0::0x0B21::0x0039::91T926460::INSTR')
# station.add_component(yoko)

'''