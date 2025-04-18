from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.utils.validators import Bool, Enum, Ints, Numbers

# --- 新增：初始化時移除 DummyInstrument 物件已存在的參數，避免重複註冊 ---
def _clear_all_parameters(inst):
    # 清除 DummyInstrument 物件所有已註冊參數
    for pname in list(inst.parameters.keys()):
        if pname not in ("IDN",):  # 保留 IDN 參數
            inst.remove_parameter(pname)

# SR860 鎖相放大器
sr860 = DummyInstrument(
    name="sr860",
    gates=["frequency", "amplitude", "phase", "sensitivity", "time_constant"],
)
_clear_all_parameters(sr860)
sr860.add_parameter("X", get_cmd=lambda: 0.0, unit="V")
sr860.add_parameter("Y", get_cmd=lambda: 0.0, unit="V")
sr860.add_parameter("R", get_cmd=lambda: 0.0, unit="V")
sr860.add_parameter("P", get_cmd=lambda: 0.0, unit="deg")

# Keithley 2000 多功能表
keithley2000 = DummyInstrument(name="keithley2000", gates=["mode"])
_clear_all_parameters(keithley2000)
keithley2000.add_parameter("amplitude", get_cmd=lambda: 0.0, unit="arb.unit")
keithley2000.add_parameter(
    "nplc",
    initial_value=1.0,
    set_cmd=lambda v: None,
    vals=Numbers(min_value=0.01, max_value=10),
)
keithley2000.add_parameter(
    "range", initial_value=10.0, set_cmd=lambda v: None, vals=Numbers()
)
keithley2000.add_parameter(
    "auto_range_enabled",
    initial_value=False,
    set_cmd=lambda v: None,
    vals=Bool(),
)
keithley2000.add_parameter(
    "digits",
    initial_value=6,
    set_cmd=lambda v: None,
    vals=Ints(min_value=4, max_value=7),
)
keithley2000.add_parameter(
    "averaging_type",
    initial_value="moving",
    set_cmd=lambda v: None,
    vals=Enum("moving", "repeat"),
)
keithley2000.add_parameter(
    "averaging_count",
    initial_value=10,
    set_cmd=lambda v: None,
    vals=Ints(min_value=1, max_value=100),
)
keithley2000.add_parameter("averaging_enabled", initial_value=False, set_cmd=lambda v: None, vals=Bool())

# Keithley 2400 源表
keithley2400 = DummyInstrument(name='keithley2400', gates=['volt', 'curr'])
_clear_all_parameters(keithley2400)
keithley2400.add_parameter('mode', initial_value='VOLT', set_cmd=lambda v: None, vals=Enum('VOLT', 'CURR'))
keithley2400.add_parameter('measured_volt', get_cmd=lambda: 0.0, unit='V')
keithley2400.add_parameter('measured_curr', get_cmd=lambda: 0.0, unit='A')
keithley2400.add_parameter('compliance_voltage', initial_value=10.0, set_cmd=lambda v: None, vals=Numbers(min_value=0.0, max_value=100.0), unit='V')
keithley2400.add_parameter('compliance_current', initial_value=1.0, set_cmd=lambda v: None, vals=Numbers(min_value=0.0, max_value=10.0), unit='A')

# Keithley 6500 多功能表
keithley6500 = DummyInstrument(name='keithley6500', gates=[])
_clear_all_parameters(keithley6500)
keithley6500.add_parameter('voltage_dc', get_cmd=lambda: 0.0, unit='V')
keithley6500.add_parameter('current_dc', get_cmd=lambda: 0.0, unit='A')
keithley6500.add_parameter('resistance', get_cmd=lambda: 0.0, unit='Ohm')
keithley6500.add_parameter('resistance_4w', get_cmd=lambda: 0.0, unit='Ohm')
keithley6500.add_parameter('nplc', initial_value=1.0, set_cmd=lambda v: None, vals=Numbers(min_value=0.01, max_value=10))
keithley6500.add_parameter('range', initial_value=10.0, set_cmd=lambda v: None, vals=Numbers())
keithley6500.add_parameter('auto_range_enabled', initial_value=False, set_cmd=lambda v: None, vals=Bool())
keithley6500.add_parameter('digits', initial_value=6, set_cmd=lambda v: None, vals=Ints(min_value=4, max_value=7))
keithley6500.add_parameter('averaging_type', initial_value='moving', set_cmd=lambda v: None, vals=Enum('moving', 'repeat'))
keithley6500.add_parameter('averaging_count', initial_value=10, set_cmd=lambda v: None, vals=Ints(min_value=1, max_value=100))
keithley6500.add_parameter('averaging_enabled', initial_value=False, set_cmd=lambda v: None, vals=Bool())

# Rohde & Schwarz SGS100A 信號發生器
sgs100a = DummyInstrument(name='sgs100a', gates=['frequency', 'power'])
_clear_all_parameters(sgs100a)
sgs100a.add_parameter('status', initial_value='off', set_cmd=lambda v: None, vals=Enum('on', 'off'))
sgs100a.add_parameter('phase', get_cmd=lambda: 0.0, unit='deg')
sgs100a.add_parameter('IQ_state', initial_value='off', set_cmd=lambda v: None, vals=Enum('on', 'off'))

# QuTech IVVI DAC 架
ivvi = DummyInstrument(name='ivvi', gates=['dac1', 'dac2', 'dac3', 'dac4'])
_clear_all_parameters(ivvi)
ivvi.add_parameter('dac_voltages', get_cmd=lambda: [0.0, 0.0, 0.0, 0.0], unit='mV')  # 模擬 DAC 電壓
ivvi.add_parameter('check_setpoints', initial_value=False, set_cmd=lambda v: None, vals=Bool(),
                   docstring='Whether to check if the setpoint matches the current DAC value.')

# AMI430 磁場電源
ami430 = DummyInstrument(name='ami430', gates=['field_setpoint', 'ramp_rate'])
_clear_all_parameters(ami430)
ami430.add_parameter('field', get_cmd=lambda: 0.0, unit='T')
ami430.add_parameter('current', get_cmd=lambda: 0.0, unit='A')
ami430.add_parameter('state', get_cmd=lambda: 'holding')
ami430.add_parameter('ramp_rate', initial_value=0.1, set_cmd=lambda v: None, vals=Numbers(min_value=0.0, max_value=1.0), unit='T/s')
ami430.add_parameter('persistent_mode', initial_value=False, set_cmd=lambda v: None, vals=Bool())

# AMI_3D 模擬三軸磁場控制
ami_3d = DummyInstrument(name='ami_3d', gates=['x', 'y', 'z'])
_clear_all_parameters(ami_3d)
ami_3d.add_parameter('x_field', get_cmd=lambda: 0.0, set_cmd=lambda x: None, unit='T')
ami_3d.add_parameter('y_field', get_cmd=lambda: 0.0, set_cmd=lambda y: None, unit='T')
ami_3d.add_parameter('z_field', get_cmd=lambda: 0.0, set_cmd=lambda z: None, unit='T')
ami_3d.add_parameter('vector_magnitude', get_cmd=lambda: 0.0, unit='T')
ami_3d.add_parameter('vector_angle', get_cmd=lambda: (0.0, 0.0), unit='deg')  # (theta, phi)
