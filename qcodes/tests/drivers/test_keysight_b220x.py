import pytest
from pyvisa.errors import VisaIOError
from qcodes.instrument_drivers.Keysight.keysight_b220x import KeysightB220X


@pytest.fixture
def uut() -> KeysightB220X:
    try:
        uut = KeysightB220X('switch_matrix',
                            address='GPIB::22::INSTR')
    except (ValueError, VisaIOError):
        # Either there is no VISA lib installed or there was no real instrument found at the
        # specified address => use simulated instrument
        import qcodes.instrument.sims as sims
        path_to_yaml = sims.__file__.replace('__init__.py', 'keysight_b220x.yaml')

        uut = KeysightB220X('switch_matrix',
                            address='GPIB::1::INSTR',
                            visalib=path_to_yaml + '@sim'
                            )

    uut.get_status()
    uut.clear_status()
    uut.reset()

    yield uut

    uut.close()


def test_idn_command(uut):
    assert "AGILENT" in uut.IDN()['vendor']
    assert 0 == uut.get_status()


def test_connect(uut):
    uut.connect(2, 48)
    assert 0 == uut.get_status()


def test_disconnect_all(uut):
    uut.connect(2, 48)
    uut.disconnect_all()
    assert 0 == uut.get_status()


def test_disconnect(uut):
    uut.connect(2, 48)
    assert 0 == uut.get_status()
    uut.disconnect(2, 48)
    assert 0 == uut.get_status()
    uut.disconnect(3, 22)
    assert 0 == uut.get_status()


def test_connection_rule(uut):
    uut.connection_rule('single')
    assert 0 == uut.get_status()
    assert 'single' == uut.connection_rule()
    assert 0 == uut.get_status()

def test_connection_sequence(uut):
    assert 'bbm' == uut.connection_sequence()
    assert 0 == uut.get_status()
    uut.connection_sequence('mbb')
    assert 0 == uut.get_status()
    assert 'mbb' == uut.connection_sequence()


def test_bias_disable_all_outputs(uut):
    uut.bias_disable_all_outputs()
    assert 0 == uut.get_status()


def test_bias_disable_ouput(uut):
    uut.bias_disable_output(1)
    assert 0 == uut.get_status()


def test_bias_enable_all_outputs(uut):
    uut.bias_enable_all_outputs()
    assert 0 == uut.get_status()


def test_bias_enable_output(uut):
    uut.bias_enable_output(1)
    assert 0 == uut.get_status()


def test_bias_input_port(uut):
    assert 10 == uut.bias_input_port()
    uut.bias_input_port(9)
    assert 9 == uut.bias_input_port()
    assert 0 == uut.get_status()


def test_bias_mode(uut):
    uut.bias_mode(True)
    assert uut.bias_mode()
    assert 0 == uut.get_status()


def test_gnd_disable_all_outputs(uut):
    uut.gnd_disable_all_outputs()
    assert 0 == uut.get_status()


def test_gnd_disable_output(uut):
    uut.gnd_disable_output(1)
    assert 0 == uut.get_status()


def test_gnd_enable_all_outputs(uut):
    uut.gnd_enable_all_outputs()
    assert 0 == uut.get_status()


def test_gnd_enable_output(uut):
    uut.gnd_enable_output(1)
    assert 0 == uut.get_status()


def test_gnd_input_port(uut):
    assert 12 == uut.gnd_input_port()
    uut.gnd_input_port(5)
    assert 5 == uut.gnd_input_port()
    assert 0 == uut.get_status()


def test_gnd_mode(uut):
    assert not uut.gnd_mode()
    uut.gnd_mode(True)
    assert uut.gnd_mode()
    assert 0 == uut.get_status()


def test_ground_enabled_inputs(uut):
    uut.gnd_enabled_inputs()
    assert 0 == uut.get_status()

    uut.gnd_enabled_inputs([3])
    assert 0 == uut.get_status()
    assert [3] == uut.gnd_enabled_inputs()

    uut.gnd_enabled_inputs([5, 6, 7, 8])
    assert 0 == uut.get_status()
    assert [5, 6, 7, 8] == uut.gnd_enabled_inputs()


def test_couple_mode(uut):
    assert not uut.couple_mode()
    uut.couple_mode(True)
    assert uut.couple_mode()
    assert 0 == uut.get_status()


def test_couple_ports(uut):
    assert not uut.couple_ports()
    assert 0 == uut.get_status()

    uut.couple_ports([1])
    assert 0 == uut.get_status()
    assert [1] == uut.couple_ports()

    uut.couple_ports([1, 3, 5])
    assert 0 == uut.get_status()
    assert [1,3,5] == uut.couple_ports()

    with pytest.raises(ValueError):
        uut.couple_ports([2])
    with pytest.raises(ValueError):
        uut.couple_ports([2, 3])


def test_couple_port_autodetect(uut):
    uut.couple_port_autodetect()
    assert 0 == uut.get_status()


def test_get_error(uut):
    uut.get_error()
    assert 0 == uut.get_status()


def test_notebook():
    import qcodes as qc

    from qcodes.instrument_drivers.Keysight.keysight_b220x import KeysightB220X
    from pyvisa.errors import VisaIOError

    # Create a station to hold all the instruments

    station = qc.Station()

    # instantiate the Switch Matrix and add it to the station
    try:
        switch_matrix = KeysightB220X('switch_matrix',
                                      address='GPIB::22::INSTR')
        print("using physical instrument")
    except (ValueError, VisaIOError):
        # Either there is no VISA lib installed or there was no real instrument found at the
        # specified address => use simulated instrument
        import qcodes.instrument.sims as sims
        path_to_yaml = sims.__file__.replace('__init__.py', 'keysight_b220x.yaml')

        switch_matrix = KeysightB220X('switch_matrix',
                                      address='GPIB::1::INSTR',
                                      visalib=path_to_yaml + '@sim'
                                      )
        print("using simulated instrument")

    station.add_component(switch_matrix)

    switch_matrix.connect(2, 48)  # connect input 2 to output 48
    switch_matrix.connect(14, 3)  # connect input 14 to output 3
    switch_matrix.disconnect(14, 3)  # disconnect input 14 from output 3


