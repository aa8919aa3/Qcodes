from qcodes import VisaInstrument
from qcodes.utils.helpers import create_on_off_val_mapping


class Gentec_Maestro(VisaInstrument):
    """
    Instrument driver for the Gentec Maestro powermeter.
    Args:
        name (str): Instrument name.
        address (str): ASRL address
        baud_rate (int): Baud rate for the connection.
    Attributes:
        model (str): Model identification.
        firmware_version (str): Firmware version.
        
    """

    def __init__(self, name, address, baud_rate=115200, **kwargs):

        super().__init__(name, address, **kwargs)

        # set baud rate
        self.visa_handle.baud_rate = baud_rate

        # get instrument information
        self.model = self.visa_handle.query('*VER').split()[0]
        self.firmware_version = self.visa_handle.query('*VER').split()[2]

        # add parameters
        self.add_parameter('power',
                           get_cmd='*CVU',
                           get_parser=float,
                           label='power',
                           unit='W')

        def wavelength_get_parser(ans):
            return int(ans[4:])
        self.add_parameter('wavelength',
                           get_cmd='*GWL',
                           set_cmd='*PWC{0:0>5}',
                           get_parser=wavelength_get_parser,
                           label='wavelength',
                           unit='nm')

        def zero_offset_get_parser(ans):
            return int(ans[5:])
        self.add_parameter('zero_offset_status',
                           get_cmd='*GZO',
                           get_parser=zero_offset_get_parser,
                           val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
                           label='zero offset status')

        # add functions
        self.add_function('set_zero_offset',
                          call_cmd='*SOU')

        self.add_function('clear_zero_offset',
                          call_cmd='*COU')

        # print connect message
        self.connect_message(idn_param='IDN')

    # get methods
    def get_idn(self):
        return {'vendor': 'Gentec', 'model': self.model,
                'firmware': str(self.firmware_version)}
