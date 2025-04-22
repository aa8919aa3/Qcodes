import pyvisa
import socket
import qcodes as qc
from pprint import pprint
from qcodes.instrument_drivers.stanford_research.SR860 import SR860
from qcodes.instrument_drivers.Keithley.Keithley_2000 import Keithley2000
from qcodes.instrument_drivers.Keithley.Keithley_2400 import Keithley2400
from qcodes.instrument_drivers.tektronix.Keithley_6500 import Keithley_6500
from qcodes.instrument_drivers.rohde_schwarz.SGS100A import RohdeSchwarzSGS100A
from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430, AMI430_3D
from qcodes_contrib_drivers.drivers.QuTech.IVVI import IVVI

# 初始化站點
station = qc.Station()

# 清理現有的儀器
rm = pyvisa.ResourceManager()
resources = rm.list_resources()
for name, instrument in list(station.components.items()):
    instrument.close()
    station.remove_component(name)

# 掃描並添加儀器
for resource in resources:
    try:
        my_device = rm.open_resource(resource)
        my_device.timeout = 5000
        idn_string = my_device.query('*IDN?')
        print(f"Device: {resource}\nIDN: {idn_string}")
        
        if "KEITHLEY" in idn_string and "MODEL DMM6500" in idn_string:
            DMM6500 = Keithley_6500('DMM6500', resource)
            station.add_component(DMM6500)
            print(f"Added Keithley DMM6500 at {resource} to the station.")

        elif "KEITHLEY" in idn_string and "MODEL 2000" in idn_string:
            K2000 = Keithley2000('K2000', resource)
            station.add_component(K2000)
            print(f"Added Keithley 2000 at {resource} to the station.")
            
        elif "KEITHLEY" in idn_string and "MODEL 2400" in idn_string:
            K2400 = Keithley2400('K2400', resource)
            station.add_component(K2400)
        
        elif "KEITHLEY" in idn_string and "MODEL 2440" in idn_string:
            K2440 = Keithley2400('K2440', resource)
            station.add_component(K2440)
            
        elif "SR860" in idn_string:
            SR860_1 = SR860('SR860', resource)
            station.add_component(SR860_1)
            print(f"Added SR860_1 at {resource} to the station.")
            
        elif "SMB100A" in idn_string:
            SMB100A = RohdeSchwarzSGS100A('SMB100A', resource)
            station.add_component(SMB100A)
            print(f"Added SGS100A at {resource} to the station.")
        # 檢查 ASRL3::INSTR 是否存在於資源清單中
        # elif 'ASRL3::INSTR' in resources:
        #     try:
        #         # 嘗試連接 ASRL3::INSTR
        #         ivvi_device = rm.open_resource('ASRL3::INSTR')
        #         ivvi_device.timeout = 10000
        #         idn_response = ivvi_device.query('*IDN?')  # 嘗試發送 IDN 指令
        #         print(f"IVVI Response: {idn_response}")
                
        #         # 如果 IVVI 有正確回應，則添加到 station
        #         ivvi = IVVI('ivvi', 'ASRL3::INSTR')
        #         station.add_component(ivvi)
        #         print("Added IVVI at ASRL3::INSTR to the station.")
                
        #     except Exception as e:
        #         print(f"Error connecting to IVVI at ASRL3::INSTR: {e}")
        # else:
        #     print("ASRL3::INSTR not found in resource list.")
        
    except Exception as e:
        print(f"Error connecting to {resource}: {e}")

ivvi = IVVI('ivvi', 'ASRL3::INSTR')
station.add_component(ivvi)
print("Added IVVI at ASRL3::INSTR to the station.")
# 指定本地IP地址
local_ip = '169.254.115.159'

def ping_address(ip, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((local_ip, 0))  # 0表示自動選擇一個可用的本地端口
            s.connect((ip, port))
        return True
    except Exception as e:
        print(f"Failed to ping {ip}: {e}")
        return False

# 進行 ping 測試
addresses = ['169.254.115.1', '169.254.115.2', '169.254.115.3']
ping_results = [ping_address(ip, 7180) for ip in addresses]

if all(ping_results):
    magnets = [AMI430(name, address=ip, port=7180) for name, ip in zip("xyz", addresses)]
    for magnet in magnets:
        print(f"{magnet.name}, IP: {magnet._address}, Port: {magnet._port}")
    
    magnet_x, magnet_y, magnet_z = magnets

    field_limit = [lambda x, y, z: x < 1 and y < 1 and z < 9]

    i3d = AMI430_3D("AMI430_3D", *magnets, field_limit=field_limit)

    for magnet in magnets + [i3d]:
        station.add_component(magnet)
        print(f"Added {magnet.name} to the station.")
else:
    print("One or more pings failed, cannot establish connection to all magnets.")

# 打印站點列表
print('\nStation list:')
pprint(station.components)