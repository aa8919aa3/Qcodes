import pyvisa
import socket
import qcodes as qc
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Callable, Optional, Any

# 匯入儀器驅動程式
from qcodes.instrument_drivers.stanford_research.SR860 import SR860
from qcodes.instrument_drivers.Keithley.Keithley_2000 import Keithley2000
from qcodes.instrument_drivers.Keithley.Keithley_2400 import Keithley2400
from qcodes.instrument_drivers.tektronix.Keithley_6500 import Keithley_6500
from qcodes.instrument_drivers.rohde_schwarz.SGS100A import RohdeSchwarzSGS100A
from qcodes.instrument_drivers.american_magnetics import AMIModel430
from qcodes_contrib_drivers.drivers.QuTech.IVVI import IVVI

# 設定記錄器
log = logging.getLogger(__name__)

# 儀器識別和建立函式
INSTRUMENT_IDENTIFIERS = {
    "KEITHLEY.*MODEL DMM6500": (Keithley_6500, "DMM6500"),
    "KEITHLEY.*MODEL 2000": (Keithley2000, "K2000"),
    "KEITHLEY.*MODEL 2400": (Keithley2400, "K2400"),
    "KEITHLEY.*MODEL 2440": (Keithley2400, "K2440"),
    "SR860": (SR860, "SR860"),
    "SMB100A": (RohdeSchwarzSGS100A, "SMB100A")
}


def load_config(config_path: str = "instrument_config.yaml") -> Dict:
    """載入儀器配置檔"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        log.warning(f"配置檔 {config_path} 找不到，使用預設配置")
        return {
            "local_ip": "169.254.115.159",
            "magnet_ips": ["169.254.115.1", "169.254.115.2", "169.254.115.3"],
            "magnet_port": 7180,
            "ivvi_address": "ASRL3::INSTR",
            "field_limits": {"x": 1, "y": 1, "z": 9}
        }


def initialize_station() -> qc.Station:
    """初始化並回傳 QCoDeS 站點"""
    station = qc.Station()
    return station


def cleanup_station(station: qc.Station) -> None:
    """清理現有儀器連線"""
    for name, instrument in list(station.components.items()):
        log.info(f"Closing instrument {name}")
        try:
            instrument.close()
            station.remove_component(name)
        except Exception as e:
            log.error(f"Error closing instrument {name}: {e}")


def scan_and_add_instruments(station: qc.Station) -> None:
    """掃描可用儀器並將其新增到站點"""
    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    log.info(f"Found {len(resources)} VISA resources")

    for resource in resources:
        try:
            my_device = rm.open_resource(resource)
            my_device.timeout = 5000
            idn_string = my_device.query('*IDN?').strip()
            log.info(
                f"Device: {resource}\n"
                f"IDN: {idn_string}"
            )

            # 尋找匹配的儀器驅動程式
            for (
                identifier, (driver_class, default_name)
            ) in INSTRUMENT_IDENTIFIERS.items():
                import re
                if re.search(identifier, idn_string):
                    instrument = driver_class(default_name, resource)
                    station.add_component(instrument)
                    log.info(f"Added {default_name} ({resource}) to station")
                    break
            else:
                log.info(f"Unrecognized instrument: {idn_string}")

        except Exception as e:
            log.error(f"Error connecting to {resource}: {e}")


def add_ivvi(station: qc.Station, address: str) -> Optional[IVVI]:
    """新增 IVVI 控制器到站點"""
    try:
        ivvi = IVVI('ivvi', address)
        station.add_component(ivvi)
        log.info(f"Added IVVI at {address} to station")
        return ivvi
    except Exception as e:
        log.error(f"Error connecting to IVVI at {address}: {e}")
        return None


def check_connection(ip: str, port: int, local_ip: str) -> bool:
    """檢查是否可連接指定 IP 和連接埠"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.bind((local_ip, 0))  # 0 表示自動選擇可用連接埠
            s.connect((ip, port))
        log.info(f"Successfully connected to {ip}:{port}")
        return True
    except Exception as e:
        log.error(f"Cannot connect to {ip}:{port}: {e}")
        return False


def add_magnets(
    station: qc.Station, config: Dict
) -> Optional[AMIModel430.AMIModel430_3D]:
    """新增磁鐵控制器到站點"""
    magnet_ips = config.get("magnet_ips", [])
    magnet_port = config.get("magnet_port", 7180)
    local_ip = config.get("local_ip", "169.254.115.159")
    field_limits = config.get("field_limits", {"x": 1, "y": 1, "z": 9})

    # 檢查連接
    ping_results = [
        check_connection(ip, magnet_port, local_ip) for ip in magnet_ips
    ]

    if not all(ping_results) or len(magnet_ips) != 3:
        log.error(
            "Cannot connect to all magnets, field control initialization failed")
        return None

    try:
        magnets = [
            AMIModel430.AMIModel430(name, address=ip, port=magnet_port)
            for name, ip in zip("xyz", magnet_ips)
        ]

        for magnet in magnets:
            log.info(
                f"{magnet.name}, IP: {magnet._address}, "
                f"Port: {magnet._port}"
            )

        magnet_x, magnet_y, magnet_z = magnets

        # 設定磁場限制函式
        def field_constraint(x, y, z):
            return (x < field_limits["x"] and
                    y < field_limits["y"] and
                    z < field_limits["z"])

        field_limit = [field_constraint]

        i3d = AMIModel430.AMIModel430_3D(
            "AMI430_3D", *magnets, field_limit=field_limit
        )

        for magnet in magnets + [i3d]:
            station.add_component(magnet)
            log.info(f"Added {magnet.name} to station")

        return i3d
    except Exception as e:
        log.error(f"Error initializing magnets: {e}")
        return None


def main() -> qc.Station:
    """主要執行函式"""
    # 設定 QCoDeS 記錄器
    qc.logger.start_logger()
    log.info("Starting instrument initialization")

    # 載入配置
    config = load_config()

    # 初始化站點
    station = initialize_station()

    # 清理現有儀器
    cleanup_station(station)

    # 掃描和新增儀器
    scan_and_add_instruments(station)

    # 新增 IVVI
    add_ivvi(station, config.get("ivvi_address", "ASRL3::INSTR"))

    # 新增磁鐵
    add_magnets(station, config)

    # 顯示站點內容
    log.info("Station instrument list:")
    for name, component in station.components.items():
        log.info(f"- {name}: {type(component).__name__}")

    return station


if __name__ == "__main__":
    station = main()
