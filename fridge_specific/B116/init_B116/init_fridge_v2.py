import sys
import runpy
import os

import qcodes as qc

station = qc.Station()

fridge_name = sys.argv[1]

def find_qcodes_local_dir():
    dirpath = os.getcwd()
    while True:
        dirpath, folder_name = os.path.split(dirpath)
        if folder_name == 'QCoDeS_local':
            return os.path.join(dirpath, folder_name)
        if not folder_name:  # Reached the root directory
            return None

qcodes_local_dir = find_qcodes_local_dir()
if qcodes_local_dir:
    sys.path.append(qcodes_local_dir)
    drivers_path = os.path.join(qcodes_local_dir, 'drivers')
    sys.path.append(drivers_path)
    fridge_specific_path = os.path.join(qcodes_local_dir, 'fridge_specific')
    sys.path.append(fridge_specific_path)
    
    # 檢查 init_BF1_v2.py 是否存在於 fridge_specific 資料夾中
    init_BF1_v2_path = os.path.join(fridge_specific_path, 'init_BF1_v2.py')
    if os.path.isfile(init_BF1_v2_path):
        print(f"Found init_BF1_v2.py at: {init_BF1_v2_path}")
        # exec(open(f'{init_BF1_v2_path}').read())  
    else:
        print("init_BF1_v2.py not found in fridge_specific folder.")

if fridge_name == 'Janis':
    exec(open('../../../../../code/fridge_specific/init_Janis.py').read())
elif fridge_name == 'Gecko':
    exec(open('../../../../../code/fridge_specific/init_Gecko.py').read())
elif fridge_name == 'Fristi':
    exec(open('../../../../../code/fridge_specific/init_Fristi.py').read())
elif fridge_name == 'Cactus':
    exec(open('../../../../../code/fridge_specific/init_Cactus.py').read())
elif fridge_name == 'CF900':
    exec(open('../../../../../code/fridge_specific/init_CF900.py').read())
elif fridge_name == 'SG1':
    exec(open('../../../../../code/fridge_specific/init_SG1.py').read())
elif fridge_name == 'CF1400':
    exec(open('../../../../../code/fridge_specific/init_CF1400.py').read())
elif fridge_name == 'BF1':
#     exec(open('../fridge_specific/init_BF1.py').read())
    #exec(open(f'{init_BF1_path}').read()) 
    exec(open(f'{init_BF1_v2_path}').read()) 
    
    
else:
    print('Wrong fridge name!')


