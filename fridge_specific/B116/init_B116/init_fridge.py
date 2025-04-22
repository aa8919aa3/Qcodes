import sys
import runpy
import os

import qcodes as qc

station = qc.Station()

fridge_name = sys.argv[1]

dirpath = os.getcwd()
code_path = '\\'.join(dirpath.split('\\')[0:-1])

sys.path.append(code_path+'\\drivers')

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

    exec(open('C:\\qcodes\\fridge_specific\\init_BF1.py').read())  
    
else:
    print('Wrong fridge name!')


