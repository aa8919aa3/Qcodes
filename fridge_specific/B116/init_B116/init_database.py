import os, sys
import datetime as dt
from qcodes import initialise_or_create_database_at

dirpath = os.getcwd()
sample_name = dirpath.split('\\')[-2]
wafer_name = dirpath.split('\\')[-3]

date = str(dt.date.today())

db_name = f"{wafer_name}_{sample_name}_{date}"
db_path = dirpath.rsplit('\\', 1)[0] + '\\data'
db_full_path = db_path + '\\' + db_name

i = 1
while os.path.exists(db_full_path + '_%02d.db' %i):
    i += 1

reuse = eval(sys.argv[1])

files = []
for r, d, f in os.walk(db_path):
    for file in f:
        if ('.db' in file) and not ('Thumbs' in file):
            files.append(os.path.join(r, file))

files.sort()

if reuse and len(files) > 0:
    db_full_path = files[-1].rsplit('.',1)[0] + '.db'
    print('Using existing database file:')
else:
    db_full_path = db_full_path + '_%02d.db' %i
    print('Creating a new database file:')

print(db_full_path)
initialise_or_create_database_at(db_full_path)