import os, sys
import datetime as dt
from qcodes import initialise_or_create_database_at

# Get the current working directory path
dirpath = os.getcwd()

# Get the current date
date = str(dt.date.today())

sample_name = os.path.basename(dirpath)
db_name = f"{sample_name}_{date}"

db_path = os.path.join(dirpath, sample_name)
db_full_path = os.path.join(db_path, db_name)

i = 1
while os.path.exists(f"{db_full_path}_{i:02d}.db"):
    i += 1

reuse = eval(sys.argv[1])

files = []
for root, dirs, filenames in os.walk(db_path):
    for filename in filenames:
        if filename.endswith('.db') and 'Thumbs' not in filename:
            files.append(os.path.join(root, filename))

files.sort()

if reuse and len(files) > 0:
    db_full_path = os.path.splitext(files[-1])[0] + '.db'
    print('Using existing database file:')
else:
    db_full_path = f"{db_full_path}_{i:02d}.db"
    print('Creating a new database file:')

print(db_full_path)
initialise_or_create_database_at(db_full_path)

'''
這段程式碼的主要目的是初始化或創建一個資料庫文件。首先，它從 os、sys 和 datetime 模組中導入必要的函數和類別，並從 qcodes 模組中導入 initialise_or_create_database_at 函數。

程式碼開始時，使用 os.getcwd() 獲取當前工作目錄的路徑，並使用 datetime.date.today() 獲取當前日期，將其轉換為字串格式。接著，使用 os.path.basename 獲取當前目錄的名稱，並將目錄名稱和日期組合成資料庫名稱 db_name。

接下來，程式碼構建了資料庫的完整路徑 db_full_path，並使用 while 迴圈檢查是否已存在相同名稱的資料庫文件。如果存在，則通過增加計數器 i 來生成新的文件名，直到找到一個不存在的文件名為止。

程式碼使用 eval(sys.argv[1]) 來獲取命令行參數，決定是否重用現有的資料庫文件。然後，使用 os.walk 遍歷資料庫目錄，收集所有以 .db 結尾且不包含 'Thumbs' 的文件，並將這些文件路徑存儲在 files 列表中。

如果 reuse 為真且 files 列表中有文件，則使用最新的資料庫文件，否則創建一個新的資料庫文件。最後，打印出選擇的資料庫文件路徑，並調用 initialise_or_create_database_at 函數來初始化或創建資料庫。
'''
