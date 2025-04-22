import os
import sys
import datetime as dt
from qcodes import initialise_or_create_database_at
import logging

# 設定記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_parent_folders_name(path, levels=2):
    """取得指定層數的父資料夾名稱"""
    parts = []
    current = path
    for _ in range(levels):
        current = os.path.dirname(current)
        name = os.path.basename(current)
        if name:
            parts.insert(0, name)
    return parts


def get_user_name(path):
    """從路徑中取得使用者名稱"""
    parts = path.split(os.sep)
    try:
        user_index = parts.index('user')
        if len(parts) > user_index + 1:
            return parts[user_index + 1]
    except ValueError:
        pass
    return None


def get_database_path(user_name):
    """取得使用者的資料庫路徑"""
    if not user_name:
        raise ValueError("無法確定使用者名稱")

    # 從目前路徑向上尋找 Qcodes 根目錄
    current = os.getcwd()
    while os.path.basename(current) != "Qcodes" and current != "/":
        current = os.path.dirname(current)

    if current == "/":
        raise ValueError("無法找到 Qcodes 根目錄")

    # 建立資料庫基礎路徑
    base_path = os.path.join(current, "user", user_name, "database")

    # 確保資料夾存在
    os.makedirs(base_path, exist_ok=True)
    return base_path


def init_database():
    try:
        # 取得使用者名稱
        user_name = get_user_name(os.getcwd())
        if not user_name:
            raise ValueError("無法確定使用者名稱")

        # 取得資料庫儲存路徑
        db_path = get_database_path(user_name)

        # 取得目前資料夾名稱與父資料夾名稱
        current_folder = os.path.basename(os.getcwd())
        parent_folders = get_parent_folders_name(os.getcwd(), 2)

        # 組合完整的樣本名稱
        sample_name = '_'.join(parent_folders + [current_folder])
        date = str(dt.date.today())
        db_name = f"{sample_name}_{date}"

        reuse = eval(sys.argv[1])

        # 搜尋現有資料庫檔案
        db_files = []
        for root, _, files in os.walk(db_path):
            for file in files:
                if file.endswith('.db') and 'Thumbs' not in file:
                    db_files.append(os.path.join(root, file))

        if reuse and db_files:
            db_files.sort()
            db_full_path = os.path.splitext(db_files[-1])[0] + '.db'
            
            # 檢查是否為有效的 SQLite 資料庫
            import sqlite3
            try:
                conn = sqlite3.connect(db_full_path)
                conn.close()
                print('使用現有資料庫檔案:')
            except sqlite3.DatabaseError:
                print('警告: 現有檔案不是有效的資料庫，將建立新檔案')
                reuse = False

        else:
            i = 1
            while os.path.exists(os.path.join(db_path, f"{db_name}_{i:02d}.db")):
                i += 1
            db_full_path = os.path.join(db_path, f"{db_name}_{i:02d}.db")
            print('建立新資料庫檔案:')

        logging.info(f"資料庫檔案路徑: {db_full_path}")
        initialise_or_create_database_at(db_full_path)

    except Exception as e:
        logging.error(f"錯誤: {str(e)}")


if __name__ == '__main__':
    init_database()
