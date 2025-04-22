import sys
from pathlib import Path
import os


def get_init_database_v4_path():
    """
    尋找並取得 init_database_v4.py 檔案的路徑

    函式會嘗試從以下位置尋找檔案：
    1. 使用相對路徑從目前工作目錄向上查找
    2. 嘗試在 Qcodes 根目錄的標準位置尋找
    3. 全域搜尋 Qcodes 目錄結構

    Returns:
        Path: init_database_v4.py 的路徑物件（若找到）
        None: 若找不到檔案
    """
    # 1. 從當前目錄開始尋找 Qcodes 根目錄
    current_path = Path.cwd()
    qcodes_root = None

    # 向上最多尋找 10 層目錄
    for _ in range(10):
        if current_path.name == "Qcodes":
            qcodes_root = current_path
            break

        # 檢查是否達到檔案系統根目錄
        if current_path == current_path.parent:
            break

        current_path = current_path.parent

    # 如果找不到，嘗試使用路徑字串解析
    if qcodes_root is None:
        path_str = str(Path.cwd())
        if "Qcodes" in path_str:
            qcodes_root = Path(path_str.split("Qcodes")[0] + "Qcodes")

    # 2. 檢查標準位置
    possible_paths = []

    if qcodes_root:
        possible_paths.extend([
            qcodes_root / "fridge_specific" / "B116" / "init_B116" / "init_database_v4.py",
            qcodes_root / "user" / "Albert" / "fridge_specific" /
            "B116" / "init_B116" / "init_database_v4.py"
        ])

    # 3. 嘗試從使用者主目錄尋找
    user_home = Path.home()
    if "albert-mac" in str(user_home):
        possible_paths.append(Path(
            "/Users/albert-mac/Code/GitHub/Qcodes/fridge_specific/B116/init_B116/init_database_v4.py"))

    # 檢查每個可能的路徑
    for path in possible_paths:
        if path.exists():
            print(f"成功找到 init_database_v4.py: {path}")
            return path

    # 如果還找不到，使用 os.walk 進行較深層的搜尋
    if qcodes_root:
        print("進行深層搜尋，這可能需要一些時間...")
        for root, dirs, files in os.walk(qcodes_root):
            if "init_database_v4.py" in files:
                path = Path(root) / "init_database_v4.py"
                print(f"搜尋成功找到 init_database_v4.py: {path}")
                return path

    print("找不到 init_database_v4.py 檔案，請確認檔案位置")
    return None


init_database_path = get_init_database_v4_path()
