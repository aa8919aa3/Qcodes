import sys
import datetime as dt
from pathlib import Path
import logging
import sqlite3
from qcodes import initialise_or_create_database_at

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_parent_folders_name(path: Path, levels: int = 2):
    path_obj = Path(path)
    return [path_obj.parents[i].name for i in reversed(range(levels)) if i < len(path_obj.parents)]


def get_user_name(path: Path):
    parts = path.parts
    try:
        user_index = parts.index('user')
        return parts[user_index + 1]
    except (ValueError, IndexError):
        return None


def get_qcodes_root(start_path: Path):
    current = start_path
    while current.name != "Qcodes_Albert" and current != current.parent:
        current = current.parent
    if current.name != "Qcodes_Albert":
        raise FileNotFoundError("無法從當前路徑往上找到 Qcodes_Albert 根目錄")
    return current


def get_database_path(user_name: str, qcodes_root: Path):
    base_path = qcodes_root / "user" / user_name / "database"
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def create_or_reuse_db(db_path: Path, db_name: str, reuse: bool = True) -> Path:
    existing_dbs = sorted(
        [f for f in db_path.glob("*.db") if 'Thumbs' not in f.name])
    if reuse and existing_dbs:
        try:
            sqlite3.connect(existing_dbs[-1]).close()
            logging.info("使用現有資料庫")
            return existing_dbs[-1]
        except sqlite3.DatabaseError:
            logging.warning("發現無效資料庫，建立新檔案")
    # 建立新檔案
    i = 1
    while (db_path / f"{db_name}_{i:02d}.db").exists():
        i += 1
    new_db = db_path / f"{db_name}_{i:02d}.db"
    logging.info("建立新資料庫")
    return new_db


def init_database(reuse: bool = True, base_path: Path = None, hide_logs: bool = True) -> Path:
    try:
        # 暫時降低日誌等級來隱藏升級訊息
        if hide_logs:
            qcodes_logger = logging.getLogger('qcodes')
            original_level = qcodes_logger.level
            qcodes_logger.setLevel(logging.ERROR)

        cwd = base_path if base_path else Path.cwd()
        user_name = get_user_name(cwd)
        if not user_name:
            raise ValueError("無法從路徑中推斷使用者名稱")

        qcodes_root = get_qcodes_root(cwd)
        db_path = get_database_path(user_name, qcodes_root)

        current_folder = cwd.name
        parent_folders = get_parent_folders_name(cwd, levels=2)
        sample_name = '_'.join(parent_folders + [current_folder])
        date_str = str(dt.date.today())
        db_name = f"{sample_name}_{date_str}"

        db_full_path = create_or_reuse_db(db_path, db_name, reuse)
        initialise_or_create_database_at(str(db_full_path))
        logging.info(f"初始化資料庫：{db_full_path}")
        print(f"目前使用資料庫：{db_full_path}")

        # 恢復原始日誌等級
        if hide_logs:
            qcodes_logger.setLevel(original_level)

        return db_full_path

    except Exception as e:
        logging.error(f"初始化失敗: {e}")
        raise


# CLI 使用
if __name__ == '__main__':
    reuse = eval(sys.argv[1]) if len(sys.argv) > 1 else True
    hide_logs = eval(sys.argv[2]) if len(sys.argv) > 2 else False
    init_database(reuse=reuse, hide_logs=hide_logs)
