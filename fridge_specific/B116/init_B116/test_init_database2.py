import os
import unittest
from unittest.mock import patch
from datetime import date
from pathlib import Path
from init_database_v4 import get_user_name, get_database_path, init_database


class TestInitDatabase(unittest.TestCase):
    def setUp(self):
        """測試前準備"""
        self.test_path = Path("/Users/albert-mac/Code/GitHub/Qcodes/user/Albert/measurement/sample_test/test001/RT")
        # 確保測試目錄存在
        self.test_path.mkdir(parents=True, exist_ok=True)
        # 確保資料庫目錄存在
        self.db_dir = Path("/Users/albert-mac/Code/GitHub/Qcodes/user/Albert/database")
        self.db_dir.mkdir(parents=True, exist_ok=True)
        # 切換當前工作目錄
        os.chdir(self.test_path)

    def test_get_user_name(self):
        """測試取得使用者名稱"""
        user_name = get_user_name(str(self.test_path))
        self.assertEqual(user_name, "Albert")

    def test_get_database_path(self):
        """測試取得資料庫路徑"""
        db_path = get_database_path("Albert")
        expected_path = "/Users/albert-mac/Code/GitHub/Qcodes/user/Albert/database"
        self.assertEqual(db_path, expected_path)

    @patch('sys.argv', ['script.py', 'False'])
    def test_init_database_new(self):
        """測試建立新資料庫"""
        init_database()
        today = date.today()
        expected_db = f"sample_test_test001_RT_{today}_01.db"
        self.assertTrue((self.db_dir / expected_db).exists())

    @patch('sys.argv', ['script.py', 'True'])
    def test_init_database_reuse(self):
        """測試重用現有資料庫"""
        # 先建立一個測試用的資料庫檔案 (使用 QCoDeS 函式建立有效的資料庫)
        from qcodes import initialise_or_create_database_at
        
        today = date.today()
        test_db = f"sample_test_test001_RT_{today}_01.db"
        test_db_path = self.db_dir / test_db
        
        # 建立有效的 SQLite 資料庫
        initialise_or_create_database_at(str(test_db_path))

        init_database()
        # 驗證是否使用現有檔案
        self.assertTrue(test_db_path.exists())
        self.assertEqual(len(list(self.db_dir.glob('*.db'))), 1)

    def tearDown(self):
        """測試後清理"""
        # 清理測試過程中建立的資料庫檔案
        for db_file in self.db_dir.glob('*.db'):
            db_file.unlink()


if __name__ == '__main__':
    unittest.main()
