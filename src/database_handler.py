import logging
import pickle
import os
from PyQt5.QtSql import QSqlDatabase, QSqlQuery, QSqlTableModel, QSqlError
from PyQt5.QtCore import QByteArray, QDateTime, Qt
from datetime import datetime
import re

logger = logging.getLogger(__name__)
# Thiết lập mức độ logging cao hơn để loại bỏ thông báo không cần thiết
logger.setLevel(logging.WARNING)

class DatabaseHandler:
    # Constants
    STUDENT_ID_PATTERN = r"\d{8}"
    EMAIL_PATTERN = r"[\w.+-]+@(gmail\.com|hus\.edu\.vn)"
    
    def __init__(self, db_path="attendance.db", connection_name="attendance"):
        """Khởi tạo kết nối cơ sở dữ liệu SQLite."""
        try:
            # Sử dụng connection_name để tránh trùng lặp kết nối
            self.connection_name = connection_name
            self.db = QSqlDatabase.addDatabase("QSQLITE", connection_name)
            self.db.setDatabaseName(db_path)
            if not self.db.open():
                logger.error(f"Cannot open database: {self.db.lastError().text()}")
                raise Exception("Cannot open database")
            self.create_tables()
            self.ensure_created_at_for_existing_users()
        except Exception as e:
            logger.error(f"Exception in DatabaseHandler init: {e}")
            raise

    def create_tables(self):
        """Tạo các bảng users và attendance nếu chưa tồn tại."""
        query = QSqlQuery(self.db)
        
        # Tạo bảng users
        if not query.exec_("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                student_id TEXT NOT NULL UNIQUE,
                class TEXT,
                email TEXT,
                embedding BLOB,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """):
            logger.error(f"Failed to create users table: {query.lastError().text()}")
            raise Exception("Failed to create users table")
        
        # Kiểm tra schema users: nếu thiếu cột embedding
        columns = self._get_table_columns("users")
        if "embedding" not in columns:
            if not query.exec_("ALTER TABLE users ADD COLUMN embedding BLOB"):
                logger.error(f"Failed to add embedding column: {query.lastError().text()}")
        
        # Kiểm tra schema attendance, nếu thiếu student_id thì xóa để tạo lại
        columns = self._get_table_columns("attendance")
        if "student_id" not in columns and len(columns) > 0:
            if not query.exec_("DROP TABLE IF EXISTS attendance"):
                logger.error(f"Failed to drop old attendance table: {query.lastError().text()}")
        
        # Tạo bảng attendance
        if not query.exec_("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                name TEXT NOT NULL,
                class TEXT,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL,
                FOREIGN KEY(student_id) REFERENCES users(student_id)
            )
        """):
            logger.error(f"Failed to create attendance table: {query.lastError().text()}")
            raise Exception("Failed to create attendance table")

    def _get_table_columns(self, table_name):
        """Trả về danh sách tên cột trong bảng."""
        columns = []
        query = QSqlQuery(self.db)
        if query.exec_(f"PRAGMA table_info({table_name})"):
            while query.next():
                columns.append(query.value(1))  # Cột thứ 2 là tên cột
        return columns

    def ensure_created_at_for_existing_users(self):
        """Cập nhật thời gian đăng ký cho các user cũ chưa có created_at."""
        try:
            # Kiểm tra xem cột created_at có tồn tại trong bảng users không
            columns = self._get_table_columns("users")
            if "created_at" not in columns:
                # Thêm cột created_at nếu chưa tồn tại
                # Sử dụng cách an toàn hơn để thêm cột với giá trị mặc định
                query = QSqlQuery(self.db)
                try:
                    # Thử thêm cột với DEFAULT
                    if not query.exec_("ALTER TABLE users ADD COLUMN created_at TEXT DEFAULT (datetime('now', 'localtime'))"):
                        # Nếu lỗi, thử thêm cột không có DEFAULT
                        if not query.exec_("ALTER TABLE users ADD COLUMN created_at TEXT"):
                            logger.error(f"Error adding created_at column: {query.lastError().text()}")
                            return
                        # Sau đó cập nhật giá trị
                        if not query.exec_("UPDATE users SET created_at = datetime('now', 'localtime')"):
                            logger.error(f"Error setting initial created_at values: {query.lastError().text()}")
                            return
                    logger.info("Added created_at column to users table")
                except Exception as e:
                    logger.error(f"Error adding created_at column: {e}")
                    return
            
            # Cập nhật các user có created_at NULL hoặc rỗng
            query = QSqlQuery(self.db)
            update_sql = "UPDATE users SET created_at = datetime('now', 'localtime') WHERE created_at IS NULL OR created_at = ''"
            if not query.exec_(update_sql):
                logger.error(f"Error updating missing created_at: {query.lastError().text()}")
        except Exception as e:
            logger.error(f"Error updating missing created_at: {e}")

    def get_all_classes(self):
        """Trả về danh sách tên lớp duy nhất từ bảng users (bỏ qua NULL/rỗng)."""
        try:
            classes = []
            query = QSqlQuery(self.db)
            if query.exec_("SELECT DISTINCT class FROM users WHERE class IS NOT NULL AND class != ''"):
                while query.next():
                    classes.append(query.value(0))
            return classes
        except Exception as e:
            logger.error(f"Exception in get_all_classes: {e}")
            return []

    def validate_student_id(self, student_id):
        """Xác thực mã sinh viên (8 chữ số)."""
        return bool(re.fullmatch(self.STUDENT_ID_PATTERN, student_id))

    def validate_email(self, email):
        """Xác thực email (gmail.com hoặc hus.edu.vn)."""
        return bool(re.fullmatch(self.EMAIL_PATTERN, email))

    def get_class_by_student_id(self, student_id):
        """
        Trả về tên lớp (class) của sinh viên dựa trên student_id.
        Nếu không tìm thấy, trả về None.
        """
        try:
            query = QSqlQuery(self.db)
            query.prepare("SELECT class FROM users WHERE student_id = :student_id")
            query.bindValue(":student_id", student_id)
            if query.exec_() and query.next():
                return query.value(0)
            else:
                return None
        except Exception as e:
            logger.error(f"Exception in get_class_by_student_id: {e}")
            return None


    def add_user(self, name, student_id, class_name, email):
        """Thêm người dùng mới vào bảng users."""
        try:
            # Kiểm tra dữ liệu đầu vào
            if not name or not student_id:
                logger.error("Name and student_id cannot be empty")
                return None
                
            # Mã sinh viên phải là 8 số
            if not self.validate_student_id(student_id):
                logger.error("Student ID must be exactly 8 digits")
                return 'INVALID_STUDENT_ID'
                
            # Email phải đúng định dạng
            if email and not self.validate_email(email):
                logger.error("Email must be @gmail.com or @hus.edu.vn")
                return 'INVALID_EMAIL'
                
            # Kiểm tra student_id đã tồn tại
            query = QSqlQuery(self.db)
            query.prepare("SELECT COUNT(*) FROM users WHERE student_id = :student_id")
            query.bindValue(":student_id", student_id)
            if query.exec_() and query.next() and query.value(0) > 0:
                logger.error(f"Student ID {student_id} already exists")
                return 'EXISTS'
                
            # Thêm người dùng
            self.db.transaction()
            query.prepare("""
                INSERT INTO users (name, student_id, class, email)
                VALUES (:name, :student_id, :class_name, :email)
            """)
            query.bindValue(":name", name)
            query.bindValue(":student_id", student_id)
            query.bindValue(":class_name", class_name)
            query.bindValue(":email", email)
            
            if not query.exec_():
                err = query.lastError().text()
                logger.error(f"Error adding user: {err}")
                self.db.rollback()
                # Nếu lỗi do trùng mã sinh viên (UNIQUE constraint)
                if 'UNIQUE constraint failed' in err or 'UNIQUE' in err:
                    return 'EXISTS'
                return None
                
            self.db.commit()
            return query.lastInsertId()
        except Exception as e:
            self.db.rollback()
            logger.error(f"Exception in add_user: {e}")
            return None

    def update_user_embedding(self, student_id, embedding):
        """Cập nhật embedding khuôn mặt cho người dùng."""
        try:
            if not student_id or embedding is None:
                logger.error("student_id or embedding cannot be empty")
                return False
                
            if isinstance(embedding, str):
                logger.error(f"Embedding for {student_id} is str, not allowed. Refusing to save.")
                return False
                
            # Chuyển đổi embedding thành QByteArray
            blob_data = pickle.dumps(embedding)
            byte_array = QByteArray(blob_data)
            
            self.db.transaction()
            query = QSqlQuery(self.db)
            query.prepare("UPDATE users SET embedding = ? WHERE student_id = ?")
            query.bindValue(0, byte_array)
            query.bindValue(1, student_id)
            
            if not query.exec_():
                logger.error(f"Error updating embedding: {query.lastError().text()}")
                self.db.rollback()
                return False
                
            self.db.commit()
            logger.info(f"Updated embedding for student_id={student_id}")
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Exception in update_user_embedding: {e}")
            return False

    def update_user(self, user_id, name, student_id, class_name, email):
        """Cập nhật thông tin người dùng."""
        try:
            if not user_id or not name or not student_id:
                logger.error("User_id, name, and student_id cannot be empty")
                return False
                
            # Kiểm tra định dạng student_id
            if not self.validate_student_id(student_id):
                logger.error("Student ID must be exactly 8 digits")
                return False
                
            # Kiểm tra định dạng email nếu có
            if email and not self.validate_email(email):
                logger.error("Email must be @gmail.com or @hus.edu.vn")
                return False
                
            self.db.transaction()
            query = QSqlQuery(self.db)
            query.prepare("""
                UPDATE users
                SET name = :name, student_id = :student_id, class = :class_name, email = :email
                WHERE id = :id
            """)
            query.bindValue(":name", name)
            query.bindValue(":student_id", student_id)
            query.bindValue(":class_name", class_name)
            query.bindValue(":email", email)
            query.bindValue(":id", user_id)
            
            if not query.exec_():
                logger.error(f"Error updating user: {query.lastError().text()}")
                self.db.rollback()
                return False
                
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Exception in update_user: {e}")
            return False

    def get_user_id_by_student_id(self, student_id):
        """Trả về user_id dựa trên student_id."""
        try:
            if not student_id:
                logger.error("student_id cannot be empty")
                return None
                
            query = QSqlQuery(self.db)
            query.prepare("SELECT id FROM users WHERE student_id = :student_id")
            query.bindValue(":student_id", student_id)
            if query.exec_() and query.next():
                return query.value(0)
            return None
        except Exception as e:
            logger.error(f"Exception in get_user_id_by_student_id: {e}")
            return None

    def delete_user(self, student_id):
        """Xóa người dùng dựa trên student_id."""
        try:
            if not student_id:
                logger.error("Student_id cannot be empty")
                return False
                
            self.db.transaction()
            query = QSqlQuery(self.db)
            
            # Xóa các bản ghi điểm danh liên quan trước
            query.prepare("DELETE FROM attendance WHERE student_id = :student_id")
            query.bindValue(":student_id", student_id)
            if not query.exec_():
                logger.error(f"Error deleting attendance records: {query.lastError().text()}")
                self.db.rollback()
                return False
                
            # Sau đó xóa người dùng
            query.prepare("DELETE FROM users WHERE student_id = :student_id")
            query.bindValue(":student_id", student_id)
            if not query.exec_():
                logger.error(f"Error deleting user: {query.lastError().text()}")
                self.db.rollback()
                return False
                
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Exception in delete_user: {e}")
            return False

    def delete_users_without_embedding(self):
        """Xóa toàn bộ user không có embedding (đồng bộ dữ liệu)."""
        try:
            query = QSqlQuery(self.db)
            # Lấy danh sách student_id không có embedding
            query.exec_("SELECT student_id FROM users WHERE embedding IS NULL")
            ids = []
            while query.next():
                ids.append(query.value(0))
                
            count = 0
            for sid in ids:
                if self.delete_user(sid):
                    count += 1
                    
            logger.info(f"Deleted {count} users without embedding.")
            return count
        except Exception as e:
            logger.error(f"Exception in delete_users_without_embedding: {e}")
            return 0

    def get_all_users(self):
        """Lấy danh sách tất cả người dùng."""
        users = []
        try:
            query = QSqlQuery(self.db)
            # Đã loại bỏ log không cần thiết
            
            # Kiểm tra kết nối cơ sở dữ liệu
            if not self.db.isOpen():
                logger.error("Cơ sở dữ liệu không được mở khi gọi get_all_users")
                self.open_connection()
            
            # Kiểm tra xem cột created_at có tồn tại không
            columns = self._get_table_columns("users")
            has_created_at = "created_at" in columns
            
            # Xây dựng câu truy vấn dựa trên các cột có sẵn
            if has_created_at:
                sql = "SELECT name, student_id, class, email, created_at FROM users"
            else:
                sql = "SELECT name, student_id, class, email FROM users"
                
            # Thực hiện truy vấn với kiểm tra lỗi
            if not query.exec_(sql):
                logger.error(f"Lỗi SQL: {query.lastError().text()}")
                return []
                
            # Đếm số lượng bản ghi
            count = 0
            while query.next():
                if has_created_at:
                    users.append((
                        query.value(0),  # name
                        query.value(1),  # student_id
                        query.value(2),  # class
                        query.value(3),  # email
                        query.value(4)   # created_at
                    ))
                else:
                    # Nếu không có cột created_at, thêm giá trị mặc định
                    users.append((
                        query.value(0),  # name
                        query.value(1),  # student_id
                        query.value(2),  # class
                        query.value(3),  # email
                        ""            # created_at (empty string)
                    ))
                count += 1
                
            # Đã loại bỏ log không cần thiết
            return users
        except Exception as e:
            logger.error(f"Exception in get_all_users: {e}")
            return []
            
    def get_users_model(self):
        """Trả về QSqlTableModel cho bảng users."""
        try:
            model = QSqlTableModel(db=self.db)
            model.setTable("users")
            model.setEditStrategy(QSqlTableModel.OnManualSubmit)
            model.select()
            return model
        except Exception as e:
            logger.error(f"Exception in get_users_model: {e}")
            return None

    def get_all_embeddings(self):
        """Lấy embedding của tất cả người dùng có embedding."""
        try:
            embeddings = {}
            count_total = 0
            count_loaded = 0
            
            query = QSqlQuery(self.db)
            if query.exec_("SELECT student_id, name, embedding FROM users WHERE embedding IS NOT NULL"):
                while query.next():
                    sid = query.value(0)
                    name = query.value(1)
                    emb_blob = query.value(2)
                    count_total += 1
                    
                    try:
                        # Xử lý QByteArray hoặc bytes
                        if isinstance(emb_blob, QByteArray):
                            emb = pickle.loads(bytes(emb_blob))
                        elif isinstance(emb_blob, (bytes, bytearray)):
                            emb = pickle.loads(emb_blob)
                        else:
                            logger.error(f"Unsupported embedding type for {sid}: {type(emb_blob)}")
                            self._mark_embedding_corrupt(sid)
                            continue
                            
                        embeddings[sid] = {"name": name, "embedding": emb}
                        count_loaded += 1
                    except Exception as e:
                        logger.error(f"Error loading embedding for {sid}: {e}, marking as corrupt.")
                        self._mark_embedding_corrupt(sid)
                        continue
            
            # Tự động xóa user không có embedding để đồng bộ dữ liệu
            self.delete_users_without_embedding()
            # Đã loại bỏ log không cần thiết
            return embeddings
        except Exception as e:
            logger.error(f"Exception in get_all_embeddings: {e}")
            return {}

    def _mark_embedding_corrupt(self, student_id):
        """Đánh dấu embedding bị lỗi bằng cách xóa giá trị."""
        try:
            query = QSqlQuery(self.db)
            query.prepare("UPDATE users SET embedding = NULL WHERE student_id = ?")
            query.bindValue(0, student_id)
            query.exec_()
            logger.info(f"Marked corrupt embedding as NULL for {student_id}")
        except Exception as e:
            logger.error(f"Failed to mark corrupt embedding: {e}")

    def get_user_embedding(self, student_id):
        """Lấy embedding khuôn mặt của người dùng."""
        try:
            query = QSqlQuery(self.db)
            query.prepare("SELECT embedding FROM users WHERE student_id = ?")
            query.bindValue(0, student_id)
            
            if query.exec_() and query.next() and not query.isNull(0):
                raw = query.value(0)
                try:
                    if isinstance(raw, QByteArray):
                        return pickle.loads(bytes(raw))
                    elif isinstance(raw, (bytes, bytearray)):
                        return pickle.loads(raw)
                    else:
                        logger.error(f"Unsupported embedding type for {student_id}: {type(raw)}")
                        return None
                except Exception as e:
                    logger.error(f"Error unpickling embedding for {student_id}: {e}")
                    self._mark_embedding_corrupt(student_id)
                    return None
            return None
        except Exception as e:
            logger.error(f"Exception in get_user_embedding: {e}")
            return None

    def check_all_embedding_integrity(self):
        """Kiểm tra tất cả embedding trong DB, log các trường hợp lỗi/corrupt."""
        try:
            query = QSqlQuery(self.db)
            if not query.exec_("SELECT student_id, name, embedding FROM users WHERE embedding IS NOT NULL"):
                logger.error(f"Query failed: {query.lastError().text()}")
                return
                
            total = 0
            ok = 0
            error = 0
            
            while query.next():
                sid = query.value(0)
                name = query.value(1)
                emb_blob = query.value(2)
                total += 1
                
                try:
                    if isinstance(emb_blob, QByteArray):
                        pickle.loads(bytes(emb_blob))
                    elif isinstance(emb_blob, (bytes, bytearray)):
                        pickle.loads(emb_blob)
                    else:
                        logger.error(f"Embedding for {sid} is of unsupported type: {type(emb_blob)}")
                        self._mark_embedding_corrupt(sid)
                        error += 1
                        continue
                    ok += 1
                except Exception as e:
                    logger.error(f"Embedding for {sid} is corrupted or invalid: {e}")
                    self._mark_embedding_corrupt(sid)
                    error += 1
                    
            # Đã loại bỏ log không cần thiết
            return ok, error
        except Exception as e:
            logger.error(f"Exception in check_all_embedding_integrity: {e}")
            return 0, 0

    def get_attendance_records(self, date=None, from_date=None, to_date=None, class_name=None, status=None):
        """Lấy bản ghi điểm danh sử dụng JOIN trên QSqlQuery."""
        try:
            records = []
            query = QSqlQuery(self.db)
            
            sql = (
                "SELECT u.name, a.student_id, u.class, a.date, a.time, a.status "
                "FROM attendance a JOIN users u ON a.student_id = u.student_id"
            )
            
            conditions = []
            values = []
            
            if date:
                conditions.append("a.date = ?")
                values.append(date)
            if from_date:
                conditions.append("a.date >= ?")
                values.append(from_date)
            if to_date:
                conditions.append("a.date <= ?")
                values.append(to_date)
            if class_name:
                conditions.append("u.class = ?")
                values.append(class_name)
            if status:
                conditions.append("a.status = ?")
                values.append(status)
                
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
                
            sql += " ORDER BY a.date, a.time"
            
            query.prepare(sql)
            for i, value in enumerate(values):
                query.bindValue(i, value)
                
            if query.exec_():
                while query.next():
                    records.append((
                        query.value(0),  # name
                        query.value(1),  # student_id
                        query.value(2),  # class
                        query.value(3),  # date
                        query.value(4),  # time
                        query.value(5)   # status
                    ))
            else:
                logger.error(f"Error executing attendance query: {query.lastError().text()}")
                
            return records
        except Exception as e:
            logger.error(f"Exception in get_attendance_records: {e}")
            return []

    def record_attendance(self, student_id, name, class_name, status, confidence):
        """Ghi lại bản ghi điểm danh."""
        try:
            self.db.transaction()
            query = QSqlQuery(self.db)
            query.prepare("""
                INSERT INTO attendance (student_id, name, class, date, time, status, confidence)
                VALUES (:student_id, :name, :class_name, :date, :time, :status, :confidence)
            """)
            query.bindValue(":student_id", student_id)
            query.bindValue(":name", name)
            query.bindValue(":class_name", class_name)
            query.bindValue(":date", datetime.now().strftime("%Y-%m-%d"))
            query.bindValue(":time", datetime.now().strftime("%H:%M:%S"))
            query.bindValue(":status", status)
            query.bindValue(":confidence", confidence)
            
            if not query.exec_():
                logger.error(f"Error recording attendance: {query.lastError().text()}")
                self.db.rollback()
                return False
                
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Exception in record_attendance: {e}")
            return False

    def clear_attendance(self):
        """Xóa tất cả bản ghi điểm danh."""
        try:
            self.db.transaction()
            query = QSqlQuery(self.db)
            if not query.exec_("DELETE FROM attendance"):
                logger.error(f"Error clearing attendance: {query.lastError().text()}")
                self.db.rollback()
                return False
                
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Exception in clear_attendance: {e}")
            return False

    def clear_today_attendance(self):
        """Xóa bản ghi điểm danh của ngày hôm nay."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            self.db.transaction()
            query = QSqlQuery(self.db)
            query.prepare("DELETE FROM attendance WHERE date = :today")
            query.bindValue(":today", today)
            
            if not query.exec_():
                logger.error(f"Error clearing today's attendance: {query.lastError().text()}")
                self.db.rollback()
                return False
                
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Exception in clear_today_attendance: {e}")
            return False

    def backup(self, backup_path):
        """Sao lưu cơ sở dữ liệu sử dụng .backup của SQLite."""
        try:
            # Đảm bảo tất cả các thay đổi được ghi vào DB
            self.db.close()
            
            # Sử dụng QProcess để thực hiện lệnh sqlite3 cho việc sao lưu
            from PyQt5.QtCore import QProcess
            process = QProcess()
            db_path = self.db.databaseName()
            
            # Tạo lệnh backup
            process.start("sqlite3", [db_path, f".backup {backup_path}"])
            process.waitForFinished()
            
            # Kiểm tra trạng thái
            if process.exitCode() != 0:
                logger.error(f"Backup failed: {process.readAllStandardError().data().decode()}")
                return False
                
            # Mở lại kết nối
            if not self.db.open():
                logger.error(f"Failed to reopen database after backup: {self.db.lastError().text()}")
                return False
                
            logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Exception in backup: {e}")
            # Đảm bảo kết nối được mở lại
            if not self.db.isOpen():
                self.db.open()
            return False

    def restore(self, backup_path):
        """Khôi phục cơ sở dữ liệu từ file sao lưu."""
        try:
            if not os.path.exists(backup_path):
                logger.error(f"Backup file {backup_path} does not exist")
                return False
                
            # Đóng kết nối trước khi thay thế file
            self.db.close()
            db_path = self.db.databaseName()
            
            # Thay thế file DB hiện tại bằng file backup
            try:
                os.replace(backup_path, db_path)
            except OSError as e:
                logger.error(f"Failed to replace database file: {e}")
                self.db.open()
                return False
                
            # Mở lại kết nối
            if not self.db.open():
                logger.error(f"Failed to reopen database after restore: {self.db.lastError().text()}")
                return False
                
            logger.info(f"Database restored from {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Exception in restore: {e}")
            # Đảm bảo kết nối được mở lại
            if not self.db.isOpen():
                self.db.open()
            return False

    def close(self):
        """Đóng kết nối cơ sở dữ liệu."""
        try:
            if self.db.isOpen():
                self.db.close()
            QSqlDatabase.removeDatabase(self.connection_name)
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Exception in close: {e}")

    def get_total_students(self):
        """Trả về tổng số sinh viên đã đăng ký (có embedding)."""
        try:
            query = QSqlQuery(self.db)
            if query.exec_("SELECT COUNT(*) FROM users WHERE embedding IS NOT NULL"):
                if query.next():
                    return int(query.value(0))
            return 0
        except Exception as e:
            logger.error(f"Exception in get_total_students: {e}")
            return 0