"""Append-only-per-result SQLite archive, independent of Qt and calculations."""
from contextlib import contextmanager
from pathlib import Path
import re
import sqlite3

from arraylab.features.linear.records import MAX_BYTES, linear_record_bytes, linear_record_from_bytes


APPLICATION_ID = 0x414C4C48
FORMAT = "arraylab.linear_history"
VERSION = 1


class HistoryStoreError(ValueError):
    pass


class LinearHistoryStore:
    """Keep every committed result; individual transactions never flush a view."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        try:
            if self.path.exists():
                # A legitimate interrupted write may leave a hot rollback
                # journal. SQLite needs write permission to roll that back to
                # the last committed state before a SELECT can read the schema.
                with self._connect() as connection:
                    self._check_schema(connection)
                    if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                        raise HistoryStoreError("自动历史数据库完整性检查失败；原文件未覆盖。")
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(self.path, timeout=3)
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    if connection.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0]:
                        self._check_schema(connection)  # Another new window won creation.
                    else:
                        connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
                        connection.execute(f"PRAGMA user_version={VERSION}")
                        connection.execute("CREATE TABLE history_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                        connection.execute("CREATE TABLE linear_results (id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT NOT NULL, payload BLOB NOT NULL)")
                        connection.executemany("INSERT INTO history_meta VALUES (?,?)", [
                            ("format", FORMAT), ("sequence", "0"), ("selected_id", "-1")])
                    connection.commit()
                finally:
                    connection.close()
        except (sqlite3.Error, OSError) as exc:
            raise HistoryStoreError(f"无法打开自动历史档案，原文件未覆盖：{exc}") from exc

    @contextmanager
    def _connect(self, readonly=False):
        # Reject unrelated/corrupt files before allowing SQLite's own journal
        # recovery. Application ID is a big-endian field at header offset 68.
        with self.path.open("rb") as stream:
            header = stream.read(100)
        if (len(header) < 100 or header[:16] != b"SQLite format 3\0"
                or int.from_bytes(header[68:72], "big") != APPLICATION_ID
                or int.from_bytes(header[60:64], "big") != VERSION):
            raise HistoryStoreError("不是有效的ArrayLab自动历史档案；原文件未覆盖。")
        connection = sqlite3.connect(self.path.as_uri() + ("?mode=ro" if readonly else "?mode=rw"),
                                     uri=True, timeout=3)
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _check_schema(connection):
        if (connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
                or connection.execute("PRAGMA user_version").fetchone()[0] != VERSION):
            raise HistoryStoreError("自动历史文件格式或版本不受支持；原文件未覆盖。")
        objects = connection.execute("SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 4").fetchall()
        if set(objects) != {("table", "history_meta"), ("table", "linear_results")}:
            raise HistoryStoreError("自动历史文件结构异常；原文件未覆盖。")
        if [row[1] for row in connection.execute("PRAGMA table_info(linear_results)")] != ["id", "label", "payload"]:
            raise HistoryStoreError("自动历史记录字段异常；原文件未覆盖。")
        count, key_length, value_length = connection.execute(
            "SELECT COUNT(*),MAX(LENGTH(key)),MAX(LENGTH(value)) FROM history_meta").fetchone()
        if (count != 3 or type(key_length) is not int or type(value_length) is not int
                or key_length > 32 or value_length > 128):
            raise HistoryStoreError("自动历史元数据大小异常；原文件未覆盖。")
        meta = dict(connection.execute("SELECT key,value FROM history_meta"))
        if (not all(type(key) is str and type(value) is str for key, value in meta.items())
                or set(meta) != {"format", "sequence", "selected_id"} or meta["format"] != FORMAT
                or not re.fullmatch(r"\d+", meta["sequence"])
                or not re.fullmatch(r"-1|\d+", meta["selected_id"])):
            raise HistoryStoreError("自动历史元数据异常；原文件未覆盖。")
        return meta

    def load(self):
        """Return all entries only after every entry validates successfully."""
        try:
            with self._connect() as connection:
                connection.execute("BEGIN")
                meta = self._check_schema(connection)
                records = []
                metadata = connection.execute("SELECT id,typeof(label),length(label),typeof(payload),length(payload) FROM linear_results ORDER BY id")
                for row_id, label_type, label_length, payload_type, payload_length in metadata:
                    if (label_type != "text" or not 1 <= label_length <= 1024
                            or payload_type != "blob" or not 1 <= payload_length <= MAX_BYTES):
                        raise HistoryStoreError("自动历史记录类型或大小异常；原文件未覆盖。")
                    label, payload = connection.execute("SELECT label,payload FROM linear_results WHERE id=?", (row_id,)).fetchone()
                    record = linear_record_from_bytes(payload)
                    if record["label"] != label:
                        raise HistoryStoreError("自动历史标签与快照不一致；原文件未覆盖。")
                    records.append((row_id, record))
                return records, int(meta["sequence"]), int(meta["selected_id"])
        except (sqlite3.Error, UnicodeError, OSError) as exc:
            raise HistoryStoreError(f"无法读取自动历史档案，原文件未覆盖：{exc}") from exc

    def append(self, record, sequence, *, generated=False):
        # Validate and detach the serialization before starting a write lock.
        detached = linear_record_from_bytes(linear_record_bytes(record))
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                meta = self._check_schema(connection)
                current = int(meta["sequence"])
                actual_sequence = max(int(sequence), current + int(generated))
                if generated:
                    detached["label"] = re.sub(r"^#\d+(?=\s|$)", f"#{actual_sequence}", detached["label"])
                raw = linear_record_bytes(detached)
                cursor = connection.execute("INSERT INTO linear_results(label,payload) VALUES (?,?)",
                                            (detached["label"], raw))
                row_id = cursor.lastrowid
                connection.executemany("UPDATE history_meta SET value=? WHERE key=?", [
                    (str(actual_sequence), "sequence"), (str(row_id), "selected_id")])
                connection.commit()
                return row_id, actual_sequence, detached["label"]
        except (sqlite3.Error, OSError) as exc:
            raise HistoryStoreError(f"自动保存失败，本次结果仍保留在内存：{exc}") from exc

    def select(self, row_id):
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._check_schema(connection)
                if connection.execute("SELECT 1 FROM linear_results WHERE id=?", (row_id,)).fetchone():
                    connection.execute("UPDATE history_meta SET value=? WHERE key='selected_id'", (str(row_id),))
                connection.commit()
        except (sqlite3.Error, OSError) as exc:
            raise HistoryStoreError(f"所选历史状态未保存：{exc}") from exc

    def remove(self, row_id, selected_id=-1):
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._check_schema(connection)
                connection.execute("DELETE FROM linear_results WHERE id=?", (row_id,))
                if not connection.execute("SELECT 1 FROM linear_results WHERE id=?", (selected_id,)).fetchone():
                    newest = connection.execute("SELECT id FROM linear_results ORDER BY id DESC LIMIT 1").fetchone()
                    selected_id = newest[0] if newest else -1
                connection.execute("UPDATE history_meta SET value=? WHERE key='selected_id'", (str(selected_id),))
                connection.commit()
        except (sqlite3.Error, OSError) as exc:
            raise HistoryStoreError(f"删除历史记录失败，档案未变更：{exc}") from exc
