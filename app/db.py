import re
import sqlite3
from typing import Any
from app.config import settings
from app.auth import hash_password, verify_password


def normalize_query(query: str) -> str:
    norm = re.sub(r"\s+", " ", query.strip().lower())
    norm = re.sub(r"[^a-z0-9\s+#.]", "", norm)
    return norm[:400]


def normalize_user_id(user_id: str) -> str:
    cleaned = (user_id or "").strip().replace(" ", "_")
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "", cleaned)
    return cleaned[:64]


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_chat_columns(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(chat_history)")
    cols = {row[1] for row in cur.fetchall()}
    if "subject" not in cols:
        cur.execute("ALTER TABLE chat_history ADD COLUMN subject TEXT DEFAULT ''")
    if "request_style" not in cols:
        cur.execute("ALTER TABLE chat_history ADD COLUMN request_style TEXT DEFAULT ''")


def _ensure_app_user_columns(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(app_users)")
    cols = {row[1] for row in cur.fetchall()}
    if "password_hash" not in cols:
        cur.execute("ALTER TABLE app_users ADD COLUMN password_hash TEXT DEFAULT ''")
    if "role" not in cols:
        cur.execute("ALTER TABLE app_users ADD COLUMN role TEXT DEFAULT 'user'")


def _backfill_user_query_stats(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM question_reward_stats_user")
    n = int(cur.fetchone()["n"])
    if n > 0:
        return

    cur.execute(
        """
        INSERT OR REPLACE INTO question_reward_stats_user
            (user_id, query_key, sample_question, avg_reward, total_feedback, low_rating_count, last_rating, updated_at)
        SELECT
            f.user_id,
            c.query_key,
            MAX(c.query) AS sample_question,
            AVG(f.reward) AS avg_reward,
            COUNT(*) AS total_feedback,
            SUM(CASE WHEN f.rating <= 2 THEN 1 ELSE 0 END) AS low_rating_count,
            (
                SELECT f2.rating
                FROM feedback_log f2
                JOIN chat_history c2 ON c2.id = f2.chat_id
                WHERE f2.user_id = f.user_id AND c2.query_key = c.query_key
                ORDER BY f2.id DESC
                LIMIT 1
            ) AS last_rating,
            MAX(f.created_at) AS updated_at
        FROM feedback_log f
        JOIN chat_history c ON c.id = f.chat_id
        GROUP BY f.user_id, c.query_key
        """
    )


def _backfill_users(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO app_users (user_id)
        SELECT DISTINCT user_id FROM (
            SELECT user_id FROM chat_history WHERE TRIM(user_id) <> ''
            UNION ALL
            SELECT user_id FROM feedback_log WHERE TRIM(user_id) <> ''
            UNION ALL
            SELECT user_id FROM user_prefs WHERE TRIM(user_id) <> ''
        )
        """
    )


def _ensure_default_admin(conn: sqlite3.Connection) -> None:
    admin_id = normalize_user_id(settings.default_admin_user)
    if len(admin_id) < 3:
        return

    cur = conn.cursor()
    cur.execute(
        "SELECT user_id, password_hash, role FROM app_users WHERE user_id = ?",
        (admin_id,),
    )
    row = cur.fetchone()
    if row is None:
        cur.execute(
            """
            INSERT INTO app_users (user_id, password_hash, role)
            VALUES (?, ?, 'admin')
            """,
            (admin_id, hash_password(settings.default_admin_password)),
        )
        return

    updates: list[str] = []
    params: list[Any] = []
    if str(row["role"] or "").lower() != "admin":
        updates.append("role = 'admin'")
    if not str(row["password_hash"] or "").strip():
        updates.append("password_hash = ?")
        params.append(hash_password(settings.default_admin_password))
    if updates:
        params.append(admin_id)
        cur.execute(f"UPDATE app_users SET {', '.join(updates)} WHERE user_id = ?", tuple(params))


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            query TEXT NOT NULL,
            query_key TEXT NOT NULL,
            response TEXT NOT NULL,
            sources TEXT,
            mode_used TEXT,
            subject TEXT DEFAULT '',
            request_style TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            user_id TEXT NOT NULL,
            rating INTEGER NOT NULL,
            reward REAL NOT NULL,
            mode_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chat_history (id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_prefs (
            user_id TEXT PRIMARY KEY,
            preferred_mode TEXT DEFAULT 'detailed',
            avg_reward REAL DEFAULT 0.0,
            total_feedback INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS question_reward_stats (
            query_key TEXT PRIMARY KEY,
            sample_question TEXT NOT NULL,
            avg_reward REAL DEFAULT 0.0,
            total_feedback INTEGER DEFAULT 0,
            low_rating_count INTEGER DEFAULT 0,
            last_rating INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS question_reward_stats_user (
            user_id TEXT NOT NULL,
            query_key TEXT NOT NULL,
            sample_question TEXT NOT NULL,
            avg_reward REAL DEFAULT 0.0,
            total_feedback INTEGER DEFAULT 0,
            low_rating_count INTEGER DEFAULT 0,
            last_rating INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, query_key)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_users (
            user_id TEXT PRIMARY KEY,
            password_hash TEXT DEFAULT '',
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback_log(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_query_key ON chat_history(query_key)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_chat_id ON feedback_log(chat_id)")

    _ensure_chat_columns(conn)
    _ensure_app_user_columns(conn)
    cur.execute("PRAGMA table_info(chat_history)")
    cols = {row[1] for row in cur.fetchall()}
    if "query_key" not in cols:
        cur.execute("ALTER TABLE chat_history ADD COLUMN query_key TEXT DEFAULT ''")
        cur.execute("SELECT id, query FROM chat_history")
        for row in cur.fetchall():
            cur.execute(
                "UPDATE chat_history SET query_key = ? WHERE id = ?",
                (normalize_query(row["query"]), row["id"]),
            )

    _backfill_user_query_stats(conn)
    _backfill_users(conn)
    _ensure_default_admin(conn)

    conn.commit()
    conn.close()


def save_chat(
    user_id: str,
    query: str,
    response: str,
    sources: str,
    mode_used: str,
    subject: str = "",
    request_style: str = "",
) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chat_history (user_id, query, query_key, response, sources, mode_used, subject, request_style)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            normalize_user_id(user_id),
            query,
            normalize_query(query),
            response,
            sources,
            mode_used,
            subject,
            request_style,
        ),
    )
    chat_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(chat_id)


def get_chat_by_id(chat_id: int) -> dict[str, Any] | None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM chat_history WHERE id = ?", (chat_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def save_feedback(chat_id: int, user_id: str, rating: int, reward: float, mode_used: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO feedback_log (chat_id, user_id, rating, reward, mode_used)
        VALUES (?, ?, ?, ?, ?)
        """,
        (chat_id, normalize_user_id(user_id), rating, reward, mode_used),
    )
    conn.commit()
    conn.close()


def upsert_question_reward(query_key: str, sample_question: str, rating: int, reward: float) -> dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM question_reward_stats WHERE query_key = ?", (query_key,))
    row = cur.fetchone()

    if row:
        n = row["total_feedback"]
        avg = row["avg_reward"]
        low = row["low_rating_count"]
        new_n = n + 1
        new_avg = ((avg * n) + reward) / new_n
        new_low = low + (1 if rating <= 2 else 0)
        cur.execute(
            """
            UPDATE question_reward_stats
            SET sample_question=?, avg_reward=?, total_feedback=?, low_rating_count=?, last_rating=?, updated_at=CURRENT_TIMESTAMP
            WHERE query_key=?
            """,
            (sample_question, new_avg, new_n, new_low, rating, query_key),
        )
    else:
        new_n = 1
        new_avg = reward
        new_low = 1 if rating <= 2 else 0
        cur.execute(
            """
            INSERT INTO question_reward_stats (query_key, sample_question, avg_reward, total_feedback, low_rating_count, last_rating)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (query_key, sample_question, new_avg, new_n, new_low, rating),
        )

    conn.commit()
    conn.close()
    return {
        "query_key": query_key,
        "sample_question": sample_question,
        "avg_reward": round(new_avg, 4),
        "total_feedback": new_n,
        "low_rating_count": new_low,
        "last_rating": rating,
    }


def upsert_question_reward_user(user_id: str, query_key: str, sample_question: str, rating: int, reward: float) -> dict[str, Any]:
    user_id = normalize_user_id(user_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM question_reward_stats_user WHERE user_id = ? AND query_key = ?",
        (user_id, query_key),
    )
    row = cur.fetchone()

    if row:
        n = row["total_feedback"]
        avg = row["avg_reward"]
        low = row["low_rating_count"]
        new_n = n + 1
        new_avg = ((avg * n) + reward) / new_n
        new_low = low + (1 if rating <= 2 else 0)
        cur.execute(
            """
            UPDATE question_reward_stats_user
            SET sample_question=?, avg_reward=?, total_feedback=?, low_rating_count=?, last_rating=?, updated_at=CURRENT_TIMESTAMP
            WHERE user_id=? AND query_key=?
            """,
            (sample_question, new_avg, new_n, new_low, rating, user_id, query_key),
        )
    else:
        new_n = 1
        new_avg = reward
        new_low = 1 if rating <= 2 else 0
        cur.execute(
            """
            INSERT INTO question_reward_stats_user (user_id, query_key, sample_question, avg_reward, total_feedback, low_rating_count, last_rating)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, query_key, sample_question, new_avg, new_n, new_low, rating),
        )

    conn.commit()
    conn.close()
    return {
        "user_id": user_id,
        "query_key": query_key,
        "sample_question": sample_question,
        "avg_reward": round(new_avg, 4),
        "total_feedback": new_n,
        "low_rating_count": new_low,
        "last_rating": rating,
    }


def get_question_reward(query: str, user_id: str | None = None) -> dict[str, Any]:
    key = normalize_query(query)
    conn = get_conn()
    cur = conn.cursor()
    if user_id:
        user_id = normalize_user_id(user_id)
        cur.execute(
            "SELECT * FROM question_reward_stats_user WHERE user_id = ? AND query_key = ?",
            (user_id, key),
        )
        row = cur.fetchone()
        if row:
            conn.close()
            return dict(row)
    cur.execute("SELECT * FROM question_reward_stats WHERE query_key = ?", (key,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {
            "query_key": key,
            "sample_question": query,
            "avg_reward": 0.0,
            "total_feedback": 0,
            "low_rating_count": 0,
            "last_rating": 0,
        }
    return dict(row)


def get_query_attempt_count(query: str) -> int:
    key = normalize_query(query)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM chat_history WHERE query_key = ?", (key,))
    n = cur.fetchone()["n"]
    conn.close()
    return int(n)


def get_last_response_for_query(query: str) -> str:
    key = normalize_query(query)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT response
        FROM chat_history
        WHERE query_key = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (key,),
    )
    row = cur.fetchone()
    conn.close()
    return str(row["response"]) if row else ""


def get_user_query_feedback_samples(user_id: str, query: str, limit: int = 8) -> list[dict[str, Any]]:
    uid = normalize_user_id(user_id)
    key = normalize_query(query)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            f.id AS feedback_id,
            f.chat_id,
            f.user_id,
            f.rating,
            f.reward,
            f.mode_used,
            f.created_at,
            c.query,
            c.query_key,
            c.response,
            c.request_style
        FROM feedback_log f
        JOIN chat_history c ON c.id = f.chat_id
        WHERE f.user_id = ? AND c.query_key = ?
        ORDER BY f.id DESC
        LIMIT ?
        """,
        (uid, key, limit),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_reward_history(limit: int = 100, user_id: str | None = None) -> list[dict]:
    conn = get_conn()
    cur = conn.cursor()
    if user_id:
        user_id = normalize_user_id(user_id)
        cur.execute(
            """
            SELECT f.id, f.chat_id, c.query, c.query_key, f.user_id, f.rating, f.reward, f.mode_used, f.created_at
            FROM feedback_log f
            JOIN chat_history c ON c.id = f.chat_id
            WHERE f.user_id = ?
            ORDER BY f.id DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
    else:
        cur.execute(
            """
            SELECT f.id, f.chat_id, c.query, c.query_key, f.user_id, f.rating, f.reward, f.mode_used, f.created_at
            FROM feedback_log f
            JOIN chat_history c ON c.id = f.chat_id
            ORDER BY f.id DESC
            LIMIT ?
            """,
            (limit,),
        )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_analytics(user_id: str | None = None, scope: str = "all", weak_topics_limit: int = 1000) -> dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()

    scoped_user = normalize_user_id(user_id) if (scope or "").lower() == "current_user" and user_id else None

    if scoped_user:
        cur.execute("SELECT COUNT(*) AS n FROM chat_history WHERE user_id = ?", (scoped_user,))
        total_chats = cur.fetchone()["n"]

        cur.execute("SELECT COUNT(*) AS n FROM feedback_log WHERE user_id = ?", (scoped_user,))
        total_feedback = cur.fetchone()["n"]

        cur.execute(
            """
            SELECT COALESCE(AVG(rating), 0) AS avg_rating, COALESCE(AVG(reward), 0) AS avg_reward
            FROM feedback_log
            WHERE user_id = ?
            """,
            (scoped_user,),
        )
        agg = cur.fetchone()

        cur.execute(
            """
            SELECT query_key, sample_question, avg_reward, total_feedback, low_rating_count, last_rating, updated_at
            FROM question_reward_stats_user
            WHERE user_id = ?
            ORDER BY updated_at DESC, low_rating_count DESC, avg_reward ASC
            LIMIT ?
            """,
            (scoped_user, weak_topics_limit),
        )
        weak_topics = [dict(r) for r in cur.fetchall()]
    else:
        cur.execute("SELECT COUNT(*) AS n FROM chat_history")
        total_chats = cur.fetchone()["n"]

        cur.execute("SELECT COUNT(*) AS n FROM feedback_log")
        total_feedback = cur.fetchone()["n"]

        cur.execute("SELECT COALESCE(AVG(rating), 0) AS avg_rating, COALESCE(AVG(reward), 0) AS avg_reward FROM feedback_log")
        agg = cur.fetchone()

        cur.execute(
            """
            SELECT query_key, sample_question, avg_reward, total_feedback, low_rating_count, last_rating, updated_at
            FROM question_reward_stats
            ORDER BY updated_at DESC, low_rating_count DESC, avg_reward ASC
            LIMIT ?
            """,
            (weak_topics_limit,),
        )
        weak_topics = [dict(r) for r in cur.fetchall()]

    conn.close()
    return {
        "total_chats": total_chats,
        "total_feedback": total_feedback,
        "avg_rating": round(float(agg["avg_rating"]), 4),
        "avg_reward": round(float(agg["avg_reward"]), 4),
        "weak_topics": weak_topics,
    }


def get_user_pref(user_id: str) -> dict[str, Any]:
    user_id = normalize_user_id(user_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM user_prefs WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {
            "user_id": user_id,
            "preferred_mode": "detailed",
            "avg_reward": 0.0,
            "total_feedback": 0,
        }
    return dict(row)


def upsert_user_pref(user_id: str, preferred_mode: str, avg_reward: float, total_feedback: int) -> None:
    user_id = normalize_user_id(user_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO user_prefs (user_id, preferred_mode, avg_reward, total_feedback)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            preferred_mode=excluded.preferred_mode,
            avg_reward=excluded.avg_reward,
            total_feedback=excluded.total_feedback,
            updated_at=CURRENT_TIMESTAMP
        """,
        (user_id, preferred_mode, avg_reward, total_feedback),
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 50) -> dict[str, list[dict]]:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, user_id, query, query_key, response, sources, mode_used, subject, request_style, created_at
        FROM chat_history
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    chats = [dict(row) for row in cur.fetchall()]

    cur.execute(
        """
        SELECT id, chat_id, user_id, rating, reward, mode_used, created_at
        FROM feedback_log
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    feedback = [dict(row) for row in cur.fetchall()]
    conn.close()

    return {"chats": chats, "feedback": feedback}


def create_user(user_id: str) -> tuple[bool, str]:
    uid = normalize_user_id(user_id)
    if len(uid) < 3:
        return False, "User ID must be at least 3 valid characters."

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM app_users WHERE user_id = ?", (uid,))
    if cur.fetchone() is not None:
        conn.close()
        return False, "User ID already exists."

    cur.execute("INSERT INTO app_users (user_id, role) VALUES (?, 'user')", (uid,))
    conn.commit()
    conn.close()
    return True, uid


def create_user_credentials(user_id: str, password: str, role: str = "user") -> tuple[bool, str]:
    uid = normalize_user_id(user_id)
    role_norm = "admin" if str(role or "").lower() == "admin" else "user"
    if len(uid) < 3:
        return False, "User ID must be at least 3 valid characters."
    if len(password or "") < 6:
        return False, "Password must be at least 6 characters."
    if role_norm == "admin":
        return False, "Admin accounts cannot be self-registered."

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT user_id, password_hash, role FROM app_users WHERE user_id = ?", (uid,))
    row = cur.fetchone()
    if row is not None:
        existing_hash = str(row["password_hash"] or "").strip()
        existing_role = str(row["role"] or "user").lower()
        if existing_role == "admin":
            conn.close()
            return False, "User ID already exists."
        if existing_hash:
            conn.close()
            return False, "User ID already exists."
        cur.execute(
            """
            UPDATE app_users
            SET password_hash = ?, role = 'user'
            WHERE user_id = ?
            """,
            (hash_password(password), uid),
        )
        conn.commit()
        conn.close()
        return True, uid

    cur.execute(
        """
        INSERT INTO app_users (user_id, password_hash, role)
        VALUES (?, ?, ?)
        """,
        (uid, hash_password(password), role_norm),
    )
    conn.commit()
    conn.close()
    return True, uid


def authenticate_user(user_id: str, password: str) -> dict[str, Any] | None:
    uid = normalize_user_id(user_id)
    if len(uid) < 3 or not password:
        return None
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT user_id, password_hash, role, created_at
        FROM app_users
        WHERE user_id = ?
        """,
        (uid,),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    if not verify_password(password, str(row["password_hash"] or "")):
        return None
    return {
        "user_id": row["user_id"],
        "role": str(row["role"] or "user"),
        "created_at": row["created_at"],
    }


def get_user(user_id: str) -> dict[str, Any] | None:
    uid = normalize_user_id(user_id)
    if len(uid) < 3:
        return None
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT user_id, role, created_at
        FROM app_users
        WHERE user_id = ?
        """,
        (uid,),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def user_exists(user_id: str) -> bool:
    uid = normalize_user_id(user_id)
    if len(uid) < 3:
        return False
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM app_users WHERE user_id = ?", (uid,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def list_users(limit: int = 500) -> list[dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT user_id, role, created_at
        FROM app_users
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
