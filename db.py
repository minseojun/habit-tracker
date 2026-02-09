import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

DB_PATH = "habit_tracker.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()


def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            nickname TEXT NOT NULL,
            city TEXT NOT NULL,
            daily_goal_n INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS habits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            frequency_type TEXT NOT NULL CHECK (frequency_type IN ('daily','weekly')),
            frequency_n INTEGER NOT NULL DEFAULT 1,
            start_date TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS habit_logs (
            date TEXT NOT NULL,
            habit_id INTEGER NOT NULL,
            completed INTEGER NOT NULL DEFAULT 0,
            memo TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (date, habit_id),
            FOREIGN KEY (habit_id) REFERENCES habits(id)
        )
        """)


def get_profile() -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM user_profile WHERE id=1").fetchone()
        return dict(row) if row else None


def upsert_profile(nickname: str, city: str, daily_goal_n: int):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute("""
        INSERT INTO user_profile (id, nickname, city, daily_goal_n, created_at)
        VALUES (1, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            nickname=excluded.nickname,
            city=excluded.city,
            daily_goal_n=excluded.daily_goal_n
        """, (nickname.strip(), city.strip(), int(daily_goal_n), now))


def list_active_habits() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM habits
            WHERE is_active=1
            ORDER BY created_at ASC
        """).fetchall()
        return [dict(r) for r in rows]


def list_all_habits() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM habits
            ORDER BY is_active DESC, created_at ASC
        """).fetchall()
        return [dict(r) for r in rows]


def add_habit(name: str, category: str, frequency_type: str, frequency_n: int, start_date: str):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO habits (name, category, frequency_type, frequency_n, start_date, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
        """, (name.strip(), category.strip(), frequency_type, int(frequency_n), start_date, now))


def update_habit(habit_id: int, name: str, category: str, frequency_type: str, frequency_n: int, start_date: str, is_active: int):
    with get_conn() as conn:
        conn.execute("""
            UPDATE habits
            SET name=?, category=?, frequency_type=?, frequency_n=?, start_date=?, is_active=?
            WHERE id=?
        """, (name.strip(), category.strip(), frequency_type, int(frequency_n), start_date, int(is_active), int(habit_id)))


def delete_habit(habit_id: int):
    # “삭제”는 안전하게 비활성 처리(로그 정합성 보존)
    with get_conn() as conn:
        conn.execute("UPDATE habits SET is_active=0 WHERE id=?", (int(habit_id),))


def get_log(d: str, habit_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("""
            SELECT * FROM habit_logs
            WHERE date=? AND habit_id=?
        """, (d, int(habit_id))).fetchone()
        return dict(row) if row else None


def upsert_log(d: str, habit_id: int, completed: bool, memo: Optional[str] = None):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute("""
        INSERT INTO habit_logs (date, habit_id, completed, memo, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(date, habit_id) DO UPDATE SET
            completed=excluded.completed,
            memo=COALESCE(excluded.memo, habit_logs.memo)
        """, (d, int(habit_id), 1 if completed else 0, memo, now))


def set_memo(d: str, habit_id: int, memo: str):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute("""
        INSERT INTO habit_logs (date, habit_id, completed, memo, created_at)
        VALUES (?, ?, 0, ?, ?)
        ON CONFLICT(date, habit_id) DO UPDATE SET
            memo=excluded.memo
        """, (d, int(habit_id), memo, now))


def get_today_counts(d: str) -> Tuple[int, int]:
    """
    오늘 완료 수 / 오늘 전체 습관 수
    """
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM habits WHERE is_active=1").fetchone()["c"]
        completed = conn.execute("""
            SELECT COUNT(*) AS c
            FROM habit_logs hl
            JOIN habits h ON h.id=hl.habit_id
            WHERE hl.date=? AND hl.completed=1 AND h.is_active=1
        """, (d,)).fetchone()["c"]
        return int(completed), int(total)


def get_last_7_days_series(end_date: date) -> List[Dict[str, Any]]:
    """
    최근 7일(오늘 포함) 완료수/전체습관/달성률
    """
    days = [(end_date - timedelta(days=i)) for i in range(6, -1, -1)]
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM habits WHERE is_active=1").fetchone()["c"]
        total = int(total) if total else 0

        out = []
        for day in days:
            d = day.isoformat()
            completed = conn.execute("""
                SELECT COUNT(*) AS c
                FROM habit_logs hl
                JOIN habits h ON h.id=hl.habit_id
                WHERE hl.date=? AND hl.completed=1 AND h.is_active=1
            """, (d,)).fetchone()["c"]
            completed = int(completed) if completed else 0
            rate = (completed / total) if total > 0 else 0.0
            out.append({"date": d, "completed": completed, "total": total, "rate": rate})
        return out


def get_avg_7d(end_date: date) -> float:
    s = get_last_7_days_series(end_date)
    if not s:
        return 0.0
    return sum(x["rate"] for x in s) / len(s)


def calc_streak(today: date, daily_goal_n: int) -> int:
    """
    streak 계산 규칙:
    - 오늘 기준 연속 일수
    - 하루 완료 수가 goal 미만이면 끊김
    """
    goal = max(1, int(daily_goal_n))
    streak = 0
    cursor = today
    with get_conn() as conn:
        total_habits = conn.execute("SELECT COUNT(*) AS c FROM habits WHERE is_active=1").fetchone()["c"]
        total_habits = int(total_habits) if total_habits else 0

        # 습관이 하나도 없으면 streak는 0
        if total_habits == 0:
            return 0

        while True:
            d = cursor.isoformat()
            completed = conn.execute("""
                SELECT COUNT(*) AS c
                FROM habit_logs hl
                JOIN habits h ON h.id=hl.habit_id
                WHERE hl.date=? AND hl.completed=1 AND h.is_active=1
            """, (d,)).fetchone()["c"]
            completed = int(completed) if completed else 0

            if completed >= goal:
                streak += 1
                cursor = cursor - timedelta(days=1)
            else:
                break

    return streak


def get_weekday_heatmap(end_date: date, weeks: int = 8) -> List[Dict[str, Any]]:
    """
    간단 요일 히트맵용: 최근 N주 동안 날짜별 달성률(완료/전체)
    """
    start = end_date - timedelta(days=weeks * 7 - 1)
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM habits WHERE is_active=1").fetchone()["c"]
        total = int(total) if total else 0

        out = []
        d = start
        while d <= end_date:
            ds = d.isoformat()
            completed = conn.execute("""
                SELECT COUNT(*) AS c
                FROM habit_logs hl
                JOIN habits h ON h.id=hl.habit_id
                WHERE hl.date=? AND hl.completed=1 AND h.is_active=1
            """, (ds,)).fetchone()["c"]
            completed = int(completed) if completed else 0
            rate = (completed / total) if total > 0 else 0.0
            out.append({
                "date": ds,
                "weekday": d.weekday(),  # 0=Mon
                "rate": rate,
                "completed": completed,
                "total": total,
            })
            d += timedelta(days=1)
        return out
