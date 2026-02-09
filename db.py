import sqlite3
from datetime import datetime, date, timedelta
import random
import string

DB_PATH = "habit_tracker.db"


def _now():
    return datetime.utcnow().isoformat(timespec="seconds")


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # --- core tables ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS habits(
        habit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        frequency TEXT NOT NULL CHECK (frequency IN ('daily','weekly')),
        goal INTEGER NOT NULL DEFAULT 1,
        reminder_text TEXT,
        created_at TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS checkins(
        checkin_id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL UNIQUE,
        note TEXT,
        created_at TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS checkin_items(
        checkin_id INTEGER NOT NULL,
        habit_id INTEGER NOT NULL,
        value INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(checkin_id, habit_id),
        FOREIGN KEY(checkin_id) REFERENCES checkins(checkin_id),
        FOREIGN KEY(habit_id) REFERENCES habits(habit_id)
    )
    """)

    # --- legacy coaching logs (kept if existed) ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS coaching_logs(
        coaching_id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        tone TEXT,
        weather_summary TEXT,
        input_summary TEXT,
        output_text TEXT,
        created_at TEXT NOT NULL
    )
    """)

    # --- NEW: coach_logs_v2 (type/model/content) ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS coach_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        type TEXT NOT NULL CHECK (type IN ('daily','weekly')),
        tone TEXT,
        model TEXT,
        weather_summary TEXT,
        input_summary TEXT,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    # --- NEW: dog collection + milestones ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dog_collection(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        habit_id INTEGER,
        image_url TEXT NOT NULL,
        rarity TEXT NOT NULL,
        earned_by TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dog_milestones(
        date TEXT NOT NULL,
        rate_bucket INTEGER NOT NULL,
        claimed INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        PRIMARY KEY(date, rate_bucket)
    )
    """)

    # --- NEW: smart scheduler helper indexes (optional) ---
    cur.execute("CREATE INDEX IF NOT EXISTS idx_checkins_date ON checkins(date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_items_habit ON checkin_items(habit_id)")

    # --- NEW: groups ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS groups(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS group_members(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id INTEGER NOT NULL,
        nickname TEXT NOT NULL,
        joined_at TEXT NOT NULL,
        UNIQUE(group_id, nickname),
        FOREIGN KEY(group_id) REFERENCES groups(id)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS group_streak_logs(
        group_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        achieved INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(group_id, date),
        FOREIGN KEY(group_id) REFERENCES groups(id)
    )
    """)

    conn.commit()
    conn.close()


# -------------------------
# seed
# -------------------------
def seed_sample_habits_if_empty():
    if list_habits():
        return
    create_habit("물 한 잔", "물 마시기", "daily", 1, "물 한 잔 어때요?")
    create_habit("스트레칭", "10분 스트레칭", "daily", 1, "가볍게 몸 풀기!")
    create_habit("운동", "주 3회 운동", "weekly", 3, "이번 주 운동 1회 채우기!")


# -------------------------
# habits CRUD
# -------------------------
def list_habits():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM habits ORDER BY habit_id ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_habit(name, description, frequency, goal, reminder_text):
    conn = get_conn()
    conn.execute(
        "INSERT INTO habits(name, description, frequency, goal, reminder_text, created_at) VALUES (?,?,?,?,?,?)",
        (name, description, frequency, int(goal), reminder_text, _now()),
    )
    conn.commit()
    conn.close()


def update_habit(habit_id, name, description, frequency, goal, reminder_text):
    conn = get_conn()
    conn.execute(
        """UPDATE habits SET name=?, description=?, frequency=?, goal=?, reminder_text=? WHERE habit_id=?""",
        (name, description, frequency, int(goal), reminder_text, int(habit_id)),
    )
    conn.commit()
    conn.close()


def delete_habit(habit_id):
    conn = get_conn()
    conn.execute("DELETE FROM habits WHERE habit_id=?", (int(habit_id),))
    conn.commit()
    conn.close()


# -------------------------
# checkins
# -------------------------
def upsert_checkin(date_str, note):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO checkins(date, note, created_at)
           VALUES (?,?,?)
           ON CONFLICT(date) DO UPDATE SET note=excluded.note
        """,
        (date_str, note, _now()),
    )
    # get id
    row = cur.execute("SELECT checkin_id FROM checkins WHERE date=?", (date_str,)).fetchone()
    conn.commit()
    conn.close()
    return int(row["checkin_id"])


def get_checkin(date_str):
    conn = get_conn()
    chk = conn.execute("SELECT * FROM checkins WHERE date=?", (date_str,)).fetchone()
    if not chk:
        conn.close()
        return None
    items = conn.execute(
        """SELECT i.habit_id, h.name, h.frequency, h.goal, i.value
           FROM checkin_items i
           JOIN habits h ON h.habit_id=i.habit_id
           WHERE i.checkin_id=?
           ORDER BY h.habit_id ASC
        """,
        (int(chk["checkin_id"]),),
    ).fetchall()
    conn.close()
    return {"checkin": dict(chk), "items": [dict(r) for r in items]}


def upsert_checkin_item(checkin_id, habit_id, value):
    conn = get_conn()
    conn.execute(
        """INSERT INTO checkin_items(checkin_id, habit_id, value)
           VALUES (?,?,?)
           ON CONFLICT(checkin_id, habit_id) DO UPDATE SET value=excluded.value
        """,
        (int(checkin_id), int(habit_id), int(value)),
    )
    conn.commit()
    conn.close()


def get_items_between(start_date_str, end_date_str):
    conn = get_conn()
    rows = conn.execute(
        """SELECT c.date, h.habit_id, h.name, h.frequency, h.goal, i.value
           FROM checkins c
           JOIN checkin_items i ON i.checkin_id=c.checkin_id
           JOIN habits h ON h.habit_id=i.habit_id
           WHERE c.date BETWEEN ? AND ?
           ORDER BY c.date ASC, h.habit_id ASC
        """,
        (start_date_str, end_date_str),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -------------------------
# coaching logs (legacy)
# -------------------------
def add_coaching_log(date_str, tone, weather_summary, input_summary, output_text):
    conn = get_conn()
    conn.execute(
        """INSERT INTO coaching_logs(date, tone, weather_summary, input_summary, output_text, created_at)
           VALUES (?,?,?,?,?,?)
        """,
        (date_str, tone, weather_summary, input_summary, output_text, _now()),
    )
    conn.commit()
    conn.close()


def list_coaching_logs(limit=200):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM coaching_logs ORDER BY coaching_id DESC LIMIT ?",
        (int(limit),),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -------------------------
# coaching logs v2 (NEW)
# -------------------------
def add_coaching_log_v2(date_str, coach_type, tone, model, weather_summary, input_summary, content):
    conn = get_conn()
    conn.execute(
        """INSERT INTO coach_logs(date, type, tone, model, weather_summary, input_summary, content, created_at)
           VALUES (?,?,?,?,?,?,?,?)
        """,
        (date_str, coach_type, tone, model, weather_summary, input_summary, content, _now()),
    )
    conn.commit()
    conn.close()


def list_coaching_logs_v2(limit=200):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM coach_logs ORDER BY id DESC LIMIT ?",
        (int(limit),),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -------------------------
# Smart Scheduler (NEW)
# -------------------------
def get_week_window(date_str):
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    monday = d - timedelta(days=d.weekday())  # Mon
    sunday = monday + timedelta(days=6)
    return monday.strftime("%Y-%m-%d"), sunday.strftime("%Y-%m-%d")


def get_habit_done_count_between(habit_id, start_date_str, end_date_str):
    conn = get_conn()
    row = conn.execute(
        """
        SELECT COUNT(*) AS c
        FROM checkins c
        JOIN checkin_items i ON i.checkin_id=c.checkin_id
        JOIN habits h ON h.habit_id=i.habit_id
        WHERE c.date BETWEEN ? AND ?
          AND h.habit_id=?
          AND i.value >= h.goal
        """,
        (start_date_str, end_date_str, int(habit_id)),
    ).fetchone()
    conn.close()
    return int(row["c"]) if row else 0


def recommend_habits(date_str, top_k=3):
    """
    규칙 기반 점수:
    - weekly: 압박도=(남은횟수)/(남은일수+0.5) + 최근7일 성공률(0~1)
    - daily: (미완료면 1.0 완료면 0.1) + 최근7일 미완료빈도(0~1)
    """
    hs = list_habits()
    if not hs:
        return []

    # 오늘 완료 여부
    chk = get_checkin(date_str)
    done_today = set()
    if chk:
        for it in chk["items"]:
            if int(it["value"]) >= int(it["goal"]):
                done_today.add(int(it["habit_id"]))

    # 최근 7일 성공률 계산용 범위
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    start_7 = (d - timedelta(days=6)).strftime("%Y-%m-%d")
    items_7 = get_items_between(start_7, date_str)

    # habit_id별 7일 성공/총일 집계(간단)
    by = {}
    for it in items_7:
        hid = int(it["habit_id"])
        ok = int(it["value"]) >= int(it["goal"])
        by.setdefault(hid, {"succ": 0, "tot": 0})
        by[hid]["tot"] += 1
        if ok:
            by[hid]["succ"] += 1

    week_start, week_end = get_week_window(date_str)
    week_end_date = datetime.strptime(week_end, "%Y-%m-%d").date()
    today_date = d
    remaining_days = (week_end_date - today_date).days + 1  # 오늘 포함

    recs = []
    for h in hs:
        hid = int(h["habit_id"])
        freq = h["frequency"]
        goal = max(1, int(h["goal"]))

        succ = by.get(hid, {}).get("succ", 0)
        tot = by.get(hid, {}).get("tot", 0)
        recent_rate = (succ / tot) if tot > 0 else 0.0

        if freq == "weekly":
            done = get_habit_done_count_between(hid, week_start, week_end)
            remaining = max(0, goal - done)
            pressure = (remaining / (remaining_days + 0.5)) if remaining > 0 else 0.0
            score = pressure + recent_rate
            reason = f"이번 주 목표까지 {remaining}회 남았고 남은 날이 {remaining_days}일이에요."
            progress = f"이번 주 진행: {done}/{goal}회"
            recs.append({**h, "score": score, "reason": reason, "progress_text": progress})
        else:
            # daily
            base = 0.1 if hid in done_today else 1.0
            miss_rate = 1.0 - recent_rate
            score = base + miss_rate
            reason = "오늘 미완료라면 지금 짧게 처리해두면 좋아요." if hid not in done_today else "이미 완료했지만 유지가 중요해요."
            recs.append({**h, "score": score, "reason": reason, "progress_text": ""})

    recs.sort(key=lambda x: x["score"], reverse=True)
    return recs[: int(top_k)]


# -------------------------
# Dog Collection (NEW)
# -------------------------
def add_dog_to_collection(date_str, habit_id, image_url, rarity, earned_by):
    conn = get_conn()
    conn.execute(
        """INSERT INTO dog_collection(date, habit_id, image_url, rarity, earned_by, created_at)
           VALUES (?,?,?,?,?,?)
        """,
        (date_str, habit_id, image_url, rarity, earned_by, _now()),
    )
    conn.commit()
    conn.close()


def list_dog_collection(date_from=None, date_to=None, rarity=None):
    conn = get_conn()
    q = "SELECT * FROM dog_collection WHERE 1=1"
    params = []
    if date_from:
        q += " AND date >= ?"
        params.append(date_from)
    if date_to:
        q += " AND date <= ?"
        params.append(date_to)
    if rarity:
        q += " AND rarity = ?"
        params.append(rarity)
    q += " ORDER BY id DESC"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_claimed_buckets(date_str):
    conn = get_conn()
    rows = conn.execute(
        "SELECT rate_bucket FROM dog_milestones WHERE date=? AND claimed=1",
        (date_str,),
    ).fetchall()
    conn.close()
    return set(int(r["rate_bucket"]) for r in rows)


def claim_milestone_if_needed(date_str, bucket):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO dog_milestones(date, rate_bucket, claimed, created_at)
               VALUES (?,?,1,?)
            """,
            (date_str, int(bucket), _now()),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


# -------------------------
# Groups (NEW) - MVP (닉네임 기반)
# -------------------------
def _gen_group_code(n=8):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


def create_group(name):
    conn = get_conn()
    code = _gen_group_code(8)
    # retry in rare collision
    for _ in range(5):
        try:
            conn.execute(
                "INSERT INTO groups(group_code, name, created_at) VALUES (?,?,?)",
                (code, name, _now()),
            )
            conn.commit()
            conn.close()
            return code
        except sqlite3.IntegrityError:
            code = _gen_group_code(8)

    conn.close()
    raise RuntimeError("그룹 코드 생성에 실패했습니다.")


def get_group_by_code(group_code):
    conn = get_conn()
    row = conn.execute("SELECT * FROM groups WHERE group_code=?", (group_code,)).fetchone()
    conn.close()
    if not row:
        raise ValueError("존재하지 않는 그룹 코드입니다.")
    return dict(row)


def join_group(group_code, nickname):
    g = get_group_by_code(group_code)
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO group_members(group_id, nickname, joined_at) VALUES (?,?,?)",
            (int(g["id"]), nickname, _now()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # already joined
        pass
    finally:
        conn.close()


def list_groups_for_nickname(nickname):
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT g.*
        FROM groups g
        JOIN group_members m ON m.group_id=g.id
        WHERE m.nickname=?
        ORDER BY g.id DESC
        """,
        (nickname,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_group_members(group_code):
    g = get_group_by_code(group_code)
    conn = get_conn()
    rows = conn.execute(
        "SELECT nickname, joined_at FROM group_members WHERE group_id=? ORDER BY joined_at ASC",
        (int(g["id"]),),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def compute_member_today_achieved(nickname, date_str):
    """
    MVP: 이 DB에 해당 날짜 체크인이 있고,
    그 체크인에서 '성공(값>=goal)'한 습관이 1개 이상이면 achieved=True,
    체크인이 있지만 성공 0이면 False,
    체크인 자체가 없으면 None(데이터 없음)
    """
    chk = get_checkin(date_str)
    if not chk:
        return None

    # 이 앱은 단일 체크인 구조라 "닉네임별 분리"는 없음.
    # 같은 서버에서 여러 사용자가 서로 다른 세션으로 쓰면,
    # 현실적으로는 '서로 다른 DB'를 쓰지 않는 이상 충돌 가능.
    # MVP에서는 "현재 인스턴스에서 기록된 데이터"를 기반으로만 판단.
    # 즉, achieved는 '데이터 존재/성공 여부'만 판단
    for it in chk["items"]:
        if int(it["value"]) >= int(it["goal"]):
            return True
    return False


def update_group_daily_status(group_id, date_str):
    """
    group achieved 정의:
    - 멤버 전원이 achieved=True 일 때만 achieved=1
    - 멤버 중 None(데이터없음) 또는 False가 있으면 achieved=0
    """
    conn = get_conn()
    members = conn.execute(
        "SELECT nickname FROM group_members WHERE group_id=?",
        (int(group_id),),
    ).fetchall()
    conn.close()

    if not members:
        achieved = 0
    else:
        flags = []
        for m in members:
            flags.append(compute_member_today_achieved(m["nickname"], date_str))
        achieved = 1 if all(f is True for f in flags) else 0

    conn = get_conn()
    conn.execute(
        """INSERT INTO group_streak_logs(group_id, date, achieved, created_at)
           VALUES (?,?,?,?)
           ON CONFLICT(group_id, date) DO UPDATE SET achieved=excluded.achieved
        """,
        (int(group_id), date_str, int(achieved), _now()),
    )
    conn.commit()
    conn.close()


def list_group_streak_logs(group_id, date_from=None):
    conn = get_conn()
    q = "SELECT date, achieved, created_at FROM group_streak_logs WHERE group_id=?"
    params = [int(group_id)]
    if date_from:
        q += " AND date >= ?"
        params.append(date_from)
    q += " ORDER BY date DESC"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def calc_group_streak(group_id):
    """
    오늘 기준으로 achieved=1이 연속인 일수
    """
    conn = get_conn()
    rows = conn.execute(
        "SELECT date, achieved FROM group_streak_logs WHERE group_id=? ORDER BY date DESC",
        (int(group_id),),
    ).fetchall()
    conn.close()

    if not rows:
        return 0

    streak = 0
    prev = None
    for r in rows:
        d = datetime.strptime(r["date"], "%Y-%m-%d").date()
        if prev is None:
            prev = d
        else:
            if (prev - d).days != 1:
                break
            prev = d

        if int(r["achieved"]) == 1:
            streak += 1
        else:
            break
    return streak


def update_groups_for_member_on_date(nickname, date_str):
    """
    체크인 저장 시 호출:
    - 내가 속한 그룹들에 대해 오늘 status를 갱신
    """
    groups = list_groups_for_nickname(nickname)
    for g in groups:
        update_group_daily_status(int(g["id"]), date_str)
