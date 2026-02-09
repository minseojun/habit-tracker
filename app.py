import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime
import requests

import db

# OpenAI (없으면 코칭 기능 비활성화)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="AI Habit Tracker", page_icon="✅", layout="wide")


# =========================================================
# Secrets / Sidebar input
# =========================================================
def get_secret_or_sidebar(key_name: str, label: str, password: bool = True) -> str:
    if key_name in st.secrets and st.secrets[key_name]:
        return str(st.secrets[key_name])

    ss_key = f"__{key_name}"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = ""
    t = st.sidebar.text_input(label, value=st.session_state[ss_key], type="password" if password else "default")
    st.session_state[ss_key] = t
    return t


# =========================================================
# Inlined utils.stats / utils.streaks equivalents
# =========================================================
def items_to_dataframe(items):
    if not items:
        return pd.DataFrame()
    return pd.DataFrame(items)


def compute_today_achievement(habits, today_values: dict):
    """
    returns: (rate_percent, success_count, total_count)
    success: value >= goal
    """
    if not habits:
        return 0.0, 0, 0
    total = 0
    success = 0
    for h in habits:
        total += 1
        hid = int(h["habit_id"])
        goal = int(h["goal"])
        v = int(today_values.get(hid, 0))
        if v >= goal:
            success += 1
    rate = (success / total * 100.0) if total > 0 else 0.0
    return rate, success, total


def build_seven_day_summary(items_7d):
    """
    아주 단순한 7일 요약(사람이 보기 좋게)
    """
    if not items_7d:
        return "최근 7일 데이터가 없어요."
    df = pd.DataFrame(items_7d)
    if df.empty:
        return "최근 7일 데이터가 없어요."

    df["success"] = df["value"].astype(int) >= df["goal"].astype(int)
    lines = []
    lines.append("### 최근 7일 요약")
    daily = df.groupby("date")["success"].mean().reset_index()
    daily["rate"] = (daily["success"] * 100).round(0).astype(int)
    lines.append("- 일자별 달성률:")
    for _, r in daily.iterrows():
        lines.append(f"  - {r['date']}: {r['rate']}%")
    by_habit = df.groupby("name")["success"].mean().reset_index()
    by_habit["rate"] = (by_habit["success"] * 100).round(0).astype(int)
    lines.append("- 습관별 평균 달성률:")
    for _, r in by_habit.sort_values("rate", ascending=False).iterrows():
        lines.append(f"  - {r['name']}: {r['rate']}%")
    return "\n".join(lines)


def compute_daily_streak(items, habit_id: int, goal: int, end_date_str: str):
    """
    특정 daily 습관이 end_date 기준으로 연속 성공한 일수.
    items: db.get_items_between 결과(list of dict)
    """
    # date -> success mapping for this habit
    end_d = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    m = {}
    for it in items:
        if int(it["habit_id"]) != int(habit_id):
            continue
        d = it["date"]
        ok = int(it["value"]) >= int(it["goal"])
        # 같은 날짜가 중복되면 success면 True 유지
        m[d] = m.get(d, False) or ok

    streak = 0
    cur = end_d
    while True:
        ds = cur.strftime("%Y-%m-%d")
        if m.get(ds, False):
            streak += 1
            cur = cur - timedelta(days=1)
        else:
            break
    return streak


# =========================================================
# Weather (OpenWeatherMap)
# =========================================================
def fetch_current_weather(city: str, api_key: str):
    if not api_key or not city:
        return None
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric", "lang": "kr"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def weather_to_summary(weather):
    if not weather:
        return ""
    main = weather.get("main", {})
    w0 = (weather.get("weather") or [{}])[0]
    desc = w0.get("description", "")
    temp = main.get("temp")
    feels = main.get("feels_like")
    return f"{desc} / {temp}°C (체감 {feels}°C)"


def simple_weather_hint(weather):
    if not weather:
        return None
    w0 = (weather.get("weather") or [{}])[0]
    desc = (w0.get("main") or "") + " " + (w0.get("description") or "")
    d = desc.lower()
    if "rain" in d or "비" in d:
        return "비 오는 날엔 실내 습관(스트레칭/정리)로 가볍게 가보세요."
    if "snow" in d or "눈" in d:
        return "눈/추위가 있으면 무리하지 말고 실내 루틴을 추천해요."
    if "clear" in d or "맑" in d:
        return "날씨가 좋아요! 짧은 산책 같은 야외 습관을 붙여보세요."
    if "cloud" in d or "구름" in d:
        return "구름 낀 날엔 집중 루틴(25분)로 컨디션을 끌어올려봐요."
    return None


@st.cache_data(ttl=600)
def cached_weather(city: str, api_key: str):
    return fetch_current_weather(city=city, api_key=api_key)


# =========================================================
# Dog API
# =========================================================
def fetch_random_dog_images(n: int = 1):
    n = max(1, int(n))
    url = "https://dog.ceo/api/breeds/image/random"
    out = []
    for _ in range(n):
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "success" and data.get("message"):
            out.append(data["message"])
    return out


@st.cache_data(ttl=60)
def cached_dogs(n: int):
    return fetch_random_dog_images(n=n)


# =========================================================
# Coach (OpenAI)
# =========================================================
TONES = ["친근하게", "차분하게", "엄격하게", "유쾌하게"]

SYSTEM_COACH = """너는 습관 코치다. 한국어로 답한다.
규칙:
- 120~220자
- (칭찬 1) + (개선 제안 1) + (오늘 할 행동 1) 포함
- 과장 금지, 실행 가능하게
"""


def generate_coaching(
    api_key: str,
    model: str,
    tone: str,
    date_str: str,
    city: str,
    weather_summary: str,
    today_items,
    seven_day_summary: str,
    note: str,
):
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다. requirements.txt에 openai를 추가하세요.")
    client = OpenAI(api_key=api_key)

    items_lines = []
    for it in today_items:
        items_lines.append(f"- {it['name']}: {it['value']}/{it['goal']} ({it['frequency']})")

    user_prompt = f"""
날짜: {date_str}
도시: {city}
날씨: {weather_summary}
코칭 톤: {tone}

오늘 체크:
{chr(10).join(items_lines) if items_lines else "- (없음)"}

오늘 메모:
{note or "-"}

최근 7일 요약:
{seven_day_summary}

요청:
규칙을 지키며 오늘의 코칭 메시지를 작성해줘.
"""

    resp = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": SYSTEM_COACH},
            {"role": "user", "content": user_prompt},
        ],
    )
    out = (resp.choices[0].message.content or "").strip()
    return out, user_prompt


# =========================================================
# Boot
# =========================================================
def ensure_seed():
    db.init_db()
    if hasattr(db, "seed_sample_habits_if_empty"):
        db.seed_sample_habits_if_empty()


ensure_seed()

# Sidebar
st.sidebar.title("AI Habit Tracker")

nickname = st.sidebar.text_input("닉네임(그룹용)", value=st.session_state.get("nickname", "guest"))
st.session_state["nickname"] = nickname.strip() if nickname.strip() else "guest"

city = st.sidebar.text_input("도시 (기본: Seoul)", value=st.session_state.get("city", "Seoul"))
st.session_state["city"] = city

tone = st.sidebar.selectbox("코칭 톤", options=TONES, index=TONES.index(st.session_state.get("tone", TONES[0])))
st.session_state["tone"] = tone

openai_key = get_secret_or_sidebar("OPENAI_API_KEY", "OpenAI API Key")
owm_key = get_secret_or_sidebar("OPENWEATHER_API_KEY", "OpenWeatherMap API Key")

menu = st.sidebar.radio(
    "메뉴",
    options=["오늘 체크인", "습관 관리", "대시보드/통계", "AI 코칭 기록"],
)

st.sidebar.divider()
with st.sidebar.expander("고급 설정"):
    model = st.text_input("OpenAI 모델", value=st.session_state.get("model", "gpt-4o-mini"))
    st.session_state["model"] = model
    if st.button("캐시 초기화(날씨/강아지)"):
        st.cache_data.clear()
        st.success("초기화 완료!")


# =========================================================
# Pages
# =========================================================
def page_habits():
    st.header("습관 관리")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("습관 추가")
        with st.form("add_habit_form", clear_on_submit=True):
            name = st.text_input("이름", placeholder="예: 물 8잔 마시기")
            description = st.text_area("설명(선택)", height=80)
            frequency = st.selectbox("주기", options=["daily", "weekly"])
            goal = st.number_input("목표(goal, 정수)", min_value=1, value=1, step=1)
            reminder_text = st.text_input("알림 메시지(선택)", placeholder="예: 지금 물 한 잔!")
            submitted = st.form_submit_button("추가")
            if submitted:
                if not name.strip():
                    st.error("이름(name)은 필수입니다.")
                else:
                    db.create_habit(name.strip(), description, frequency, int(goal), reminder_text)
                    st.success("습관을 추가했어요.")
                    st.rerun()

    with col2:
        st.subheader("기존 습관")
        hs = db.list_habits()
        if not hs:
            st.info("아직 습관이 없어요. 왼쪽에서 추가해보세요.")
            return

        for h in hs:
            with st.expander(f"#{h['habit_id']} • {h['name']} ({h['frequency']}, goal={h['goal']})", expanded=False):
                st.caption(f"created_at: {h['created_at']}")
                st.write(h.get("description") or "_설명 없음_")
                st.write(f"알림: {h.get('reminder_text') or '-'}")

                with st.form(f"edit_habit_{h['habit_id']}"):
                    name = st.text_input("이름", value=h["name"], key=f"n_{h['habit_id']}")
                    description = st.text_area("설명", value=h.get("description") or "", height=80, key=f"d_{h['habit_id']}")
                    frequency = st.selectbox(
                        "주기", options=["daily", "weekly"], index=["daily", "weekly"].index(h["frequency"]), key=f"f_{h['habit_id']}"
                    )
                    goal = st.number_input("목표(goal)", min_value=1, value=int(h["goal"]), step=1, key=f"g_{h['habit_id']}")
                    reminder_text = st.text_input("알림 메시지", value=h.get("reminder_text") or "", key=f"r_{h['habit_id']}")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.form_submit_button("수정 저장"):
                            db.update_habit(int(h["habit_id"]), name.strip(), description, frequency, int(goal), reminder_text)
                            st.success("수정했어요.")
                            st.rerun()
                    with c2:
                        if st.form_submit_button("삭제", type="primary"):
                            db.delete_habit(int(h["habit_id"]))
                            st.success("삭제했어요.")
                            st.rerun()


def page_today():
    st.header("오늘 체크인")

    default_date = st.session_state.get("selected_date", date.today())
    selected_date = st.date_input("날짜 선택", value=default_date)
    st.session_state["selected_date"] = selected_date
    date_str = selected_date.strftime("%Y-%m-%d")

    # weather
    weather = None
    try:
        if owm_key:
            weather = cached_weather(city, owm_key)
    except Exception as e:
        st.warning(f"날씨 정보를 불러오지 못했어요: {e}")
        weather = None

    weather_summary = weather_to_summary(weather)
    weather_hint = simple_weather_hint(weather)

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("오늘의 날씨")
        if weather:
            st.write(f"**{city}**")
            st.write(weather_summary)
            if weather_hint:
                st.info(weather_hint)
        else:
            st.info("날씨 정보 없음 (API Key가 없거나 호출 실패)")

    existing = db.get_checkin(date_str)
    existing_note = existing["checkin"].get("note") if existing else ""
    existing_items = {int(it["habit_id"]): int(it["value"]) for it in (existing["items"] if existing else [])}

    with right:
        st.subheader("습관 체크인")
        hs = db.list_habits()
        if not hs:
            st.warning("습관이 없습니다. 먼저 '습관 관리'에서 습관을 추가하세요.")
            return

        with st.form("checkin_form"):
            values = {}
            for h in hs:
                hid = int(h["habit_id"])
                goal = int(h["goal"])
                if goal == 1:
                    checked = existing_items.get(hid, 0) >= 1
                    v = st.checkbox(f"{h['name']} (goal=1)", value=checked, key=f"chk_{date_str}_{hid}")
                    values[hid] = 1 if v else 0
                else:
                    v = st.number_input(
                        f"{h['name']} (목표 {goal})",
                        min_value=0,
                        value=int(existing_items.get(hid, 0)),
                        step=1,
                        key=f"num_{date_str}_{hid}",
                    )
                    values[hid] = int(v)

            note = st.text_area("오늘 메모(선택)", value=existing_note or "", height=100)
            saved = st.form_submit_button("저장")

        if saved:
            try:
                checkin_id = db.upsert_checkin(date_str, note)
                for hid, v in values.items():
                    db.upsert_checkin_item(checkin_id, hid, int(v))
                st.success("오늘 체크인을 저장했어요.")
                st.rerun()
            except Exception as e:
                st.error(f"저장 중 오류: {e}")

    st.divider()
    st.subheader("오늘 요약")

    fresh = db.get_checkin(date_str)
    today_values = {}
    today_items_for_ai = []
    hs = db.list_habits()
    if fresh:
        for it in fresh["items"]:
            hid = int(it["habit_id"])
            today_values[hid] = int(it["value"])
            today_items_for_ai.append(
                {"name": it["name"], "goal": int(it["goal"]), "value": int(it["value"]), "frequency": it["frequency"]}
            )

    rate, success_count, total_count = compute_today_achievement(hs, today_values)
    st.write(f"- 달성률: **{rate:.0f}%** ({success_count}/{total_count})")

    start_30 = (selected_date - timedelta(days=60)).strftime("%Y-%m-%d")
    items_60d = db.get_items_between(start_30, date_str)
    streak_rows = []
    for h in hs:
        if h["frequency"] != "daily":
            continue
        s = compute_daily_streak(items_60d, int(h["habit_id"]), int(h["goal"]), date_str)
        streak_rows.append((h["name"], s))
    streak_rows.sort(key=lambda x: x[1], reverse=True)
    top3 = streak_rows[:3]
    if top3:
        st.write("**streak TOP 3 (daily)**")
        for name, s in top3:
            st.write(f"- {name}: {s}일 연속")

    # ✅ 강아지 보상: 체크인 저장 + 성공 1개 이상일 때만 노출
    st.divider()
    st.subheader("오늘의 보상 🐶")
    if total_count == 0:
        st.info("습관이 없어서 보상을 계산할 수 없어요.")
    elif not fresh:
        st.info("체크인을 저장하면 강아지 보상이 열려요!")
    elif success_count <= 0:
        st.info("습관을 1개 이상 목표 달성하면 강아지 보상이 나타나요!")
    else:
        try:
            if rate >= 100:
                st.success("퍼펙트! 100% 달성 🎉🎉")
                urls = cached_dogs(2)
                cols = st.columns(2)
                for i, u in enumerate(urls[:2]):
                    with cols[i]:
                        st.image(u, use_container_width=True)
            elif rate >= 70:
                st.success("좋아요! 70% 이상 달성 🎉")
                urls = cached_dogs(1)
                if urls:
                    st.image(urls[0], use_container_width=True)
            else:
                st.success("좋아요! 목표 달성한 습관이 있어요 🧡")
                urls = cached_dogs(1)
                if urls:
                    st.image(urls[0], use_container_width=True)
        except Exception as e:
            st.warning(f"Dog API 호출 실패: {e}")

    # AI coaching
    st.divider()
    st.subheader("AI 코칭")

    start_7 = (selected_date - timedelta(days=6)).strftime("%Y-%m-%d")
    items_7d = db.get_items_between(start_7, date_str)
    seven_day_summary = build_seven_day_summary(items_7d)

    with st.expander("최근 7일 요약 보기", expanded=False):
        st.markdown(seven_day_summary)

    can_generate = bool(openai_key) and bool(fresh) and bool(today_items_for_ai)
    c1, c2 = st.columns([1, 1])
    with c1:
        gen = st.button("AI 코칭 생성", disabled=not can_generate, type="primary")
    with c2:
        regen = st.button("코칭 다시 생성", disabled=not can_generate)

    if (gen or regen) and not openai_key:
        st.error("OpenAI API Key가 필요해요.")
        return

    if (gen or regen) and not can_generate:
        st.warning("코칭을 생성하려면 먼저 오늘 체크인을 저장해 주세요.")
        return

    if gen or regen:
        try:
            output, input_summary = generate_coaching(
                api_key=openai_key,
                model=st.session_state.get("model", "gpt-4o-mini"),
                tone=tone,
                date_str=date_str,
                city=city,
                weather_summary=weather_summary,
                today_items=today_items_for_ai,
                seven_day_summary=seven_day_summary,
                note=fresh["checkin"].get("note") if fresh else "",
            )
            db.add_coaching_log(date_str, tone, weather_summary, input_summary, output)
            st.markdown(output)
        except Exception as e:
            st.error(f"코칭 생성 실패: {e}")


def page_dashboard():
    st.header("대시보드 / 통계")

    hs = db.list_habits()
    if not hs:
        st.warning("습관이 없습니다. 먼저 '습관 관리'에서 습관을 추가하세요.")
        return

    preset = st.selectbox("기간", options=["최근 7일", "최근 30일", "커스텀"], index=0)
    today_ = date.today()
    if preset == "최근 7일":
        start = today_ - timedelta(days=6)
        end = today_
    elif preset == "최근 30일":
        start = today_ - timedelta(days=29)
        end = today_
    else:
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("시작일", value=today_ - timedelta(days=29), key="dash_start")
        with c2:
            end = st.date_input("종료일", value=today_, key="dash_end")

    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    items = db.get_items_between(start_s, end_s)
    df = items_to_dataframe(items)

    if df.empty:
        st.info("선택한 기간에 데이터가 없어요.")
        return

    df["success"] = df["value"].astype(int) >= df["goal"].astype(int)

    st.subheader("전체 달성률 추이")
    daily = df.groupby("date")["success"].mean().reset_index()
    daily["success_rate"] = daily["success"] * 100.0
    daily = daily.drop(columns=["success"])
    st.line_chart(daily.set_index("date"))

    st.subheader("습관별 달성률")
    by_habit = df.groupby("name")["success"].mean().reset_index()
    by_habit["success_rate"] = by_habit["success"] * 100.0
    st.bar_chart(by_habit.set_index("name")[["success_rate"]])


def page_logs():
    st.header("AI 코칭 기록")

    logs = db.list_coaching_logs(limit=200)
    if not logs:
        st.info("아직 코칭 기록이 없어요.")
        return

    options = [f"{l['date']} | {l.get('tone','-')} | #{l['coaching_id']}" for l in logs]
    idx = st.selectbox("기록 선택", options=list(range(len(options))), format_func=lambda i: options[i])
    selected = logs[idx]

    st.subheader(f"{selected['date']} • {selected.get('tone','-')}")
    st.caption(f"created_at: {selected['created_at']}")
    if selected.get("weather_summary"):
        st.write(f"날씨: {selected['weather_summary']}")

    chk = db.get_checkin(selected["date"])
    if chk:
        st.write("**체크인 메모**")
        st.write(chk["checkin"].get("note") or "-")
        st.write("**체크인 항목**")
        df = pd.DataFrame(chk["items"])
        if not df.empty:
            st.dataframe(df[["name", "goal", "value", "frequency"]], use_container_width=True)

    st.divider()
    st.markdown(selected["output_text"])

    st.divider()
    export_df = pd.DataFrame(logs)
    st.download_button(
        "코칭 로그 CSV 다운로드",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="coaching_logs.csv",
        mime="text/csv",
    )


# Router
if menu == "습관 관리":
    page_habits()
elif menu == "대시보드/통계":
    page_dashboard()
elif menu == "AI 코칭 기록":
    page_logs()
else:
    page_today()
