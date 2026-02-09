import os
import sys
import streamlit as st
import pandas as pd
from datetime import date, timedelta

# ✅ 현재 파일(app.py) 폴더를 sys.path에 강제로 추가 (Streamlit Cloud에서 확실하게 잡힘)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import db

# ✅ services 모듈 import (폴더/패키지 인식 문제 대비)
try:
    from services.weather import fetch_current_weather, weather_to_summary, simple_weather_hint
    from services.dog import fetch_random_dog_images
    from services.coach import generate_coaching, TONES
except ModuleNotFoundError:
    # fallback: services 폴더를 직접 path에 추가
    SERVICES_DIR = os.path.join(BASE_DIR, "services")
    if SERVICES_DIR not in sys.path:
        sys.path.insert(0, SERVICES_DIR)
    from weather import fetch_current_weather, weather_to_summary, simple_weather_hint
    from dog import fetch_random_dog_images
    from coach import generate_coaching, TONES

from utils.stats import build_seven_day_summary, compute_today_achievement, items_to_dataframe
from utils.streaks import compute_daily_streak


# ---------- Helpers ----------
def get_secret_or_sidebar(key_name: str, label: str, password: bool = True) -> str:
    # 1) secrets
    if key_name in st.secrets and st.secrets[key_name]:
        return str(st.secrets[key_name])
    # 2) session state
    ss_key = f"__{key_name}"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = ""
    # 3) sidebar input
    t = st.sidebar.text_input(label, value=st.session_state[ss_key], type="password" if password else "default")
    st.session_state[ss_key] = t
    return t


@st.cache_data(ttl=600)
def cached_weather(city: str, api_key: str):
    return fetch_current_weather(city=city, api_key=api_key)


@st.cache_data(ttl=60)
def cached_dogs(n: int):
    return fetch_random_dog_images(n=n)


def ensure_seed():
    db.init_db()
    db.seed_sample_habits_if_empty()


def _date_str(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _milestone_buckets_for_rate(rate: float):
    # rate: 0~100
    buckets = []
    if rate >= 20:
        buckets.append(20)
    if rate >= 50:
        buckets.append(50)
    if rate >= 80:
        buckets.append(80)
    if rate >= 100:
        buckets.append(100)
    return buckets


def _rarity_for_bucket(bucket: int) -> str:
    if bucket >= 100:
        return "epic"
    if bucket >= 80:
        return "rare"
    if bucket >= 50:
        return "common_or_rare"
    return "common"


# ---------- UI: Sidebar ----------
ensure_seed()

st.sidebar.title("AI Habit Tracker")

nickname = st.sidebar.text_input("닉네임(그룹용)", value=st.session_state.get("nickname", "guest"))
st.session_state["nickname"] = nickname.strip() if nickname.strip() else "guest"

city = st.sidebar.text_input("도시 (기본: Seoul)", value=st.session_state.get("city", "Seoul"))
st.session_state["city"] = city

tone = st.sidebar.selectbox("코칭 톤", options=TONES, index=TONES.index(st.session_state.get("tone", TONES[0])))
st.session_state["tone"] = tone

openai_key = get_secret_or_sidebar("OPENAI_API_KEY", "OpenAI API Key")
owm_key = get_secret_or_sidebar("OPENWEATHER_API_KEY", "OpenWeatherMap API Key")

storage = st.sidebar.radio("저장소", options=["sqlite3 (default)", "json (옵션-미구현)"], index=0)
if storage != "sqlite3 (default)":
    st.sidebar.warning("json 저장소는 옵션이며 현재 예시는 sqlite3만 구현되어 있어요.")

menu = st.sidebar.radio(
    "메뉴",
    options=[
        "오늘 체크인",
        "습관 관리",
        "대시보드/통계",
        "AI 코칭 기록",
        "🐶 도감",
        "👥 그룹(함께 streak)",
    ],
)

st.sidebar.divider()
with st.sidebar.expander("고급 설정"):
    model = st.text_input("OpenAI 모델", value=st.session_state.get("model", "gpt-4o-mini"))
    st.session_state["model"] = model

    if st.button("날씨 캐시 초기화"):
        st.cache_data.clear()
        st.success("캐시 초기화 완료!")


# ---------- Data ----------
habits = db.list_habits()


# ---------- Page: Habits Management ----------
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
        if not habits:
            st.info("아직 습관이 없어요. 왼쪽에서 추가해보세요.")
            return

        for h in habits:
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


# ---------- Smart Scheduler ----------
def render_recommendations(selected_date: date):
    st.subheader("오늘의 추천 TOP 3 🎯")
    if not habits:
        st.info("습관이 없어서 추천을 만들 수 없어요. 먼저 습관을 추가해 주세요.")
        return

    recs = db.recommend_habits(_date_str(selected_date), top_k=3)
    if not recs:
        st.info("추천할 항목이 아직 없어요. 오늘 체크인을 저장해보세요.")
        return

    cols = st.columns(3)
    for i, r in enumerate(recs[:3]):
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"**{r['name']}**")
                st.caption(f"{r['frequency']} · goal={r['goal']}")
                if r.get("progress_text"):
                    st.write(r["progress_text"])
                st.info(r.get("reason", "오늘 해두면 좋아요."))

                # "바로 체크하기": 오늘 체크인에 이 습관을 goal만큼 채우는 최소 구현
                if st.button("바로 체크하기", key=f"quick_{selected_date}_{r['habit_id']}"):
                    d = _date_str(selected_date)
                    chk = db.get_checkin(d)
                    checkin_id = db.upsert_checkin(d, chk["checkin"].get("note") if chk else "")
                    # weekly/daily 모두 value를 goal로 채워 '성공' 처리
                    db.upsert_checkin_item(checkin_id, int(r["habit_id"]), int(r["goal"]))
                    st.success("체크 완료! (추천에서 바로 반영)")
                    st.rerun()


# ---------- Dog Collection (Album) ----------
def maybe_award_milestones(date_str: str, rate: float, last_checked_habit_id: int | None):
    """
    체크 이벤트 직후 호출:
    - 달성률 버킷(20/50/80/100) 신규 도달 시 도감에 1장 저장
    - Dog API는 1분 캐시. 동일 이벤트에서 추가 호출 최소화
    """
    if not habits:
        return

    buckets = _milestone_buckets_for_rate(rate)
    if not buckets:
        return

    claimed = db.get_claimed_buckets(date_str)
    new_buckets = [b for b in buckets if b not in claimed]
    if not new_buckets:
        return

    # 버킷별 1장씩 주되, 한 번의 이벤트에서 최대 1~2장 정도만 (MVP)
    # 우선 "가장 높은 신규 버킷 1개"만 지급
    bucket = max(new_buckets)

    ok = db.claim_milestone_if_needed(date_str, bucket)
    if not ok:
        return

    # 이미지 1장만 호출
    urls = cached_dogs(1)
    if not urls:
        return

    rarity = _rarity_for_bucket(bucket)
    db.add_dog_to_collection(
        date_str=date_str,
        habit_id=last_checked_habit_id,
        image_url=urls[0],
        rarity=rarity,
        earned_by="milestone",
    )
    st.toast(f"신규 도감 획득! ({bucket}% 달성)", icon="🐶")


# ---------- Page: Today Check-in ----------
def page_today():
    st.header("오늘 체크인")

    # date selection
    default_date = st.session_state.get("selected_date", date.today())
    selected_date = st.date_input("날짜 선택", value=default_date)
    st.session_state["selected_date"] = selected_date
    date_str = _date_str(selected_date)

    # ✅ 스마트 스케줄러: KPI 아래 추천 TOP3
    render_recommendations(selected_date)

    st.divider()

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

    # load existing checkin (if any)
    existing = db.get_checkin(date_str)
    existing_note = existing["checkin"].get("note") if existing else ""
    existing_items = {int(it["habit_id"]): int(it["value"]) for it in (existing["items"] if existing else [])}

    with right:
        st.subheader("습관 체크인")
        if not habits:
            st.warning("습관이 없습니다. 먼저 '습관 관리'에서 습관을 추가하세요.")
            return

        last_checked_habit_id = None

        with st.form("checkin_form"):
            values = {}
            for h in habits:
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

                # 저장 + 마지막으로 "성공된 습관"을 찾아서 보상 연결(선택)
                for hid, v in values.items():
                    db.upsert_checkin_item(checkin_id, hid, int(v))
                    # 마지막으로 goal을 만족한 항목을 last로 기록
                    h = next((x for x in habits if int(x["habit_id"]) == int(hid)), None)
                    if h and int(v) >= int(h["goal"]):
                        last_checked_habit_id = int(hid)

                st.success("오늘 체크인을 저장했어요.")
                st.session_state["last_saved_date"] = date_str

                # 그룹 streak 업데이트(가능하면)
                db.update_groups_for_member_on_date(nickname=st.session_state["nickname"], date_str=date_str)

                st.rerun()
            except Exception as e:
                st.error(f"저장 중 오류: {e}")

    # summary + streaks + dog reward + coaching
    st.divider()
    st.subheader("오늘 요약")

    # compute today values from current DB (fresh)
    fresh = db.get_checkin(date_str)
    today_values = {}
    today_items_for_ai = []
    if fresh:
        for it in fresh["items"]:
            hid = int(it["habit_id"])
            today_values[hid] = int(it["value"])
            today_items_for_ai.append(
                {"name": it["name"], "goal": int(it["goal"]), "value": int(it["value"]), "frequency": it["frequency"]}
            )

    rate, success_count, total_count = compute_today_achievement(habits, today_values)
    st.write(f"- 달성률: **{rate:.0f}%** ({success_count}/{total_count})")

    # streak top 3 (daily only)
    start_30 = (selected_date - timedelta(days=60)).strftime("%Y-%m-%d")
    end_30 = date_str
    items_60d = db.get_items_between(start_30, end_30)

    streak_rows = []
    for h in habits:
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

    # ✅ Dog reward (버그 수정 + 도감/마일스톤 연동)
    st.divider()
    st.subheader("오늘의 보상 🐶")

    try:
        if total_count == 0:
            st.info("습관이 없어서 보상을 계산할 수 없어요.")
        elif not fresh:
            # ✅ 핵심 수정: 저장된 체크인이 없으면 보상 표시 금지
            st.info("체크인을 저장하면 보상이 열려요!")
        elif success_count <= 0:
            # ✅ 핵심 수정: 성공(목표 달성)한 습관이 1개 이상일 때만 강아지 표시
            st.info("습관을 1개 이상 목표 달성하면 강아지 보상이 나타나요!")
        else:
            # (1) 체크 완료 보상(기존 유지): 성공했을 때만 이미지 표시
            # 달성률별 보여주는 수는 유지하되, '성공>0'일 때만 실행됨
            if rate >= 100:
                st.success("퍼펙트! 100% 달성 🎉🎉")
                urls = cached_dogs(2)
                cols = st.columns(2)
                for i, u in enumerate(urls[:2]):
                    with cols[i]:
                        st.image(u, use_container_width=True)
                # 도감 저장(대표 1장만 저장)
                if urls:
                    db.add_dog_to_collection(date_str, None, urls[0], "epic", "check")
            elif rate >= 70:
                st.success("좋아요! 70% 이상 달성 🎉")
                urls = cached_dogs(1)
                if urls:
                    st.image(urls[0], use_container_width=True)
                    db.add_dog_to_collection(date_str, None, urls[0], "rare", "check")
            else:
                st.info("좋아요! 목표 달성한 습관이 있어요 🧡")
                urls = cached_dogs(1)
                if urls:
                    st.image(urls[0], use_container_width=True)
                    db.add_dog_to_collection(date_str, None, urls[0], "common", "check")

            # (2) 마일스톤(20/50/80/100) 자동 지급(신규 기능)
            maybe_award_milestones(date_str, rate, last_checked_habit_id=None)

    except Exception as e:
        st.warning(f"Dog API/도감 처리 실패: {e}")

    # AI coaching
    st.divider()
    st.subheader("AI 코칭")

    # 7-day summary
    start_7 = (selected_date - timedelta(days=6)).strftime("%Y-%m-%d")
    end_7 = date_str
    items_7d = db.get_items_between(start_7, end_7)
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

            # ✅ 저장(확장): model/type 포함
            db.add_coaching_log_v2(
                date_str=date_str,
                coach_type="daily",
                tone=tone,
                model=st.session_state.get("model", "gpt-4o-mini"),
                weather_summary=weather_summary,
                input_summary=input_summary,
                content=output,
            )

            st.markdown(output)
        except Exception as e:
            st.error(f"코칭 생성 실패: {e}")


# ---------- Page: Dashboard ----------
def page_dashboard():
    st.header("대시보드 / 통계")

    if not habits:
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

    start_s = _date_str(start)
    end_s = _date_str(end)

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

    st.subheader("가장 긴 streak TOP 3 (daily)")
    streak_rows = []
    for h in habits:
        if h["frequency"] != "daily":
            continue
        s = compute_daily_streak(items, int(h["habit_id"]), int(h["goal"]), end_s)
        streak_rows.append((h["name"], s))
    streak_rows.sort(key=lambda x: x[1], reverse=True)
    top3 = streak_rows[:3]
    if top3:
        for name, s in top3:
            st.write(f"- {name}: {s}일 연속")
    else:
        st.info("daily 습관이 없거나 streak를 계산할 데이터가 없어요.")

    st.divider()
    st.subheader("AI 한 줄 요약")
    if st.button("AI 한 줄 요약 생성", type="primary"):
        if not openai_key:
            st.error("OpenAI API Key가 필요해요.")
            return
        summary_lines = []
        summary_lines.append(f"기간: {start_s} ~ {end_s}")
        summary_lines.append("습관별 성공률:")
        for _, r in by_habit.sort_values("success_rate", ascending=False).iterrows():
            summary_lines.append(f"- {r['name']}: {r['success_rate']:.0f}%")
        weakest = by_habit.sort_values("success_rate", ascending=True).iloc[0]
        summary_lines.append(f"가장 약한 습관: {weakest['name']} ({weakest['success_rate']:.0f}%)")
        user_prompt = "\n".join(summary_lines) + "\n\n위 통계를 한 줄로 요약해줘. (한국어, 간결, 실행 의지 높이기)"

        try:
            output, _ = generate_coaching(
                api_key=openai_key,
                model=st.session_state.get("model", "gpt-4o-mini"),
                tone=tone,
                date_str=end_s,
                city=city,
                weather_summary="(대시보드 요약에는 날씨 생략)",
                today_items=[],
                seven_day_summary=user_prompt,
                note="(한 줄 요약 요청)",
            )
            st.markdown("**결과**")
            st.write(output.strip().splitlines()[0] if output.strip() else output)
        except Exception as e:
            st.error(f"요약 생성 실패: {e}")


# ---------- Page: Coaching Logs ----------
def page_logs():
    st.header("AI 코칭 기록")

    logs = db.list_coaching_logs_v2(limit=200)
    if not logs:
        st.info("아직 코칭 기록이 없어요.")
        return

    options = [f"{l['date']} | {l['type']} | {l['tone']} | #{l['id']}" for l in logs]
    idx = st.selectbox("기록 선택", options=list(range(len(options))), format_func=lambda i: options[i])
    selected = logs[idx]

    st.subheader(f"{selected['date']} • {selected['type']} • {selected['tone']}")
    st.caption(f"model: {selected.get('model','-')} | created_at: {selected['created_at']}")
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
    st.markdown(selected["content"])

    st.divider()
    st.subheader("내보내기")
    export_df = pd.DataFrame(logs)
    st.download_button(
        "코칭 로그 CSV 다운로드",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="coach_logs.csv",
        mime="text/csv",
    )


# ---------- Page: Dog Album ----------
def page_dog_album():
    st.header("🐶 강아지 도감")

    if not habits:
        st.info("습관이 있어야 도감을 모을 수 있어요.")
        return

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        scope = st.selectbox("기간", options=["최근 7일", "전체"], index=0)
    with c2:
        rarity = st.selectbox("등급", options=["전체", "common", "rare", "epic", "common_or_rare"], index=0)
    with c3:
        per_row = st.selectbox("열 개수", options=[3, 4, 5], index=1)

    date_from = None
    if scope == "최근 7일":
        date_from = _date_str(date.today() - timedelta(days=6))

    rows = db.list_dog_collection(date_from=date_from, rarity=None if rarity == "전체" else rarity)
    if not rows:
        st.info("아직 도감 기록이 없어요. 체크인을 저장하고 습관을 달성해보세요!")
        return

    # grid
    cols = st.columns(per_row)
    for i, r in enumerate(rows):
        with cols[i % per_row]:
            with st.container(border=True):
                st.image(r["image_url"], use_container_width=True)
                st.caption(f"{r['date']} · {r['rarity']} · {r['earned_by']}")

    st.divider()
    df = pd.DataFrame(rows)
    st.download_button(
        "도감 CSV 다운로드",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="dog_collection.csv",
        mime="text/csv",
    )


# ---------- Page: Groups (Together streak) ----------
def page_groups():
    st.header("👥 그룹: 함께 streak (MVP)")

    st.info(
        "MVP 안내: 인증/로그인 없이 닉네임 기반으로만 동작합니다.\n"
        "- 같은 서버(같은 DB)를 쓰는 사용자끼리는 그룹 streak가 의미 있게 동작합니다.\n"
        "- 로컬에서 혼자 실행하면, 본인만 체크인 데이터가 있어 다른 멤버는 '데이터 없음'으로 보일 수 있어요."
    )

    with st.container(border=True):
        st.subheader("1) 그룹 생성")
        name = st.text_input("그룹 이름", placeholder="예: 아침 루틴 팀")
        if st.button("그룹 만들기"):
            if not name.strip():
                st.error("그룹 이름이 필요해요.")
            else:
                code = db.create_group(name.strip())
                st.success(f"그룹 생성 완료! 코드: {code}")
                st.code(code)

    st.divider()

    with st.container(border=True):
        st.subheader("2) 그룹 참여")
        code_in = st.text_input("그룹 코드", placeholder="예: A1B2C3D4")
        if st.button("참여하기"):
            if not code_in.strip():
                st.error("그룹 코드를 입력해 주세요.")
            else:
                try:
                    db.join_group(code_in.strip(), st.session_state["nickname"])
                    st.success("참여 완료!")
                    st.rerun()
                except Exception as e:
                    st.error(f"참여 실패: {e}")

    st.divider()

    my_groups = db.list_groups_for_nickname(st.session_state["nickname"])
    if not my_groups:
        st.caption("아직 참여한 그룹이 없어요.")
        return

    pick = st.selectbox("내 그룹 선택", options=[g["group_code"] for g in my_groups])
    group = db.get_group_by_code(pick)
    members = db.get_group_members(pick)

    st.subheader(f"그룹 현황: {group['name']} ({group['group_code']})")

    # 오늘 상태 갱신(현재 사용자 체크인 반영)
    today_s = _date_str(date.today())
    db.update_group_daily_status(group["id"], today_s)

    # streak 계산
    group_streak = db.calc_group_streak(group["id"])
    st.metric("그룹 streak", f"{group_streak}일")

    # 멤버 상태
    st.write("**오늘 멤버 달성 상태**")
    rows = []
    for m in members:
        achieved = db.compute_member_today_achieved(m["nickname"], today_s)
        rows.append({"nickname": m["nickname"], "achieved_today": "✅" if achieved is True else ("❌" if achieved is False else "데이터 없음")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # 최근 7일 그룹 상태
    st.write("**최근 7일 그룹 달성(전원 달성) 기록**")
    logs = db.list_group_streak_logs(group["id"], date_from=_date_str(date.today() - timedelta(days=6)))
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True)
    else:
        st.caption("아직 그룹 기록이 없어요. 오늘부터 체크인을 꾸준히 저장해보세요.")


# ---------- Router ----------
if menu == "습관 관리":
    page_habits()
elif menu == "대시보드/통계":
    page_dashboard()
elif menu == "AI 코칭 기록":
    page_logs()
elif menu == "🐶 도감":
    page_dog_album()
elif menu == "👥 그룹(함께 streak)":
    page_groups()
else:
    page_today()
