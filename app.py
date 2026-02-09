import os
from datetime import date
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import db
from services.weather_service import fetch_weather
from services.dog_service import fetch_dog_image, dog_fallback_text
from services.openai_service import generate_coach_message, generate_weekly_report
from ui.components import kpi_card, info_card


def load_env_fallback():
    """
    secrets.toml이 없을 경우 .env를 지원 (선택)
    Streamlit Cloud에서는 secrets 권장.
    """
    load_dotenv()
    for k in ["OPENAI_API_KEY", "OPENWEATHER_API_KEY", "DOG_API_BASE"]:
        if k in os.environ:
            st.session_state[k] = os.environ.get(k)


def validate_profile(nickname: str, city: str) -> bool:
    return bool(nickname.strip()) and bool(city.strip())


def validate_habit_name(name: str, existing_names: set) -> str:
    name = (name or "").strip()
    if not (1 <= len(name) <= 30):
        return "습관명은 1~30자여야 해요."
    if name in existing_names:
        return "이미 같은 이름의 습관이 있어요(중복 불가)."
    return ""


def main():
    st.set_page_config(page_title="AI 습관 트래커", page_icon="✅", layout="wide")

    db.init_db()
    load_env_fallback()

    today = date.today()
    today_str = today.isoformat()

    # ---- Sidebar: Profile + Controls ----
    st.sidebar.title("✅ AI 습관 트래커")
    st.sidebar.caption("날씨·성과 기반 코칭 + 강아지 보상 🐶")

    with st.sidebar.container(border=True):
    st.subheader("API 키(로컬 입력)")
    openai_key_input = st.text_input("OpenAI API Key", type="password")
    if openai_key_input:
        st.session_state["OPENAI_API_KEY"] = openai_key_input

    profile = db.get_profile()
    if profile is None:
        profile = {"nickname": "", "city": "", "daily_goal_n": 1}

    with st.sidebar.container(border=True):
        st.subheader("프로필 설정")
        nickname = st.text_input("닉네임", value=profile.get("nickname", ""), placeholder="예: 재홍")
        city = st.text_input("도시(날씨 조회)", value=profile.get("city", ""), placeholder="예: Seoul")
        daily_goal_n = st.number_input("하루 목표(최소 완료 개수)", min_value=1, max_value=50, value=int(profile.get("daily_goal_n", 1)))

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("저장", use_container_width=True):
                if not validate_profile(nickname, city):
                    st.error("닉네임/도시는 비어있으면 안 돼요.")
                else:
                    db.upsert_profile(nickname, city, int(daily_goal_n))
                    st.success("프로필 저장 완료!")
                    st.rerun()
        with col_b:
            st.caption(f"오늘: {today_str}")

    with st.sidebar.container(border=True):
        st.subheader("AI 설정")
        model = st.selectbox("OpenAI 모델", options=["gpt-4o-mini", "gpt-4o"], index=0)
        st.caption("※ 토큰/비용 표시는 추정치(모델/요금제에 따라 다를 수 있음)")

        if st.button("🧠 오늘의 AI 코치", use_container_width=True):
            if not validate_profile(nickname, city):
                st.error("먼저 프로필(닉네임/도시)을 저장해 주세요.")
            else:
                st.session_state["trigger_coach"] = True

    # ---- Fetch weather (10min cached) ----
    weather = fetch_weather(city.strip()) if city else {"ok": False, "summary": "날씨 알 수 없음"}
    weather_summary = weather.get("summary", "날씨 알 수 없음")

    # ---- Data for KPIs ----
    today_completed, today_total = db.get_today_counts(today_str)
    avg_7d = db.get_avg_7d(today)
    streak = db.calc_streak(today, int(daily_goal_n))

    # ---- Main Layout ----
    st.title("오늘의 습관")

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("오늘 완료", f"{today_completed} / {today_total}", "체크할수록 통계/코칭이 정확해져요")
    with k2:
        kpi_card("streak", f"{streak}일", f"하루 목표: {int(daily_goal_n)}개 이상 완료")
    with k3:
        sub = ""
        if weather.get("ok"):
            sub = f"습도 {weather.get('humidity','?')}% · 바람 {weather.get('wind_speed','?')}m/s"
        kpi_card("오늘 날씨", weather_summary, sub)

    st.divider()

    # ---- Habits checklist (with dog reward) ----
    habits = db.list_active_habits()
    if not validate_profile(nickname, city):
        st.warning("👈 먼저 사이드바에서 **프로필(닉네임/도시/목표)** 을 저장해 주세요.")
    if not habits:
        st.info("습관이 아직 없어요. 아래에서 습관을 추가해 주세요.")

    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("체크리스트")
        # 오늘 체크 UI
        for h in habits:
            hid = h["id"]
            log = db.get_log(today_str, hid)
            checked = bool(log["completed"]) if log else False
            memo_val = (log["memo"] if log and log["memo"] else "")

            row = st.container(border=True)
            with row:
                c1, c2, c3 = st.columns([2.2, 1.2, 1.2])
                with c1:
                    new_checked = st.checkbox(
                        f"**{h['name']}**  ·  _{h['frequency_type']} {h['frequency_n']}_",
                        value=checked,
                        key=f"chk_{today_str}_{hid}",
                    )
                    st.caption(f"카테고리: {h['category']} · 시작일: {h['start_date']}")
                with c2:
                    with st.popover("메모"):
                        memo = st.text_area("오늘 메모", value=memo_val, key=f"memo_{today_str}_{hid}", height=90)
                        if st.button("메모 저장", key=f"save_memo_{today_str}_{hid}"):
                            db.set_memo(today_str, hid, memo)
                            st.success("메모 저장!")
                            st.rerun()
                with c3:
                    # 체크 변경 반영
                    if new_checked != checked:
                        db.upsert_log(today_str, hid, new_checked, None)
                        if new_checked:
                            st.toast("완료! 보상 강아지 등장 🐶", icon="✅")
                            st.session_state["show_dog"] = True
                        else:
                            st.toast("미완료로 변경했어요.", icon="↩️")
                        st.rerun()

    with right:
        st.subheader("보상")
        if st.session_state.get("show_dog"):
            dog = fetch_dog_image()
            with st.container(border=True):
                st.markdown("**랜덤 강아지 보상**")
                if dog.get("ok") and dog.get("url"):
                    st.image(dog["url"], use_container_width=True)
                else:
                    st.write(dog_fallback_text())
            # 보상은 한 번 보여주고 자동 해제(UX)
            st.session_state["show_dog"] = False
        else:
            st.caption("습관을 체크하면 강아지 보상이 나타나요!")

    st.divider()

    # ---- Tabs: Stats & AI report ----
    tab_stats, tab_ai = st.tabs(["📈 통계", "📝 AI 리포트"])

    with tab_stats:
        st.subheader("최근 7일 달성률")
        series = db.get_last_7_days_series(today)
        df = pd.DataFrame(series)
        if df.empty:
            st.info("아직 통계가 없어요. 체크를 시작해 주세요!")
        else:
            df["date"] = pd.to_datetime(df["date"])
            df["rate_pct"] = (df["rate"] * 100).round(0)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**달성률(%)**")
                st.line_chart(df.set_index("date")["rate_pct"])
            with c2:
                st.markdown("**완료 개수**")
                st.bar_chart(df.set_index("date")["completed"])

            st.subheader("요일별 히트맵(최근 8주)")
            heat = pd.DataFrame(db.get_weekday_heatmap(today, weeks=8))
            if not heat.empty:
                heat["date"] = pd.to_datetime(heat["date"])
                heat["week"] = heat["date"].dt.isocalendar().week.astype(int)
                heat["weekday_name"] = heat["date"].dt.day_name()

                pivot = heat.pivot_table(
                    index="week",
                    columns="weekday_name",
                    values="rate",
                    aggfunc="mean",
                ).fillna(0.0)

                # 보기 좋게 %로
                pivot_pct = (pivot * 100).round(0).astype(int)
                st.dataframe(pivot_pct, use_container_width=True)
                st.caption("값은 해당 요일의 평균 달성률(%)입니다. (간단 버전)")
            else:
                st.caption("히트맵 데이터가 부족해요.")

    with tab_ai:
        st.subheader("AI 결과")
        ai_box = st.container(border=True)

        # 오늘의 AI 코치 (버튼은 사이드바에 있음)
        if st.session_state.get("trigger_coach"):
            st.session_state["trigger_coach"] = False
            with ai_box:
                with st.spinner("AI 코치가 메시지를 작성 중..."):
                    msg, meta = generate_coach_message(
                        model=model,
                        nickname=nickname.strip(),
                        city=city.strip(),
                        weather_summary=weather_summary,
                        today_completed=today_completed,
                        today_total=today_total,
                        streak=streak,
                        avg_7d=avg_7d,
                        daily_goal_n=int(daily_goal_n),
                    )
                if meta.get("ok") and msg:
                    st.info(msg)
                    usage = meta.get("usage")
                    if usage:
                        st.caption(f"토큰: {usage['total_tokens']} (in {usage['prompt_tokens']} / out {usage['completion_tokens']})")
                    if "cost_usd_est" in meta:
                        st.caption(f"추정 비용: ${meta['cost_usd_est']:.6f}")
                else:
                    st.error(f"OpenAI 호출 실패: {meta.get('error','알 수 없음')}")
                    if st.button("재시도"):
                        st.session_state["trigger_coach"] = True
                        st.rerun()

        st.divider()

        st.subheader("주간 리포트(최근 7일)")
        extra_context = st.text_input("추가 맥락(선택)", placeholder="예: 요즘 야근이 많아서 운동이 어려웠어요.")
        if st.button("📌 주간 요약 생성"):
            if not validate_profile(nickname, city):
                st.error("먼저 프로필을 저장해 주세요.")
            else:
                with st.spinner("주간 리포트 생성 중..."):
                    report, meta = generate_weekly_report(
                        model=model,
                        nickname=nickname.strip(),
                        city=city.strip(),
                        weather_summary=weather_summary,
                        today_completed=today_completed,
                        today_total=today_total,
                        streak=streak,
                        avg_7d=avg_7d,
                        daily_goal_n=int(daily_goal_n),
                        extra_context=extra_context,
                    )
                if meta.get("ok") and report:
                    info_card("AI 주간 리포트", report)
                    usage = meta.get("usage")
                    if usage:
                        st.caption(f"토큰: {usage['total_tokens']} (in {usage['prompt_tokens']} / out {usage['completion_tokens']})")
                    if "cost_usd_est" in meta:
                        st.caption(f"추정 비용: ${meta['cost_usd_est']:.6f}")
                else:
                    st.error(f"OpenAI 호출 실패: {meta.get('error','알 수 없음')}")
                    st.caption("네트워크/키/모델명을 확인하고 다시 시도해 주세요.")

    st.divider()

    # ---- Habit management section ----
    st.subheader("습관 추가/편집")

    all_habits = db.list_all_habits()
    existing_names = {h["name"] for h in all_habits}

    add_col, edit_col = st.columns([1, 1], gap="large")

    with add_col:
        with st.container(border=True):
            st.markdown("**습관 추가**")
            new_name = st.text_input("습관명(1~30자)", key="new_habit_name")
            new_category = st.text_input("카테고리", value="생활", key="new_habit_cat")
            new_freq_type = st.selectbox("빈도 타입", options=["daily", "weekly"], index=0, key="new_habit_ft")
            new_freq_n = st.number_input("빈도 수(n)", min_value=1, max_value=7, value=1, key="new_habit_fn")
            new_start = st.date_input("시작일", value=today, key="new_habit_sd")

            if st.button("추가", use_container_width=True):
                err = validate_habit_name(new_name, existing_names)
                if err:
                    st.error(err)
                else:
                    try:
                        db.add_habit(
                            name=new_name,
                            category=new_category or "기타",
                            frequency_type=new_freq_type,
                            frequency_n=int(new_freq_n),
                            start_date=new_start.isoformat(),
                        )
                        st.success("습관 추가 완료!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"추가 실패: {e}")

    with edit_col:
        with st.container(border=True):
            st.markdown("**습관 편집/삭제(비활성)**")
            if not all_habits:
                st.caption("편집할 습관이 없어요.")
            else:
                options = {f"[{'ON' if h['is_active'] else 'OFF'}] {h['name']} (id={h['id']})": h for h in all_habits}
                pick = st.selectbox("습관 선택", options=list(options.keys()))
                h = options[pick]

                e_name = st.text_input("습관명", value=h["name"], key="edit_name")
                e_category = st.text_input("카테고리", value=h["category"], key="edit_cat")
                e_freq_type = st.selectbox("빈도 타입", options=["daily", "weekly"], index=0 if h["frequency_type"] == "daily" else 1, key="edit_ft")
                e_freq_n = st.number_input("빈도 수(n)", min_value=1, max_value=7, value=int(h["frequency_n"]), key="edit_fn")
                e_start = st.date_input("시작일", value=pd.to_datetime(h["start_date"]).date(), key="edit_sd")
                e_active = st.checkbox("활성 상태", value=bool(h["is_active"]), key="edit_active")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("저장/업데이트", use_container_width=True):
                        # 중복 검사: 이름 변경 시
                        if e_name.strip() != h["name"] and e_name.strip() in existing_names:
                            st.error("이미 같은 이름의 습관이 있어요(중복 불가).")
                        elif not (1 <= len(e_name.strip()) <= 30):
                            st.error("습관명은 1~30자여야 해요.")
                        else:
                            try:
                                db.update_habit(
                                    habit_id=h["id"],
                                    name=e_name,
                                    category=e_category or "기타",
                                    frequency_type=e_freq_type,
                                    frequency_n=int(e_freq_n),
                                    start_date=e_start.isoformat(),
                                    is_active=1 if e_active else 0,
                                )
                                st.success("업데이트 완료!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"업데이트 실패: {e}")

                with c2:
                    if st.button("삭제(비활성)", use_container_width=True):
                        db.delete_habit(h["id"])
                        st.success("비활성 처리 완료!")
                        st.rerun()

    # ---- 비용/호출 최적화 팁 ----
    with st.expander("💡 API 호출/비용 최적화 팁 (코드에도 반영됨)"):
        st.markdown("""
- **날씨**: `st.cache_data(ttl=600)`로 10분 캐시 → 잦은 새로고침에도 호출 최소화  
- **강아지 이미지**: `st.cache_data(ttl=60)`로 1분 캐시 → 연속 체크 시 과호출 방지  
- **OpenAI**:
  - “오늘의 AI 코치”는 버튼 클릭시에만 호출(자동 호출 금지)
  - 주간 리포트도 버튼 클릭시에만 호출
  - 모델을 기본 `gpt-4o-mini`로 두면 비용 절감에 유리(상황에 따라 변경 가능)
""")

if __name__ == "__main__":
    main()
