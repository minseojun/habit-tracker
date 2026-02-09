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


# -----------------------------
# 환경변수(.env) fallback 로드
# -----------------------------
def load_env_fallback():
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
        return "이미 같은 이름의 습관이 있어요."
    return ""


def main():
    st.set_page_config(page_title="AI 습관 트래커", page_icon="✅", layout="wide")

    db.init_db()
    load_env_fallback()

    today = date.today()
    today_str = today.isoformat()

    # =========================
    # 사이드바
    # =========================
    st.sidebar.title("✅ AI 습관 트래커")
    st.sidebar.caption("날씨·성과 기반 코칭 + 강아지 보상 🐶")

    profile = db.get_profile() or {"nickname": "", "city": "", "daily_goal_n": 1}

    # ---- 프로필 설정 ----
    with st.sidebar.container(border=True):
        st.subheader("프로필 설정")
        nickname = st.text_input("닉네임", value=profile["nickname"])
        city = st.text_input("도시(날씨)", value=profile["city"])
        daily_goal_n = st.number_input(
            "하루 목표 완료 개수",
            min_value=1,
            max_value=50,
            value=int(profile["daily_goal_n"]),
        )

        if st.button("저장"):
            if not validate_profile(nickname, city):
                st.error("닉네임과 도시는 필수입니다.")
            else:
                db.upsert_profile(nickname, city, daily_goal_n)
                st.success("저장 완료")
                st.rerun()

        st.caption(f"오늘 날짜: {today_str}")

    # ---- API 키 로컬 입력 (🔥 여기 indentation 문제 해결됨) ----
    with st.sidebar.container(border=True):
        st.subheader("API 키 (로컬 입력)")
        st.caption("※ 새로고침 시 유지됨, 배포 시 비권장")

        openai_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
        )

        if openai_key_input:
            st.session_state["OPENAI_API_KEY"] = openai_key_input
            st.success("OpenAI API 키가 설정되었습니다")

    # ---- AI 설정 ----
    with st.sidebar.container(border=True):
        st.subheader("AI 설정")
        model = st.selectbox("모델 선택", ["gpt-4o-mini", "gpt-4o"])
        if st.button("🧠 오늘의 AI 코치"):
            st.session_state["run_ai_coach"] = True

    # =========================
    # 데이터 계산
    # =========================
    weather = fetch_weather(city) if city else {"ok": False, "summary": "날씨 알 수 없음"}
    weather_summary = weather.get("summary", "날씨 알 수 없음")

    today_completed, today_total = db.get_today_counts(today_str)
    avg_7d = db.get_avg_7d(today)
    streak = db.calc_streak(today, daily_goal_n)

    # =========================
    # 메인 UI
    # =========================
    st.title("오늘의 습관")

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("오늘 완료", f"{today_completed}/{today_total}")
    with c2:
        kpi_card("연속 달성", f"{streak}일")
    with c3:
        kpi_card("오늘 날씨", weather_summary)

    st.divider()
    # =========================
    # 습관 추가/편집
    # =========================
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
                e_freq_type = st.selectbox(
                    "빈도 타입", options=["daily", "weekly"],
                    index=0 if h["frequency_type"] == "daily" else 1,
                    key="edit_ft"
                )
                e_freq_n = st.number_input("빈도 수(n)", min_value=1, max_value=7, value=int(h["frequency_n"]), key="edit_fn")
                e_start = st.date_input("시작일", value=pd.to_datetime(h["start_date"]).date(), key="edit_sd")
                e_active = st.checkbox("활성 상태", value=bool(h["is_active"]), key="edit_active")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("저장/업데이트", use_container_width=True):
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
    # =========================
    # 습관 체크리스트
    # =========================
    habits = db.list_active_habits()
    left, right = st.columns([2, 1])

    with left:
        st.subheader("체크리스트")

        if not habits:
            st.info("습관을 먼저 추가해 주세요.")

        for h in habits:
            hid = h["id"]
            log = db.get_log(today_str, hid)
            checked = bool(log["completed"]) if log else False

            with st.container(border=True):
                new_checked = st.checkbox(h["name"], value=checked)

                if new_checked != checked:
                    db.upsert_log(today_str, hid, new_checked)

                    if new_checked:
                        st.toast("완료! 🐶 보상 등장", icon="✅")
                        st.session_state["show_dog"] = True
                    else:
                        st.toast("미완료 처리", icon="↩️")

                    st.rerun()

    with right:
        st.subheader("보상")
        if st.session_state.get("show_dog"):
            dog = fetch_dog_image()
            if dog.get("ok"):
                st.image(dog["url"], use_container_width=True)
            else:
                st.write(dog_fallback_text())
            st.session_state["show_dog"] = False
        else:
            st.caption("습관을 완료하면 강아지가 나와요 🐕")

    st.divider()

    # =========================
    # AI 코치
    # =========================
    if st.session_state.get("run_ai_coach"):
        st.session_state["run_ai_coach"] = False

        with st.container(border=True):
            st.subheader("오늘의 AI 코칭")

            msg, meta = generate_coach_message(
                model=model,
                nickname=nickname,
                city=city,
                weather_summary=weather_summary,
                today_completed=today_completed,
                today_total=today_total,
                streak=streak,
                avg_7d=avg_7d,
                daily_goal_n=daily_goal_n,
            )

            if meta.get("ok"):
                st.info(msg)
            else:
                st.error(meta.get("error", "AI 호출 실패"))


if __name__ == "__main__":
    main()
