from typing import Dict, Any, Optional, Tuple
import streamlit as st
from openai import OpenAI

# ⚠️ 비용은 모델/요금제에 따라 변동됩니다.
# 여기서는 “대략적인” 추정용(표시 목적)으로만 둡니다.
# 정확한 비용이 필요하면 OpenAI 공식 가격표를 기준으로 업데이트하세요.
MODEL_COST_USD_PER_1M_TOKENS = {
    # 예시값(원하면 최신 가격에 맞게 조정)
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 5.00, "out": 15.00},
}

SYSTEM_COACH = """너는 습관 형성 코치다. 사용자가 실행 가능한 행동을 하도록 돕는다.
규칙:
- 한국어로 작성
- 120~220자
- 반드시: (칭찬 1) + (개선 제안 1) + (오늘 할 행동 1)
- 날씨와 최근 성과를 자연스럽게 반영
- 과장 금지, 간결하고 따뜻하게
"""

USER_COACH_TEMPLATE = """사용자 정보:
- 닉네임: {nickname}
- 도시: {city}
- 날씨: {weather_summary}
- 오늘 완료: {today_completed}/{today_total}
- 오늘 streak: {streak}일
- 최근 7일 평균 달성률: {avg_7d:.0%}
- 하루 목표(최소 완료 개수): {daily_goal_n}

요청:
오늘의 코칭 메시지를 규칙에 맞게 작성해줘.
"""

SYSTEM_WEEKLY = """너는 습관 리포트 분석가다.
규칙:
- 한국어
- 5줄 이내
- 포맷: 잘된 점 2개 / 개선점 2개 / 다음주 실험 1개
- 데이터 기반으로 구체적으로, 비난 금지
"""

USER_WEEKLY_TEMPLATE = """최근 7일 데이터 요약:
- 닉네임: {nickname}
- 도시: {city}
- 날씨: {weather_summary}
- 최근 7일 평균 달성률: {avg_7d:.0%}
- 오늘 완료: {today_completed}/{today_total}
- streak: {streak}일
- 하루 목표: {daily_goal_n}

추가 맥락(선택):
{extra_context}

요청:
규칙 포맷으로 주간 리포트를 작성해줘.
"""


def _get_openai_key() -> str:
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return st.session_state.get("OPENAI_API_KEY", "")


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    if model not in MODEL_COST_USD_PER_1M_TOKENS:
        return None
    price = MODEL_COST_USD_PER_1M_TOKENS[model]
    return (prompt_tokens / 1_000_000) * price["in"] + (completion_tokens / 1_000_000) * price["out"]


def generate_coach_message(
    model: str,
    nickname: str,
    city: str,
    weather_summary: str,
    today_completed: int,
    today_total: int,
    streak: int,
    avg_7d: float,
    daily_goal_n: int,
) -> Tuple[Optional[str], Dict[str, Any]]:
    api_key = _get_openai_key()
    if not api_key:
        return None, {"ok": False, "error": "OPENAI_API_KEY가 설정되지 않았습니다."}

    client = OpenAI(api_key=api_key)

    user_prompt = USER_COACH_TEMPLATE.format(
        nickname=nickname,
        city=city,
        weather_summary=weather_summary,
        today_completed=today_completed,
        today_total=today_total,
        streak=streak,
        avg_7d=avg_7d,
        daily_goal_n=daily_goal_n,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.7,
            messages=[
                {"role": "system", "content": SYSTEM_COACH},
                {"role": "user", "content": user_prompt},
            ],
        )

        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        meta = {"ok": True}

        if usage:
            prompt_tokens = usage.prompt_tokens or 0
            completion_tokens = usage.completion_tokens or 0
            total_tokens = usage.total_tokens or (prompt_tokens + completion_tokens)
            meta["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
            if cost is not None:
                meta["cost_usd_est"] = cost

        return text, meta
    except Exception as e:
        return None, {"ok": False, "error": str(e)}


def generate_weekly_report(
    model: str,
    nickname: str,
    city: str,
    weather_summary: str,
    today_completed: int,
    today_total: int,
    streak: int,
    avg_7d: float,
    daily_goal_n: int,
    extra_context: str = "",
) -> Tuple[Optional[str], Dict[str, Any]]:
    api_key = _get_openai_key()
    if not api_key:
        return None, {"ok": False, "error": "OPENAI_API_KEY가 설정되지 않았습니다."}

    client = OpenAI(api_key=api_key)

    user_prompt = USER_WEEKLY_TEMPLATE.format(
        nickname=nickname,
        city=city,
        weather_summary=weather_summary,
        today_completed=today_completed,
        today_total=today_total,
        streak=streak,
        avg_7d=avg_7d,
        daily_goal_n=daily_goal_n,
        extra_context=extra_context or "없음",
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.6,
            messages=[
                {"role": "system", "content": SYSTEM_WEEKLY},
                {"role": "user", "content": user_prompt},
            ],
        )

        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        meta = {"ok": True}

        if usage:
            prompt_tokens = usage.prompt_tokens or 0
            completion_tokens = usage.completion_tokens or 0
            total_tokens = usage.total_tokens or (prompt_tokens + completion_tokens)
            meta["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
            if cost is not None:
                meta["cost_usd_est"] = cost

        return text, meta
    except Exception as e:
        return None, {"ok": False, "error": str(e)}
