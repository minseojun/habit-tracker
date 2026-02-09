# services/coach.py
from typing import Tuple, List, Dict, Any
from openai import OpenAI

TONES = ["친근하게", "차분하게", "엄격하게", "유쾌하게"]

SYSTEM = """너는 습관 코치다. 한국어로 답한다.
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
    today_items: List[Dict[str, Any]],
    seven_day_summary: str,
    note: str,
) -> Tuple[str, str]:
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
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )
    out = (resp.choices[0].message.content or "").strip()
    return out, user_prompt
