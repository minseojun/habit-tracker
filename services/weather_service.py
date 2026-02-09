import requests
import streamlit as st
from typing import Dict, Any


def _get_weather_key() -> str:
    # 1) 화면 입력/세션 키 우선
    k = (st.session_state.get("OPENWEATHER_API_KEY") or "").strip()
    if k:
        return k

    # 2) 없으면 secrets
    if "OPENWEATHER_API_KEY" in st.secrets:
        return (st.secrets["OPENWEATHER_API_KEY"] or "").strip()

    return ""


@st.cache_data(ttl=600)  # 10분 캐시
def fetch_weather(city: str) -> Dict[str, Any]:
    api_key = _get_weather_key()
    if not api_key or not city:
        return {"ok": False, "summary": "날씨 알 수 없음", "raw": None}

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric", "lang": "kr"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        main = data.get("main", {})
        weather0 = (data.get("weather") or [{}])[0]
        wind = data.get("wind", {})

        temp = main.get("temp")
        feels = main.get("feels_like")
        desc = weather0.get("description", "")
        icon = weather0.get("icon", "")
        humidity = main.get("humidity")
        wind_speed = wind.get("speed")

        summary = f"{desc} / {temp}°C (체감 {feels}°C)"
        return {
            "ok": True,
            "summary": summary,
            "temp": temp,
            "feels_like": feels,
            "desc": desc,
            "icon": icon,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "raw": data,
        }
    except Exception:
        return {"ok": False, "summary": "날씨 알 수 없음", "raw": None}
