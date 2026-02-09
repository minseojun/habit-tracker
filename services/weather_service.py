# services/weather.py
import requests

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
