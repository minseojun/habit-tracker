import requests
import streamlit as st
from typing import Dict, Any


def _dog_base() -> str:
    if "DOG_API_BASE" in st.secrets:
        return st.secrets["DOG_API_BASE"]
    return "https://dog.ceo/api"


@st.cache_data(ttl=60)  # 1분 캐시: 연속 체크 시 과호출 방지
def fetch_dog_image() -> Dict[str, Any]:
    base = _dog_base()
    try:
        url = f"{base}/breeds/image/random"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "success" and data.get("message"):
            return {"ok": True, "url": data["message"]}
        return {"ok": False, "url": None}
    except Exception:
        return {"ok": False, "url": None}


def dog_fallback_text() -> str:
    return "🐶 (강아지 보상 이미지를 불러오지 못했어요)"
