# services/dog.py
import requests

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
