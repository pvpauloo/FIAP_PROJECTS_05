import requests

url = "http://127.0.0.1:8000/rank/5183"
params = {"top_n": 5}  

response = requests.post(url, params=params)

if response.status_code == 200:
    data = response.json()
    print("Ranking recebido:")
    for cand in data.get("ranking", []):
        print(cand)
else:
    print("Erro:", response.status_code, response.text)
