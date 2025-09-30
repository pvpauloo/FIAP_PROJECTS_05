import requests

BASE_URL = "http://127.0.0.1:8000"

def test_list_jobs():
    url = f"{BASE_URL}/jobs/"
    params = {"skip": 0, "limit": 5}
    r = requests.get(url, params=params)
    print("=== /jobs/ (GET) ===")
    print("Status:", r.status_code)
    print(r.json())
    return r.json()

def test_get_job_by_id(job_id: int):
    url = f"{BASE_URL}/jobs/{job_id}"
    r = requests.get(url)
    print(f"=== /jobs/{job_id} (GET) ===")
    print("Status:", r.status_code)
    print(r.json())

def test_create_job():
    url = f"{BASE_URL}/jobs/"
    payload = {
        "titulo_vaga_detalhado": "Cientista de Dados",
        "cliente_vaga": "Banco XPTO",
        "tipo_contratacao": "CLT",
        "nivel_profissional_vaga": "Sênior",
        "atividades_vaga": "Construção de modelos preditivos",
        "competencias_vaga": "Python, SQL, Machine Learning"
    }
    r = requests.post(url, json=payload)
    print("=== /jobs/ (POST) ===")
    print("Status:", r.status_code)
    print(r.json())
    return r.json()

def test_list_candidates():
    url = f"{BASE_URL}/candidates/"
    params = {"skip": 0, "limit": 5}
    r = requests.get(url, params=params)
    print("=== /candidates/ (GET) ===")
    print("Status:", r.status_code)
    print(r.json())

if __name__ == "__main__":
    # 1. Listar jobs
    jobs_resp = test_list_jobs()

    # 2. Se houver pelo menos uma vaga, pegar o primeiro id e consultar por ele
    if "jobs" in jobs_resp and jobs_resp["jobs"]:
        first_job_id = jobs_resp["jobs"][0]["id_vaga"]
        test_get_job_by_id(first_job_id)
    else:
        print("Nenhuma vaga encontrada para testar /jobs/{job_id}")

    # 3. Criar uma nova vaga
    created_job = test_create_job()

    # 4. Listar candidatos
    test_list_candidates()
