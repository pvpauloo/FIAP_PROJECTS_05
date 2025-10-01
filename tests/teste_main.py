# tests/test_main.py

import pytest
from fastapi.testclient import TestClient
import pandas as pd

# Adiciona o diretório 'src' ao caminho para que o Python encontre 'main.py'
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','app')))
print(str(os.path.abspath(os.path.join(os.path.dirname(__file__)))))
# Importa o app DEPOIS de ajustar o path
from main import app

# --- Fixture de Teste ---
# Esta função prepara um ambiente de teste limpo para cada teste. 
@pytest.fixture
def client_with_mock_data(monkeypatch):
    """
    Esta fixture simula (mocks) a função load_data ANTES que a aplicação inicie.
    """
    mock_vagas_df = pd.DataFrame([
        {'id_vaga': 5185, 'titulo_vaga_detalhado': 'Vaga de Teste 1'}
    ])
    mock_candidatos_df = pd.DataFrame([
        {'id_candidato': 31001, 'applicant_nome': 'Candidato Teste A'}
    ])

    # Função falsa que retorna nossos dados simulados
    def mock_load_data():
        print("--- Usando dados simulados (mock) para o teste ---")
        return mock_vagas_df, mock_candidatos_df

    # Substitui a função real 'load_data' pela nossa versão simulada
    monkeypatch.setattr("main.load_data", mock_load_data)

    # O TestClient iniciará a aplicação. O 'lifespan' chamará nossa
    # função 'mock_load_data' em vez da original, evitando a leitura de arquivos.
    with TestClient(app) as client:
        yield client


# --- Testes Corrigidos para cada Rota da API ---
# Note que todos os testes agora recebem 'client_with_mock_data' como argumento

def test_read_root(client_with_mock_data):
    """Testa a rota de health check (GET /)."""
    response = client_with_mock_data.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API online"}

def test_get_job_by_id_success(client_with_mock_data):
    """Testa a busca de uma vaga por ID existente."""
    job_id = 4530
    response = client_with_mock_data.get(f"/jobs/{job_id}")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["id_vaga"] == job_id

def test_get_job_by_id_not_found(client_with_mock_data):
    """Testa a busca de uma vaga com ID que não existe."""
    job_id = 99999
    response = client_with_mock_data.get(f"/jobs/{job_id}")
    assert response.status_code == 404

def test_rank_candidates_success(client_with_mock_data):
    """Testa o ranking de candidatos para uma vaga existente."""
    job_id = 5185
    top_n = 1
    response = client_with_mock_data.post(f"/rank/{job_id}?top_n={top_n}")
    assert response.status_code == 200, response.text
    data = response.json()
    assert "ranking" in data
    # A lógica simulada na rota de ranking pode estar vazia, então testamos se a chave existe
    assert isinstance(data["ranking"], list)

def test_rank_candidates_job_not_found(client_with_mock_data):
    """Testa o ranking para uma vaga inexistente."""
    job_id = 99999
    response = client_with_mock_data.post(f"/rank/{job_id}?top_n=5")
    assert response.status_code == 404


