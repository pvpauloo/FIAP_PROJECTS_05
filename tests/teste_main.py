# tests/test_main.py

import pytest
from fastapi.testclient import TestClient
import pandas as pd

# Adiciona o diretório 'src' ao caminho para que o Python encontre 'main.py'
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Importa o app DEPOIS de ajustar o path
from main import app

# --- Fixture de Teste ---
# Esta função prepara um ambiente de teste limpo para cada teste.
@pytest.fixture
def client_with_mock_data(monkeypatch):
    """
    Cria um cliente de teste e substitui os DataFrames globais da API
    por dados simulados (mock).
    """
    # 1. Dados simulados que usaremos nos testes
    mock_vagas_df = pd.DataFrame([
        {
            'id_vaga': 4530,
            'titulo_vaga_detalhado': 'Vaga de Teste 1',
            'cliente_vaga': 'Cliente A',
            'tipo_contratacao': 'CLT',
            'nivel_profissional_vaga': 'Pleno',
            'atividades_vaga': 'Atividade da Vaga 1',
            'competencias_vaga': 'Competência da Vaga 1',
        },
        {
            'id_vaga': 5185,
            'titulo_vaga_detalhado': 'Vaga de Teste 2',
            'cliente_vaga': 'Cliente B',
            'tipo_contratacao': 'PJ',
            'nivel_profissional_vaga': 'Sênior',
            'atividades_vaga': 'Atividade da Vaga 2',
            'competencias_vaga': 'Competência da Vaga 2',
        }
    ])
    
    mock_candidatos_df = pd.DataFrame([
        {'id_candidato': 31001, 'applicant_nome': 'Candidato Teste A'},
        {'id_candidato': 25632, 'applicant_nome': 'Candidato Teste B'}
    ])

    # 2. Usar monkeypatch para substituir os DataFrames globais no módulo 'main'
    monkeypatch.setattr("main.df_vagas", mock_vagas_df)
    monkeypatch.setattr("main.df_candidatos", mock_candidatos_df)
    
    # 3. Retorna um cliente de teste que agora opera sobre os dados simulados
    return TestClient(app)


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

def test_create_job_success(client_with_mock_data):
    """Testa o cadastro de uma nova vaga."""
    new_job_payload = {
        "titulo_vaga_detalhado": "Nova Vaga de Teste",
        "atividades_vaga": "Atividades de teste",
        "competencias_vaga": "Competências de teste"
    }
    response = client_with_mock_data.post("/jobs/", json=new_job_payload)
    assert response.status_code == 201, response.text
    data = response.json()
    # O novo ID será o máximo do mock (5185) + 1 = 5186
    assert data["id_vaga"] == 5186
    assert data["titulo_vaga_detalhado"] == "Nova Vaga de Teste"

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
