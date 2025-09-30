# main.py

import json
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os

# --- 1. Carregamento e Processamento de Dados (Sua Lógica) ---

def load_data():
    """
    Carrega e processa os dados dos arquivos JSON para DataFrames do Pandas.
    """
    # --- Carregamento de Candidatos ---
    try:
        with open(os.path.join('..', 'data', 'applicants.json'), 'r', encoding='utf-8') as file:
            dados_candidatos = json.load(file)
    except FileNotFoundError:
        raise RuntimeError("Arquivo 'applicants.json' não encontrado. Verifique se a API está na pasta correta.")

    info_candidatos = []
    for candidato_id, candidato_info in dados_candidatos.items():
        infos_basicas = candidato_info.get('infos_basicas', {})
        infos_prof = candidato_info.get('informacoes_profissionais', {})
        formacao = candidato_info.get('formacao_e_idiomas', {})
        registroCandidato = {
            'id_candidato': candidato_id,
            'cv_texto_pt': candidato_info.get('cv_pt', ''),
            'cv_texto_en': candidato_info.get('cv_en', ''),
            'applicant_area_atuacao': infos_prof.get('area_atuacao', ''),
            'applicant_conhecimentos': infos_prof.get('conhecimentos_tecnicos', ''),
            'applicant_certificacoes': infos_prof.get('certificacoes', ''),
            'applicant_nivel_profissional': infos_prof.get('nivel_profissional', ''),
            'applicant_nivel_academico': formacao.get('nivel_academico', ''),
            'applicant_nivel_ingles': formacao.get('nivel_ingles', ''),
            'applicant_nivel_espanhol': formacao.get('nivel_espanhol', ''),
            'applicant_nome': infos_basicas.get('nome', '')
        }

        info_candidatos.append(registroCandidato)
    df_candidatos = pd.DataFrame(info_candidatos)

    # --- Carregamento de Vagas ---
    try:
        with open(os.path.join('..', 'data', 'vagas.json'), 'r', encoding='utf-8') as file:
            vagas = json.load(file)
    except FileNotFoundError:
        raise RuntimeError("Arquivo 'vagas.json' não encontrado. Verifique se a API está na pasta correta.")

    info_vagas = []
    for id_vaga, vaga_info in vagas.items():
        info_basicas = vaga_info.get('informacoes_basicas', {})
        perfil = vaga_info.get('perfil_vaga', {})
        registro_vaga = {
            'id_vaga': id_vaga, # 
            'titulo_vaga_detalhado': info_basicas.get('titulo_vaga', ''),
            'cliente_vaga': info_basicas.get('cliente', ''),
            'tipo_contratacao': info_basicas.get('tipo_contratacao', ''),
            'pais_vaga': perfil.get('pais', ''),
            'estado_vaga': perfil.get('estado', ''),
            'cidade_vaga': perfil.get('cidade', ''),
            'nivel_profissional_vaga': perfil.get('nivel_profissional', ''),
            'nivel_academico_vaga': perfil.get('nivel_academico', ''),
            'nivel_ingles_vaga': perfil.get('nivel_ingles', ''),
            'nivel_espanhol_vaga': perfil.get('nivel_espanhol', ''),
            'areas_atuacao_vaga': perfil.get('areas_atuacao', ''),
            'atividades_vaga': perfil.get('principais_atividades', ''), 
            'competencias_vaga': perfil.get('competencia_tecnicas_e_comportamentais', ''), 
            'observacoes_vaga': perfil.get('demais_observacoes', ''),
            'pcd_vaga': perfil.get('vaga_especifica_para_pcd', '')
        }

        info_vagas.append(registro_vaga)
    df_vagas = pd.DataFrame(info_vagas)

    print(f"Dados carregados: {len(df_vagas)} vagas e {len(df_candidatos)} candidatos.")
    return df_vagas, df_candidatos


# Carrega os dados na inicialização da API
df_vagas, df_candidatos = load_data()


# --- 2. Definição dos Modelos Pydantic ---

# Modelos para Ranking
class CandidateRank(BaseModel):
    id_candidato: int
    nome_candidato: str
    score: float

class RankingResponse(BaseModel):
    ranking: List[CandidateRank]

# Modelos para Vagas
class JobCreate(BaseModel):
    titulo_vaga_detalhado: str
    cliente_vaga: Optional[str] = None
    tipo_contratacao: Optional[str] = None
    nivel_profissional_vaga: Optional[str] = None
    atividades_vaga: str
    competencias_vaga: str

class Job(JobCreate):
    id_vaga: int

# Modelos para Listagem Paginada
class JobSummary(BaseModel):
    id_vaga: int
    titulo_vaga_detalhado: Optional[str] = None
    cliente_vaga: Optional[str] = None
    tipo_contratacao: Optional[str] = None
    pais_vaga: Optional[str] = None
    estado_vaga: Optional[str] = None
    cidade_vaga: Optional[str] = None
    nivel_profissional_vaga: Optional[str] = None
    nivel_academico_vaga: Optional[str] = None
    nivel_ingles_vaga: Optional[str] = None
    nivel_espanhol_vaga: Optional[str] = None
    areas_atuacao_vaga: Optional[str] = None
    atividades_vaga: Optional[str] = None
    competencias_vaga: Optional[str] = None
    observacoes_vaga: Optional[str] = None
    pcd_vaga: Optional[str] = None

class PaginatedJobResponse(BaseModel):
    total_jobs: int
    jobs: List[JobSummary]

class CandidateSummary(BaseModel):
    id_candidato: int
    applicant_nome: Optional[str] = None
    cv_texto_pt: Optional[str] = None
    cv_texto_en: Optional[str] = None
    applicant_area_atuacao: Optional[str] = None
    applicant_conhecimentos: Optional[str] = None
    applicant_certificacoes: Optional[str] = None
    applicant_nivel_profissional: Optional[str] = None
    applicant_nivel_academico: Optional[str] = None
    applicant_nivel_ingles: Optional[str] = None
    applicant_nivel_espanhol: Optional[str] = None

class PaginatedCandidateResponse(BaseModel):
    total_candidates: int
    candidates: List[CandidateSummary]


# --- 3. Inicialização da Aplicação FastAPI ---

app = FastAPI(
    title="API de Ranking de Candidatos",
    description="Uma API para encontrar os melhores candidatos para uma vaga usando Machine Learning.",
    version="2.2.0" # Versão atualizada
)


# --- 4. Definição das Rotas (Endpoints) ---

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "API online"}

# --- ROTAS DE RANKING ---
@app.post("/rank/{job_id}", response_model=RankingResponse, tags=["Ranking"])
def get_top_candidates_for_job(
    job_id: int, top_n: int = Query(5, ge=1, le=50)
):
    
    # Procura a vaga no DataFrame
    vaga = df_vagas[df_vagas['id_vaga'] == job_id]
    
    # Se o resultado da busca for um DataFrame vazio, a vaga não foi encontrada
    if vaga.empty:
        raise HTTPException(status_code=404, detail=f"Vaga com ID {job_id} não encontrada.")
    
    # Lógica de ranking (simulada)
    # ...
    return {"ranking": []} # Lógica omitida para brevidade

# --- ROTAS DE VAGAS ---

@app.get("/jobs/", response_model=PaginatedJobResponse, tags=["Vagas"])
def list_jobs(skip: int = 0, limit: int = 20):
    """
    Retorna uma lista paginada de vagas.
    """
    total_jobs = len(df_vagas)
    jobs_slice = df_vagas.iloc[skip : skip + limit]
    return {
        "total_jobs": total_jobs,
        "jobs": jobs_slice.to_dict(orient='records')
    }

# --- NOVA ROTA PARA BUSCAR VAGA POR ID ---

@app.get("/jobs/{job_id}", response_model=Job, tags=["Vagas"])
def get_job_by_id(job_id: int):
    """
    Resgata os detalhes completos de uma vaga específica pelo seu ID.
    """
    # Procura a vaga no DataFrame
    vaga = df_vagas[df_vagas['id_vaga'] == job_id]
    
    # Se o resultado da busca for um DataFrame vazio, a vaga não foi encontrada
    if vaga.empty:
        raise HTTPException(status_code=404, detail=f"Vaga com ID {job_id} não encontrada.")
    
    # Retorna o primeiro (e único) resultado encontrado, convertido para dicionário
    return vaga.iloc[0].to_dict()

@app.post("/jobs/", response_model=Job, status_code=status.HTTP_201_CREATED, tags=["Vagas"])
def create_job(job: JobCreate):
    """
    Cadastra uma nova vaga no sistema (em memória).
    """
    # Lógica de cadastro (omitida para brevidade)
    # ...
    global df_vagas
    new_id = df_vagas['id_vaga'].max() + 1 if not df_vagas.empty else 1
    new_job_data = job.model_dump()
    new_job_data['id_vaga'] = new_id
    new_job_df = pd.DataFrame([new_job_data])
    df_vagas = pd.concat([df_vagas, new_job_df], ignore_index=True)
    return new_job_data

# --- ROTAS DE CANDIDATOS ---

@app.get("/candidates/", response_model=PaginatedCandidateResponse, tags=["Candidatos"])
def list_candidates(skip: int = 0, limit: int = 20):
    """
    Retorna uma lista paginada de candidatos.
    """
    # Lógica de listagem (omitida para brevidade)
    # ...
    total_candidates = len(df_candidatos)
    candidates_slice = df_candidatos.iloc[skip : skip + limit]
    return {
        "total_candidates": total_candidates,
        "candidates": candidates_slice.to_dict(orient='records')
    }