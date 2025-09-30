import json
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import hashlib
import pandas as pd
# --- 1. Carregamento e Processamento de Dados (Sua Lógica) ---

def load_data():
    try:
        with open(os.path.join('data', 'applicants.json'), 'r', encoding='utf-8') as file:
            dados_candidatos = json.load(file)
    except FileNotFoundError:
        raise RuntimeError("Arquivo 'applicants.json' não encontrado.")

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

    try:
        with open(os.path.join('data', 'vagas.json'), 'r', encoding='utf-8') as file:
            vagas = json.load(file)
    except FileNotFoundError:
        raise RuntimeError("Arquivo 'vagas.json' não encontrado.")

    info_vagas = []
    for id_vaga, vaga_info in vagas.items():
        info_basicas = vaga_info.get('informacoes_basicas', {})
        perfil = vaga_info.get('perfil_vaga', {})
        registro_vaga = {
            'id_vaga': id_vaga,
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
    print(df_vagas.head())
    return df_vagas, df_candidatos




# --- 2. Inicialização da Aplicação FastAPI ---

app = FastAPI(
    title="API de Ranking de Candidatos",
    description="Uma API para encontrar os melhores candidatos para uma vaga usando Machine Learning.",
    version="2.2.0"
)

# Carrega os dados na inicialização
df_vagas, df_candidatos = load_data()

print(df_candidatos.columns)
# Importa e registra as rotas
from app.routes import register_routes

register_routes(app, df_vagas, df_candidatos)
