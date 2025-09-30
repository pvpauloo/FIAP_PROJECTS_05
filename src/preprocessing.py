import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

MAPPING_POS = ["Contratado pela Decision", "Aprovado", "Finalista", "Documentação PJ", "Encaminhado ao Cliente com Aprovação",'Contratado como Hunting','Documentação CLT','Encaminhar Proposta','Proposta Aceita']
MAPPING_NEG = ["Não Aprovado pelo Cliente", "Não Aprovado pelo RH", "Desistiu", "Prospect", 'Encaminhado ao Requisitante','Não Aprovado pelo Requisitante','Desistiu da Contratação','Recusado','Documentação Cooperado','Sem interesse nesta vaga']
MAPPING_PROCESS = ['Inscrito','Entrevista Técnica','Em avaliação pelo RH','Entrevista com Cliente']

def preprocessing():
    ### Get prospects
    with open('../data/prospects.json','r',encoding='utf-8') as file:
        prospects = json.load(file)

    prospects_vaga = []

    for id_vaga,vaga_info in prospects.items():
        titulo_vaga = vaga_info.get('titulo','')
        modalidade_vaga = vaga_info.get('modalidade','')
        prospect_list = vaga_info.get('prospects',[])

        for prospect in prospect_list:
            registro = {
                'id_vaga': id_vaga,
                'titulo_vaga': titulo_vaga,
                'modalidade_vaga': modalidade_vaga,
                'nome_candidato': prospect.get('nome',''),
                'id_candidato': prospect.get('codigo', ''),
                'situacao_candidado': prospect.get('situacao_candidado', ''),
                'data_prospect': prospect.get('data_prospect', ''),
                'ultima_atualizacao': prospect.get('ultima_atualizacao', ''),
                'comentario': prospect.get('comentario', ''),
                'recrutador': prospect.get('recrutador', '')
            }

            prospects_vaga.append(registro)
    df_prospects = pd.DataFrame(prospects_vaga)

    ### Get applications
    with open('../data/applicants.json','r',encoding='utf-8') as file:
        dados_candidatos = json.load(file)

    info_candidatos = []

    for candidato_id, candidato_info in dados_candidatos.items():
        infos_basicas = candidato_info.get('info_basicas',{})
        infos_prof = candidato_info.get('informacoes_profissionais',{})
        formacao = candidato_info.get('formacao_e_idiomas',{})

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

    ### Get vagas
    with open('../data/vagas.json', 'r', encoding='utf-8') as file:
        vagas = json.load(file)


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

    df_final_temp = pd.merge(
        df_prospects,
        df_candidatos,
        on='id_candidato',
        how='left'
    )

    
    df_final_temp['id_vaga'] = df_final_temp['id_vaga'].astype(str)
    df_vagas['id_vaga'] = df_vagas['id_vaga'].astype(str)

    df_final = pd.merge(
        df_final_temp,
        df_vagas,
        on='id_vaga',
        how='left'
    )

    df_final.to_pickle('../data/df_final.pkl')

if __name__ == "__main__":
    preprocessing()