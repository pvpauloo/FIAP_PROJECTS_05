from fastapi import HTTPException, Query, status
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import hashlib
import re, unicodedata
# caminhos
model = joblib.load("app/model/model.joblib")
arts  = joblib.load("app/model/artifacts.joblib")
tfidf = arts["tfidf"]
feat_names = arts["feat_names"]  # ordem oficial das features
MAP_LVL = arts.get("map_lvl", {})
MAP_SENIOR = arts.get("map_senior", {})
skills = set(arts.get("skills_seed", [])) | set(arts.get("skills_mined_sample", []))

# --- 1. Modelos Pydantic ---

class CandidateRank(BaseModel):
    id_candidato: int
    nome_candidato: str
    score: float

class RankingResponse(BaseModel):
    ranking: List[CandidateRank]

class JobCreate(BaseModel):
    titulo_vaga_detalhado: str
    cliente_vaga: Optional[str] = None
    tipo_contratacao: Optional[str] = None
    nivel_profissional_vaga: Optional[str] = None
    atividades_vaga: str
    competencias_vaga: str

class Job(JobCreate):
    id_vaga: int

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

def _norm_txt(t: str) -> str:
    t = str(t or "")
    t = unicodedata.normalize("NFKD", t).encode("ascii","ignore").decode("ascii")
    t = re.sub(r"\s+"," ", t.lower()).strip()
    return t

def _contains_kw(text: str, kw: str) -> int:
    return int(bool(re.search(rf"\b{re.escape(kw.lower())}\b", _norm_txt(text))))

def _split_areas(s: str):
    s = _norm_txt(s)
    if not s: return []
    return [tok.strip() for tok in re.split(r"[;,/|\-•–—]+|\s{2,}", s) if tok.strip()]

def _jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb: return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def _map_level(val: str, mapping: dict) -> int:
    return mapping.get(_norm_txt(val), 0)

def _count_skills(text: str, skills: set) -> int:
    txt = _norm_txt(text)
    c = 0
    for k in skills:
        if re.search(rf"\b{re.escape(k)}\b", txt):
            c += 1
    return c

def _skills_overlap(job_text: str, cv_text: str, skills: set) -> int:
    jt = _norm_txt(job_text); ct = _norm_txt(cv_text)
    sj = {k for k in skills if re.search(rf"\b{re.escape(k)}\b", jt)}
    sc = {k for k in skills if re.search(rf"\b{re.escape(k)}\b", ct)}
    return len(sj & sc)

def build_feature_row(row):
    # textos base (iguais ao treino)
    job_text = str(row.get("job_text", "")) or (
        str(row.get("atividades_vaga","")) + " " + str(row.get("competencias_vaga",""))
    )
    cv_text  = str(row.get("cv_text", "")) or str(row.get("cv_texto_pt",""))

    # sim_tfidf
    X_job = tfidf.transform([_norm_txt(job_text)])
    X_cv  = tfidf.transform([_norm_txt(cv_text)])
    sim_tfidf = float(cosine_similarity(X_job, X_cv)[0,0])

    # idiomas
    vi = _map_level(row.get("nivel_ingles_vaga",""), MAP_LVL)
    ci = _map_level(row.get("applicant_nivel_ingles",""), MAP_LVL)
    ingles_ok = int(ci >= vi)

    ve = _map_level(row.get("nivel_espanhol_vaga",""), MAP_LVL)
    ce = _map_level(row.get("applicant_nivel_espanhol",""), MAP_LVL)
    espanhol_ok = int(ce >= ve)

    # senioridade
    vaga_sen  = _map_level(row.get("nivel_profissional_vaga",""), MAP_SENIOR)
    cand_sen  = _map_level(row.get("applicant_nivel_profissional",""), MAP_SENIOR)
    senior_ok = int(cand_sen >= vaga_sen)
    senior_gap = int(np.clip(cand_sen - vaga_sen, -3, 3))

    # flags kw (no job e no cv)
    def flag_pair(kw: str):
        cv_f  = _contains_kw(cv_text, kw)
        job_f = _contains_kw(job_text, kw)
        return cv_f, job_f, int(cv_f & job_f)

    cv_sap, job_sap, sap_match       = flag_pair("sap")
    cv_ctrlm, job_ctrlm, ctrlm_match = flag_pair("control-m")
    cv_sql, job_sql, sql_match       = flag_pair("sql")
    cv_aws, job_aws, aws_match       = flag_pair("aws")
    cv_orc, job_orc, orc_match       = flag_pair("oracle")

    # skills
    skills_in_job = _count_skills(job_text, skills)
    skills_in_cv  = _count_skills(cv_text, skills)
    skills_ovlp   = _skills_overlap(job_text, cv_text, skills)

    # áreas
    areas_vaga = _split_areas(row.get("areas_atuacao_vaga",""))
    areas_cand = _split_areas(row.get("applicant_area_atuacao",""))
    area_jacc  = float(_jaccard(areas_vaga, areas_cand))

    # Mapa: nome->valor (mesma semântica do treino)
    feats = {
        "sim_tfidf": sim_tfidf,
        "ingles_ok": ingles_ok,
        "espanhol_ok": espanhol_ok,
        "senior_ok": senior_ok,
        "senior_gap": senior_gap,
        "sap_match": sap_match,
        "ctrlm_match": ctrlm_match,
        "sql_match": sql_match,
        "aws_match": aws_match,
        "oracle_match": orc_match,
        "cv_sap": cv_sap,
        "cv_ctrlm": cv_ctrlm,
        "cv_sql": cv_sql,
        "cv_aws": cv_aws,
        "cv_oracle": cv_orc,
        "job_sap": job_sap,
        "job_ctrlm": job_ctrlm,
        "job_sql": job_sql,
        "job_aws": job_aws,
        "job_oracle": job_orc,
        "skills_in_job": skills_in_job,
        "skills_in_cv": skills_in_cv,
        "skills_overlap": skills_ovlp,
        "area_jacc": area_jacc,
    }
    # retorna vetor NA MESMA ORDEM de feat_names
    return [feats.get(name, 0.0) for name in feat_names]


def select_fixed_candidates_for_job(job_id: str, df_candidatos: pd.DataFrame, k: int = 100) -> pd.DataFrame:
    """
    Seleciona determinística e estavelmente k candidatos para uma vaga.
    Critério: ordena por hash SHA-256(job_id + id_candidato), crescente
    """
    df = df_candidatos.copy()
    if "id_candidato" not in df.columns:
        raise ValueError("Coluna 'id_candidato' não encontrada em applicants_df")

    def hash_rank(cand_id: str) -> int:
        s = f"{job_id}:{cand_id}"
        return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)

    df["__rank_hash"] = df["id_candidato"].astype(str).map(hash_rank)
    df = df.sort_values(["__rank_hash", "id_candidato"], ascending=[True, True]).drop(columns="__rank_hash")
    df = df.assign(id_vaga=str(job_id))
    return df.head(min(k, len(df)))


# --- 2. Função para Registrar Rotas ---

def register_routes(app, df_vagas, df_candidatos):

    @app.get("/", tags=["Health Check"])
    def read_root():
        return {"status": "ok", "message": "API online"}

    @app.post("/rank/{job_id}", response_model=RankingResponse, tags=["Ranking"])
    def get_top_candidates_for_job(job_id: int, top_n: int = Query(5, ge=1, le=50)):
        vaga_df = df_vagas[df_vagas["id_vaga"] == str(job_id)]
        if vaga_df.empty:
            raise HTTPException(status_code=404, detail=f"Vaga com ID {job_id} não encontrada.")

        try:
            # candidatos candidatos “fixos” para a vaga (defina sua estratégia)
            df_cands = select_fixed_candidates_for_job(job_id, df_candidatos, max(top_n*5, 50))
            if df_cands is None or df_cands.empty:
                return {"ranking": []}

            # normaliza dtypes para merge
            vaga_df_norm = vaga_df.copy()
            vaga_df_norm["id_vaga"] = vaga_df_norm["id_vaga"].astype(str)

            out_cands = df_cands.copy()
            out_cands["id_vaga"] = str(job_id)

            df_result = pd.merge(
                out_cands,
                vaga_df_norm,
                on="id_vaga",
                how="left",
                validate="m:1",
            )

            # garante colunas de texto
            if "job_text" not in df_result.columns:
                df_result["job_text"] = (
                    df_result.get("atividades_vaga","").fillna("").astype(str) + " " +
                    df_result.get("competencias_vaga","").fillna("").astype(str)
                )
            if "cv_text" not in df_result.columns:
                base_cv = "cv_texto_pt" if "cv_texto_pt" in df_result.columns else "cv_pt"
                df_result["cv_text"] = df_result.get(base_cv,"").fillna("").astype(str)

            # === >>> AQUI: mesmas 24 features do treino <<< ===
            X_rows = [build_feature_row(row) for _, row in df_result.iterrows()]
            X_new = np.array(X_rows, dtype=float)

            # checagem de sanidade
            if X_new.shape[1] != len(feat_names):
                raise HTTPException(status_code=500,
                                    detail=f"Dimensão de X_new ({X_new.shape[1]}) difere de feat_names ({len(feat_names)}).")

            probs = model.predict_proba(X_new)[:, 1]
            df_scored = df_result.copy()
            df_scored["score"] = probs

            # normaliza id_vaga no payload (opcional)
            if "id_vaga" in df_scored.columns:
                df_scored["id_vaga"] = pd.to_numeric(df_scored["id_vaga"], errors="coerce").astype("Int64")

            # ==== Fallbacks exigidos pelo response_model ====
            # nome_candidato pode vir com outros nomes
            if "nome_candidato" not in df_scored.columns:
                if "applicant_nome" in df_scored.columns:
                    df_scored["nome_candidato"] = df_scored["applicant_nome"].astype(str)
                elif "nome" in df_scored.columns:
                    df_scored["nome_candidato"] = df_scored["nome"].astype(str)
                elif "nome_cand" in df_scored.columns:
                    df_scored["nome_candidato"] = df_scored["nome_cand"].astype(str)
                else:
                    df_scored["nome_candidato"] = ""

            # id_candidato pode ter outro nome
            if "id_candidato" not in df_scored.columns:
                if "codigo_profissional" in df_scored.columns:
                    df_scored["id_candidato"] = pd.to_numeric(
                        df_scored["codigo_profissional"], errors="coerce"
                    ).fillna(-1).astype(int)
                else:
                    df_scored["id_candidato"] = -1  # fallback

            # score não pode ser NaN/inf para o Pydantic
            df_scored["score"] = pd.to_numeric(df_scored["score"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Seleciona colunas e ordena
            cols_payload = ["id_candidato", "nome_candidato", "score"]
            top = (
                df_scored.sort_values("score", ascending=False)
                         .head(top_n)
                         [cols_payload]
                         .copy()
            )

            # Cast final seguro de tipos (evita None no Pydantic)
            top["id_candidato"] = pd.to_numeric(top["id_candidato"], errors="coerce").fillna(-1).astype(int)
            top["nome_candidato"] = top["nome_candidato"].astype(str)
            top["score"] = pd.to_numeric(top["score"], errors="coerce").fillna(0.0).astype(float)

            # Constrói objetos Pydantic explícitos
            ranking_items = [
                CandidateRank(
                    id_candidato=int(row["id_candidato"]),
                    nome_candidato=str(row["nome_candidato"]),
                    score=float(row["score"]),
                )
                for _, row in top.iterrows()
            ]

            # Retorno como modelo Pydantic (nunca None)
            return RankingResponse(ranking=ranking_items)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Falha ao ranquear candidatos: {e}")

    @app.get("/jobs/", response_model=PaginatedJobResponse, tags=["Vagas"])
    def list_jobs(skip: int = 0, limit: int = 20):
        total_jobs = len(df_vagas)
        jobs_slice = df_vagas.iloc[skip: skip + limit]
        return {"total_jobs": total_jobs, "jobs": jobs_slice.to_dict(orient='records')}

    @app.get("/jobs/{job_id}", response_model=Job, tags=["Vagas"])
    def get_job_by_id(job_id: int):
        vaga = df_vagas[df_vagas['id_vaga'] == job_id]
        if vaga.empty:
            raise HTTPException(status_code=404, detail=f"Vaga com ID {job_id} não encontrada.")
        return vaga.iloc[0].to_dict()

    @app.post("/jobs/", response_model=Job, status_code=status.HTTP_201_CREATED, tags=["Vagas"])
    def create_job(job: JobCreate):
        nonlocal df_vagas
        new_id = df_vagas['id_vaga'].max() + 1 if not df_vagas.empty else 1
        new_job_data = job.model_dump()
        new_job_data['id_vaga'] = new_id
        new_job_df = pd.DataFrame([new_job_data])
        df_vagas = pd.concat([df_vagas, new_job_df], ignore_index=True)
        return new_job_data

    @app.get("/candidates/", response_model=PaginatedCandidateResponse, tags=["Candidatos"])
    def list_candidates(skip: int = 0, limit: int = 20):
        total_candidates = len(df_candidatos)
        candidates_slice = df_candidatos.iloc[skip: skip + limit]
        return {"total_candidates": total_candidates, "candidates": candidates_slice.to_dict(orient='records')}
