# ranker.py
import json, re, unicodedata
from pathlib import Path
from typing import Dict, Any, List, Set

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ======== helpers (iguais aos do treino) ========

def norm_txt(t: str) -> str:
    t = str(t or "")
    t = unicodedata.normalize("NFKD", t).encode("ascii","ignore").decode("ascii")
    t = re.sub(r"\s+"," ", t.lower()).strip()
    return t

MAP_LVL = {
    "nenhum":0, "basico":1, "básico":1,
    "intermediario":2, "intermediário":2,
    "avancado":3, "avançado":3, "fluente":4
}
MAP_SENIOR = {
    "estagiario":1, "estagiário":1, "junior":2, "jr":2,
    "analista":3, "pleno":3, "senior":4, "sênior":4, "especialista":5
}

def map_level(val: str, mapping: dict, default=0) -> int:
    return mapping.get(norm_txt(val), default)

def contains_kw(text: str, kw: str) -> int:
    return int(bool(re.search(rf"\b{re.escape(kw.lower())}\b", norm_txt(text))))

def split_areas(s: str) -> List[str]:
    s = norm_txt(s)
    if not s: return []
    toks = re.split(r"[;,/|\-•–—]+|\s{2,}", s)
    return [t for t in (tok.strip() for tok in toks) if t]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb: return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

# ======== extração da vaga (aceita chaves com/sem espaço) ========

def extract_job_payload(job_json: Dict[str, Any]) -> Dict[str, Any]:
    pj = job_json.get("perfil_vaga", {}) or {}
    ib = job_json.get("informacoes_basicas", {}) or {}

    # alguns dumps usam "nivel profissional" (com espaço) e outros "nivel_profissional"
    nivel_prof = pj.get("nivel_profissional", pj.get("nivel profissional", ""))

    return {
        "atividades_vaga": pj.get("principais_atividades", "") or "",
        "competencias_vaga": pj.get("competencia_tecnicas_e_comportamentais", "") or "",
        "vaga_sap": (str(ib.get("vaga_sap", "")).strip().lower() == "sim"),
        "nivel_ingles_vaga": pj.get("nivel_ingles", "") or "Nenhum",
        "nivel_espanhol_vaga": pj.get("nivel_espanhol", "") or "Nenhum",
        "nivel_profissional_vaga": nivel_prof or "",
        "areas_atuacao_vaga": pj.get("areas_atuacao", "") or "",
        "titulo_vaga": ib.get("titulo_vaga", ""),
        "id_vaga": next(iter(job_json.keys()), None)  # opcional se vier envelopado por ID
    }

# ======== construção das MESMAS features do treino (vaga x 1 candidato) ========

def build_pair_features(job_payload: Dict[str, Any],
                        cand_row: Dict[str, Any],
                        tfidf, feat_cfg) -> np.ndarray:
    # textos
    job_text_raw = (job_payload["atividades_vaga"] or "") + " " + (job_payload["competencias_vaga"] or "")
    cv_text_raw  = cand_row.get("cv_texto_pt", "") or ""

    job_text = norm_txt(job_text_raw)
    cv_text  = norm_txt(cv_text_raw)

    # TF-IDF + similaridade (usa o mesmo vocabulário salvo)
    X_job = tfidf.transform([job_text])
    X_cv  = tfidf.transform([cv_text])
    sim_tfidf = float(cosine_similarity(X_job, X_cv)[0,0])

    # Idiomas
    vi = map_level(job_payload.get("nivel_ingles_vaga",""), MAP_LVL)
    ci = map_level(cand_row.get("applicant_nivel_ingles",""), MAP_LVL)
    ingles_ok = int(ci >= vi)

    ve = map_level(job_payload.get("nivel_espanhol_vaga",""), MAP_LVL)
    ce = map_level(cand_row.get("applicant_nivel_espanhol",""), MAP_LVL)
    espanhol_ok = int(ce >= ve)

    # Senioridade
    vaga_sen = map_level(job_payload.get("nivel_profissional_vaga",""), MAP_SENIOR)
    cand_sen = map_level(cand_row.get("applicant_nivel_profissional",""), MAP_SENIOR)
    senior_ok = int(cand_sen >= vaga_sen)
    senior_gap = float(np.clip(cand_sen - vaga_sen, -3, 3))

    # Flags de skills (SAP/Control-M/SQL/AWS/Oracle)
    def flag_pair(kw: str):
        cvf  = contains_kw(cv_text_raw, kw)
        jobf = contains_kw(job_text_raw, kw)
        return cvf, jobf, int(cvf & jobf)

    cv_sap, job_sap, sap_match = flag_pair("sap")
    cv_ctrlm, job_ctrlm, ctrlm_match = flag_pair("control-m")
    cv_sql, job_sql, sql_match = flag_pair("sql")
    cv_aws, job_aws, aws_match = flag_pair("aws")
    cv_orc, job_orc, orc_match = flag_pair("oracle")

    # Contagem de skills e overlap (seed + vocabulário treinado)
    vocab = set(getattr(tfidf, "get_feature_names_out", lambda: [])())
    seed = set(feat_cfg.get("skills_seed", []))
    mined_sample = set(feat_cfg.get("skills_mined_sample", []))  # só indicativo
    # reconstrução conservadora do conjunto de skills (seed + interseção com vocab)
    skills: Set[str] = seed | (vocab & mined_sample)

    def count_skills(text: str, skills: Set[str]) -> int:
        t = norm_txt(text)
        return sum(1 for k in skills if re.search(rf"\b{re.escape(k)}\b", t))

    skills_in_job = count_skills(job_text_raw, skills)
    skills_in_cv  = count_skills(cv_text_raw, skills)

    sj = {k for k in skills if re.search(rf"\b{re.escape(k)}\b", norm_txt(job_text_raw))}
    sc = {k for k in skills if re.search(rf"\b{re.escape(k)}\b", norm_txt(cv_text_raw))}
    skills_overlap = len(sj & sc)

    # Área (Jaccard)
    areas_vaga = split_areas(job_payload.get("areas_atuacao_vaga",""))
    areas_cand = split_areas(cand_row.get("applicant_area_atuacao",""))
    area_jacc = jaccard(areas_vaga, areas_cand)

    # ordem EXATA das features usada no treino
    x = np.array([[
        sim_tfidf,
        ingles_ok, espanhol_ok,
        senior_ok, senior_gap,
        sap_match, ctrlm_match, sql_match, aws_match, orc_match,
        cv_sap, cv_ctrlm, cv_sql, cv_aws, cv_orc,
        job_sap, job_ctrlm, job_sql, job_aws, job_orc,
        skills_in_job, skills_in_cv, skills_overlap,
        area_jacc
    ]], dtype=float)
    return x

# ======== ranking ========

def rank_candidates(job_json_in: Dict[str, Any],
                    applicants_json_path: str = "../data/applicants.json",
                    build_dir: str = "../build",
                    top_k: int = 10) -> pd.DataFrame:
    """Recebe 1 vaga (dict no formato do vagas.json) e retorna ranking Top-K de candidatos."""
    # 1) carrega modelo e artefatos
    build = Path(build_dir)
    model = joblib.load(build / "model.joblib")
    arts  = joblib.load(build / "artifacts.joblib")
    tfidf = arts["tfidf"]
    feat_names = arts.get("feat_names", [])

    # threshold (opcional, caso queira classificar também)
    thr = 0.5
    rep_path = build / "report.json"
    if rep_path.exists():
        try:
            rep = json.loads((build / "report.json").read_text(encoding="utf-8"))
            thr = float(rep.get("threshold_best_f1", 0.5))
        except Exception:
            pass

    # 2) carrega todos os candidatos
    with open(applicants_json_path, "r", encoding="utf-8") as f:
        applicants = json.load(f)

    # 3) extrai payload da vaga (suporta input no formato inteiro do exemplo)
    # se vier "5183": {...} embrulhado, desencaixa:
    if len(job_json_in) == 1 and isinstance(next(iter(job_json_in.values())), dict) and "perfil_vaga" in next(iter(job_json_in.values())):
        job_obj = next(iter(job_json_in.values()))
    else:
        job_obj = job_json_in
    job_payload = extract_job_payload(job_obj)

    # 4) monta um DataFrame com todos os candidatos e suas features vs essa vaga
    rows = []
    for cand_id, cand in applicants.items():
        infos_basicas = cand.get("infos_basicas", cand.get("info_basicas", {})) or {}
        infos_prof    = cand.get("informacoes_profissionais", {}) or {}
        formacao      = cand.get("formacao_e_idiomas", {}) or {}

        cand_row = {
            "id_candidato": cand_id,
            "applicant_nome": infos_basicas.get("nome", ""),
            "cv_texto_pt": cand.get("cv_pt", "") or "",
            "applicant_area_atuacao": infos_prof.get("area_atuacao", ""),
            "applicant_nivel_profissional": infos_prof.get("nivel_profissional", ""),
            "applicant_nivel_ingles": formacao.get("nivel_ingles", ""),
            "applicant_nivel_espanhol": formacao.get("nivel_espanhol", ""),
        }

        x = build_pair_features(job_payload, cand_row, tfidf, arts)
        score = float(model.predict_proba(x)[0,1])
        label = int(score >= thr)

        rows.append({
            "id_candidato": cand_id,
            "nome_candidato": cand_row["applicant_nome"],
            "score": score,
            "label_pred": label
        })

    df_rank = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    if top_k is not None and top_k > 0:
        return df_rank.head(top_k)
    return df_rank

# ======== uso por CLI rápido ========

if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--job_json_path", type=str, required=True, help="arquivo JSON com UMA vaga no formato do vagas.json (pode estar embrulhada pelo id)")
    p.add_argument("--applicants_json_path", type=str, default="../../data/applicants.json")
    p.add_argument("--build_dir", type=str, default="")
    p.add_argument("--top_k", type=int, default=10)
    args = p.parse_args()

    # lê a vaga
    job_in = json.loads(Path(args.job_json_path).read_text(encoding="utf-8"))
    top = rank_candidates(job_in, args.applicants_json_path, args.build_dir, args.top_k)
    # imprime CSV no stdout
    top.to_csv(sys.stdout, index=False)
