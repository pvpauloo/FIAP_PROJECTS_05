import pandas as pd, numpy as np, re, unicodedata, joblib, json
from pathlib import Path
from typing import List, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def norm_txt(t: str) -> str:
    t = str(t or "")
    t = unicodedata.normalize("NFKD", t).encode("ascii","ignore").decode("ascii")
    t = re.sub(r"\s+"," ", t.lower()).strip()
    return t

def split_areas(s: str) -> List[str]:
    s = norm_txt(s)
    if not s:
        return []

    tokens = re.split(r"[;,/|\-•–—]+|\s{2,}", s)
    tokens = [tok.strip() for tok in tokens if tok.strip()]
    return tokens

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def map_level(val: str, mapping: dict) -> int:
    v = norm_txt(val)
    return mapping.get(v, 0)

def contains_kw(text: str, kw: str) -> int:
    return int(bool(re.search(rf"\b{re.escape(kw.lower())}\b", norm_txt(text))))

def any_kw(text: str, kws: Set[str]) -> int:
    text = norm_txt(text)
    return int(any(re.search(rf"\b{re.escape(k)}\b", text) for k in kws))

MAP_LVL = {
    "nenhum":0, "basico":1, "básico":1, "intermediario":2, "intermediário":2,
    "avancado":3, "avançado":3, "fluente":4
}

MAP_SENIOR = {
    "estagiario":1, "estagiário":1, "junior":2, "jr":2, "analista":3,
    "pleno":3, "senior":4, "sênior":4, "especialista":5
}

SEED_SKILLS = {
    "sap","control-m","controlm","sql","pl/sql","oracle","aws","azure","gcp",
    "linux","windows","vmware","jcl","abap","java","python","etl","bi","powercenter",
    "connect direct","b2b","devops","git","docker","kubernetes"
}

def build_features(df: pd.DataFrame, save_dir: str = "build"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    POS = {
        "contratado pela decision", 
        "aprovado", 
        "finalista", 
        "documentação pj", 
        "encaminhado ao cliente com aprovação",
        'contratado como hunting',
        'documentação clt',
        'encaminhar proposta',
        'proposta aceita'
    }
    sit = df["situacao_candidado"].fillna("").map(norm_txt)
    y = sit.isin(POS).astype(int).values

    job_text_raw = (df["atividades_vaga"].fillna("") + " " + df["competencias_vaga"].fillna(""))
    cv_text_raw  = df["cv_texto_pt"].fillna("")
    job_text = job_text_raw.map(norm_txt)
    cv_text  = cv_text_raw.map(norm_txt)

    tfidf = TfidfVectorizer(max_features=40000, ngram_range=(1,2), min_df=2)
    X_job = tfidf.fit_transform(job_text)
    X_cv  = tfidf.transform(cv_text)
    sim_tfidf = cosine_similarity(X_job, X_cv).diagonal()

    vi = df.get("nivel_ingles_vaga", "").fillna("").map(lambda x: map_level(x, MAP_LVL)).astype(int).values
    ci = df.get("applicant_nivel_ingles", "").fillna("").map(lambda x: map_level(x, MAP_LVL)).astype(int).values
    ingles_ok = (ci >= vi).astype(int)

    ve = df.get("nivel_espanhol_vaga", "").fillna("").map(lambda x: map_level(x, MAP_LVL)).astype(int).values
    ce = df.get("applicant_nivel_espanhol", "").fillna("").map(lambda x: map_level(x, MAP_LVL)).astype(int).values
    espanhol_ok = (ce >= ve).astype(int)

    vaga_sen = df.get("nivel_profissional_vaga","").fillna("").map(lambda x: map_level(x, MAP_SENIOR)).astype(int).values
    cand_sen = df.get("applicant_nivel_profissional","").fillna("").map(lambda x: map_level(x, MAP_SENIOR)).astype(int).values
    senior_ok = (cand_sen >= vaga_sen).astype(int)
    senior_gap = np.clip(cand_sen - vaga_sen, -3, 3)

    def flag_pair(kw: str):
        cv_f  = np.array([contains_kw(t, kw) for t in cv_text_raw], dtype=int)
        job_f = np.array([contains_kw(t, kw) for t in job_text_raw], dtype=int)
        return cv_f, job_f, (cv_f & job_f)

    cv_sap, job_sap, sap_match = flag_pair("sap")
    cv_ctrlm, job_ctrlm, ctrlm_match = flag_pair("control-m")
    cv_sql, job_sql, sql_match = flag_pair("sql")
    cv_aws, job_aws, aws_match = flag_pair("aws")
    cv_orc, job_orc, orc_match = flag_pair("oracle")

    vocab = tfidf.get_feature_names_out()

    mined = {v for v in vocab if (len(v.split())==1 and re.match(r"^[a-z][a-z0-9\-_\.]+$", v) and len(v)>=3)}

    mined = set(list(mined)[:500])
    skills = (SEED_SKILLS | mined)

    def count_skills(text: str, skills: Set[str]) -> int:
        text = norm_txt(text)
        c = 0
        for k in skills:
            if re.search(rf"\b{re.escape(k)}\b", text):
                c += 1
        return c

    skills_in_job = np.array([count_skills(t, skills) for t in job_text_raw], dtype=int)
    skills_in_cv  = np.array([count_skills(t, skills) for t in cv_text_raw], dtype=int)

    def overlap_count(jt: str, ct: str) -> int:
        jt = norm_txt(jt); ct = norm_txt(ct)
        sj = {k for k in skills if re.search(rf"\b{re.escape(k)}\b", jt)}
        sc = {k for k in skills if re.search(rf"\b{re.escape(k)}\b", ct)}
        return len(sj & sc)
    skills_overlap = np.array([overlap_count(j, c) for j, c in zip(job_text_raw, cv_text_raw)], dtype=int)


    areas_vaga = [split_areas(s) for s in df.get("areas_atuacao_vaga","").fillna("").tolist()]
    areas_cand = [split_areas(s) for s in df.get("applicant_area_atuacao","").fillna("").tolist()]
    area_jacc = np.array([jaccard(a, b) for a, b in zip(areas_vaga, areas_cand)], dtype=float)

 
    feats = np.c_[
        sim_tfidf,
        ingles_ok, espanhol_ok,
        senior_ok, senior_gap,
        sap_match, ctrlm_match, sql_match, aws_match, orc_match,
        cv_sap, cv_ctrlm, cv_sql, cv_aws, cv_orc,
        job_sap, job_ctrlm, job_sql, job_aws, job_orc,
        skills_in_job, skills_in_cv, skills_overlap,
        area_jacc
    ].astype(float)

    feat_names = [
        "sim_tfidf",
        "ingles_ok","espanhol_ok",
        "senior_ok","senior_gap",
        "sap_match","ctrlm_match","sql_match","aws_match","oracle_match",
        "cv_sap","cv_ctrlm","cv_sql","cv_aws","cv_oracle",
        "job_sap","job_ctrlm","job_sql","job_aws","job_oracle",
        "skills_in_job","skills_in_cv","skills_overlap",
        "area_jacc"
    ]


    groups_job = df["id_vaga"].values


    np.save(Path(save_dir)/"X.npy", feats)
    np.save(Path(save_dir)/"y.npy", y)
    np.save(Path(save_dir)/"groups_job.npy", groups_job)

    joblib.dump({
        "tfidf": tfidf,
        "feat_names": feat_names,
        "map_lvl": MAP_LVL,
        "map_senior": MAP_SENIOR,
        "skills_seed": sorted(SEED_SKILLS),
        "skills_mined_sample": sorted(list(mined))[:50],  
    }, Path(save_dir)/"artifacts.joblib")

    print(f"[ok] features salvas em '{save_dir}': X.npy {feats.shape}, y.npy {y.shape}, groups_job.npy {groups_job.shape}")
    print(f"[ok] exemplos de features: {feat_names[:8]} ... total={len(feat_names)}")

if __name__ == "__main__":
    df = pd.read_csv("../data/csv_df_final.csv")
    build_features(df, save_dir="../build")
