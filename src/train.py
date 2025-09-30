# train_v2.py
import json
from pathlib import Path
import numpy as np, joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BUILD = Path("build")
OUT = Path("build")
OUT.mkdir(parents=True, exist_ok=True)

def precision_at_k_per_job(y_true, y_score, job_ids, k=3):
    jobs = np.unique(job_ids)
    precs = []
    for j in jobs:
        m = (job_ids == j)
        if m.sum() == 0: 
            continue
        idx = np.argsort(-y_score[m])
        topk = idx[:min(k, m.sum())]
        yk = y_true[m][topk]
        precs.append(yk.mean() if len(yk) > 0 else 0.0)
    return float(np.mean(precs)) if len(precs) > 0 else 0.0

def choose_best_threshold_by_f1(y_true, y_score):
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    denom = (prec + rec)
    denom[denom == 0] = 1e-9
    f1s = 2 * (prec * rec) / denom
    thr_full = np.r_[thr, [1.0]]
    best_idx = int(np.nanargmax(f1s))
    return float(thr_full[best_idx]), float(f1s[best_idx])

def eval_cv(name, model, X, y, groups, n_splits=5, k_s=(3,5)):
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y), dtype=float)
    folds = []
    for f, (tr, va) in enumerate(gkf.split(X, y, groups)):
        m = model
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        oof[va] = p
        ap = average_precision_score(y[va], p)
        thr, f1b = choose_best_threshold_by_f1(y[va], p)
        metrics = {"fold": f, "auc_pr": float(ap), "f1_best": float(f1b), "thr_best": float(thr)}
        for K in k_s:
            metrics[f"p@{K}"] = precision_at_k_per_job(y[va], p, groups[va], k=K)
        folds.append(metrics)
    ap_oof = average_precision_score(y, oof)
    thr_oof, f1_oof = choose_best_threshold_by_f1(y, oof)
    agg = {"model": name, "auc_pr_oof": float(ap_oof), "f1_best_oof": float(f1_oof), "thr_best_oof": float(thr_oof)}
    for K in k_s:
        agg[f"p@{K}_oof"] = precision_at_k_per_job(y, oof, groups, k=K)
    return oof, folds, agg

def main():
    X = np.load(BUILD/"X.npy")
    y = np.load(BUILD/"y.npy")
    groups = np.load(BUILD/"groups_job.npy")
    arts = joblib.load(BUILD/"artifacts.joblib")
    feat_names = arts.get("feat_names", [f"f{i}" for i in range(X.shape[1])])

    # MODELOS (MLP adicionado):
    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "hgb": HistGradientBoostingClassifier(
            max_depth=None, learning_rate=0.08, max_iter=500,
            validation_fraction=None, l2_regularization=0.0
        ),
        "linsvc_cal": CalibratedClassifierCV(
            base_estimator=LinearSVC(class_weight="balanced"),
            method="sigmoid", cv=3
        ),
        "rf": RandomForestClassifier(
            n_estimators=600, max_depth=None, min_samples_split=2, min_samples_leaf=1,
            n_jobs=-1, class_weight="balanced_subsample", random_state=42
        ),
        # >>> MLP (com normalização) <<<
        "mlp": Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(256,128),
                activation="relu",
                alpha=1e-4,                # L2
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                n_iter_no_change=15,
                validation_fraction=0.1,
                random_state=42
            ))
        ]),
    }

    results = {}
    best_name, best_ap = None, -1.0
    best_oof, best_model = None, None
    for name, mdl in models.items():
        oof, folds, agg = eval_cv(name, mdl, X, y, groups, n_splits=5, k_s=(3,5))
        results[name] = {"folds": folds, "agg": agg}
        if agg["auc_pr_oof"] > best_ap:
            best_ap, best_name, best_oof, best_model = agg["auc_pr_oof"], name, oof, mdl

    # treina final com o melhor
    best_model.fit(X, y)
    thr = results[best_name]["agg"]["thr_best_oof"]

    joblib.dump(best_model, OUT/"model.joblib")
    with open(OUT/"report.json","w",encoding="utf-8") as f:
        json.dump({
            "best_model": best_name,
            "results": results,
            "feat_names": feat_names,
            "threshold_best_f1": float(thr)
        }, f, ensure_ascii=False, indent=2)

    np.save(OUT/"oof_scores.npy", best_oof)
    np.save(OUT/"labels.npy", y)
    np.save(OUT/"groups_job.npy", groups)

    print(f"[ok] melhor modelo: {best_name}")
    print(f"[ok] AUC-PR (OOF): {results[best_name]['agg']['auc_pr_oof']:.4f}")
    print(f"[ok] F1_best (OOF): {results[best_name]['agg']['f1_best_oof']:.4f} @thr={thr:.3f}")
    print(f"[ok] P@3 (OOF): {results[best_name]['agg']['p@3_oof']:.4f} | P@5 (OOF): {results[best_name]['agg']['p@5_oof']:.4f}")
    print(f"[ok] artefatos salvos em {OUT.resolve()}/")

if __name__ == "__main__":
    main()
