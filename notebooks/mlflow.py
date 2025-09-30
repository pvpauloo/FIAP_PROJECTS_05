# train_mlflow.py
import os, json
from pathlib import Path
import joblib
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve


BUILD_DIR = Path("../app/model")
OUT_DIR = Path("../app/model")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def precision_at_k_per_job(y_true, y_score, job_ids, k=3):
    assert len(y_true) == len(y_score) == len(job_ids)
    jobs = np.unique(job_ids)
    precs = []
    for j in jobs:
        mask = (job_ids == j)
        if mask.sum() == 0:
            continue
        idx = np.argsort(-y_score[mask])
        topk = idx[:min(k, mask.sum())]
        yk = y_true[mask][topk]
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

def eval_cv(model_name, model, X, y, groups, k_s=(3,5), n_splits=5, log_to_mlflow=True):
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y), dtype=float)
    fold_metrics = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        m = model
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        oof[va] = p

        ap = average_precision_score(y[va], p)
        thr, f1_best = choose_best_threshold_by_f1(y[va], p)

        metrics = {"fold": fold, "auc_pr": float(ap), "f1_best": float(f1_best), "thr_best": float(thr)}
        for K in k_s:
            pk = precision_at_k_per_job(y[va], p, groups[va], k=K)
            metrics[f"p@{K}"] = float(pk)
        fold_metrics.append(metrics)

        # log por fold (como métricas separadas)
        if log_to_mlflow:
            mlflow.log_metrics({f"{model_name}_fold{fold}_aucpr": ap,
                                f"{model_name}_fold{fold}_f1best": f1_best,
                                f"{model_name}_fold{fold}_thrbest": thr,
                                **{f"{model_name}_fold{fold}_p@{K}": metrics[f"p@{K}"] for K in k_s}
                                }, step=fold)


    ap_mean = float(average_precision_score(y, oof))
    thr_global, f1_global = choose_best_threshold_by_f1(y, oof)
    agg = {
        "model": model_name,
        "auc_pr_oof": ap_mean,
        "f1_best_oof": float(f1_global),
        "thr_best_oof": float(thr_global),
    }
    for K in k_s:
        agg[f"p@{K}_oof"] = float(precision_at_k_per_job(y, oof, groups, k=K))

    if log_to_mlflow:
        mlflow.log_metrics({
            f"{model_name}_aucpr_oof": ap_mean,
            f"{model_name}_f1best_oof": f1_global,
            f"{model_name}_thrbest_oof": thr_global,
            **{f"{model_name}_p@{K}_oof": agg[f"p@{K}_oof"] for K in k_s}
        })

    return oof, fold_metrics, agg


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file://" + str((Path.cwd() / "mlruns").resolve()))
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "datathon-recrutamento")
    mlflow.set_experiment(experiment_name)


    X = np.load(BUILD_DIR / "X.npy", allow_pickle=True)
    y = np.load(BUILD_DIR / "y.npy", allow_pickle=True)
    groups = np.load(BUILD_DIR / "groups_job.npy", allow_pickle=True)
    arts = joblib.load(BUILD_DIR / "artifacts.joblib")
    feat_names = arts.get("feat_names", [f"f{i}" for i in range(X.shape[1])])

    n_splits = int(os.getenv("CV_SPLITS", "5"))

    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256,128),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1,
            random_state=42
        ),
    }

    with mlflow.start_run(run_name="train_cv") as run:
        mlflow.set_tags({
            "project": "vaga-match",
            "stage": os.getenv("STAGE", "dev"),
        })
        mlflow.log_params({
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "cv_splits": n_splits
        })

        for name, mdl in models.items():
            params = {f"{name}__{k}": v for k, v in mdl.get_params().items()}

            clean_params = {k: (str(v) if not isinstance(v, (int, float, str, bool)) else v)
                            for k, v in params.items()}
            mlflow.log_params(clean_params)


        results = {}
        best_name, best_ap = None, -1.0
        best_oof = None

        for name, mdl in models.items():
            oof, fold_metrics, agg = eval_cv(name, mdl, X, y, groups, k_s=(3,5), n_splits=n_splits, log_to_mlflow=True)
            results[name] = {"folds": fold_metrics, "agg": agg}
            if agg["auc_pr_oof"] > best_ap:
                best_ap = agg["auc_pr_oof"]
                best_name = name
                best_oof = oof

        best_model = models[best_name]
        best_model.fit(X, y)
        thr_best = results[best_name]["agg"]["thr_best_oof"]

        # salva local
        joblib.dump(best_model, OUT_DIR / "model.joblib")
        with open(OUT_DIR / "report.json", "w", encoding="utf-8") as f:
            json.dump({
                "best_model": best_name,
                "results": results,
                "feat_names": feat_names,
                "threshold_best_f1": thr_best,
            }, f, ensure_ascii=False, indent=2)
        np.save(OUT_DIR / "oof_scores.npy", best_oof)
        np.save(OUT_DIR / "labels.npy", y)
        np.save(OUT_DIR / "groups_job.npy", groups)


        mlflow.log_metric("best_aucpr_oof", best_ap)
        mlflow.log_metric("best_thrbest_oof", thr_best)
        mlflow.log_param("best_model_name", best_name)

        # artefatos úteis
        mlflow.log_artifact(OUT_DIR / "report.json", artifact_path="artifacts")
        mlflow.log_artifact(OUT_DIR / "oof_scores.npy", artifact_path="artifacts")
        mlflow.log_artifact(OUT_DIR / "labels.npy", artifact_path="artifacts")
        mlflow.log_artifact(OUT_DIR / "groups_job.npy", artifact_path="artifacts")

        # log do modelo no MLflow (com assinatura simples)
        signature = None
        try:
            import mlflow.models.signature as msign
            from mlflow.types.schema import Schema, ColSpec
            signature = msign.ModelSignature(
                inputs=Schema([ColSpec("double", name) for name in feat_names]),
                outputs=Schema([ColSpec("double", "score")])
            )
        except Exception:
            pass

        # exemplo de entrada (apenas para rastreabilidade)
        input_example = np.zeros((1, X.shape[1]))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=os.getenv("MLFLOW_REGISTER_MODEL_NAME") if os.getenv("MLFLOW_REGISTER_MODEL", "0") == "1" else None
        )

        print(f"[ok] melhor modelo: {best_name}")
        print(f"[ok] AUC-PR (OOF): {results[best_name]['agg']['auc_pr_oof']:.4f}")
        print(f"[ok] F1_best (OOF): {results[best_name]['agg']['f1_best_oof']:.4f} @thr={thr_best:.3f}")
        print(f"[ok] P@3 (OOF): {results[best_name]['agg']['p@3_oof']:.4f} | P@5 (OOF): {results[best_name]['agg']['p@5_oof']:.4f}")
        print(f"[ok] artefatos salvos em {OUT_DIR.resolve()}/")
        print(f"[ok] run_id: {run.info.run_id}")

if __name__ == "__main__":
    main()