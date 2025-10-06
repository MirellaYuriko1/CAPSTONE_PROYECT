# ml/train_model.py
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone  # <- para la vista previa del preprocesamiento

from Scas.configuracion import get_db  # conexión mysql-connector

MODEL_VERSION = "v1"   # mantén sincronizado con tu app
PREGUNTAS = [f"p{i}" for i in range(1, 39)]

SQL_ULTIMO_X_ALUMNO = f"""
SELECT u.id_usuario, c.id_cuestionario, c.edad, c.genero,
       {", ".join("c."+p for p in PREGUNTAS)},
       r.nivel
FROM (
    SELECT c1.*
    FROM cuestionario c1
    JOIN (
        SELECT id_usuario, MAX(created_at) AS mx
        FROM cuestionario
        GROUP BY id_usuario
    ) ult
      ON ult.id_usuario = c1.id_usuario AND ult.mx = c1.created_at
) c
JOIN usuario u        ON u.id_usuario = c.id_usuario
LEFT JOIN resultado r ON r.id_cuestionario = c.id_cuestionario
"""

def norm_label(s: str) -> str:
    s = (s or "").strip().lower()
    if "muy" in s:
        return "Muy alto"
    if s == "alto":
        return "Alto"
    if "elev" in s:
        return "Elevado"
    return "Normal"

def main():
    print("[ML] Iniciando entrenamiento…")

    # 1) Datos
    print("[ML] Conectando a BD…")
    cn = get_db()
    cur = cn.cursor(dictionary=True)
    cur.execute(SQL_ULTIMO_X_ALUMNO)
    rows = cur.fetchall()
    cur.close(); cn.close()

    df = pd.DataFrame(rows)
    print(f"[ML] Registros leídos: {len(df)}")

    # Opcional: para que las tablas salgan bien en consola
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 60)

    # ==========================================================================
    # EVIDENCIA 1: Depuración y normalización de la etiqueta
    # ==========================================================================
    print("\n--- Depuración y normalización de la etiqueta ---")
    antes_total = len(df)
    nulos_nivel = df["nivel"].isna().sum()
    print(f"Filas totales (antes): {antes_total}")
    print(f"Nulos en 'nivel': {nulos_nivel}")

    # Depuración + normalización
    df = df.dropna(subset=["nivel"]).copy()
    df["nivel_norm"] = df["nivel"].map(norm_label) 

    despues_total = len(df)
    print(f"Filas totales (después): {despues_total}  | Eliminadas: {antes_total - despues_total}")

    print("\nClases normalizadas (conteo):")
    print(df["nivel_norm"].value_counts().sort_index().to_string())

    # -------------------------------------------------------------------------
    # Selección de variables
    feature_cols_num = PREGUNTAS + ["edad"]
    feature_cols_cat = ["genero"]
    X = df[feature_cols_num + feature_cols_cat].copy()
    y = df["nivel_norm"].copy()
    classes_sorted = sorted(y.unique())

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ]
    )

    # ==========================================================================
    # EVIDENCIA 2: Selección de variables y codificación de género
    # ==========================================================================
    print("\n--- Selección de variables y codificación de género ---")
    print(f"Cols numéricas ({len(feature_cols_num)}): {feature_cols_num}")
    print(f"Cols categóricas ({len(feature_cols_cat)}): {feature_cols_cat}")

    print("\nValores de 'genero' (conteo):")
    print(df["genero"].value_counts(dropna=False).to_string())

    # Vista previa segura del preprocesamiento (no afecta al pipeline de CV)
    pre_preview = clone(pre)
    pre_preview.fit(X)

    # Categorías que detectó OneHotEncoder para 'genero'
    ohe = pre_preview.named_transformers_["cat"]
    cats = list(ohe.categories_[0])
    print("\nCategorías codificadas de 'genero':", cats)

    # Nombres de columnas resultantes tras el preprocesamiento
    try:
        feat_names = pre_preview.get_feature_names_out()
    except Exception:
        feat_names = feature_cols_num + [f"cat__genero_{c}" for c in cats]

    print(f"Total de columnas de salida: {len(feat_names)}")
    print("Ejemplo de columnas transformadas:")
    for n in list(feat_names)[:10]:
        print("  -", n)

    cols_preview = (feature_cols_num[:5] + feature_cols_cat)
    print("\nMuestra rápida de X (5 filas):")
    print(X[cols_preview].head(5).to_string(index=False))

    # -------------------------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    clf = Pipeline(steps=[("preprocessor", pre), ("model", model)])

    # 2) Validación cruzada (k=5, estratificada)
    print("\n=== Validación Cruzada (5-fold, macro-F1) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro")
    cv_acc = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    print(f"F1_macro CV: mean={cv_f1.mean():.3f}  std={cv_f1.std():.3f}")
    print(f"Accuracy  CV: mean={cv_acc.mean():.3f}  std={cv_acc.std():.3f}")

    # Tamaños de entrenamiento/validación por fold (solo para informar cantidades)
    train_sizes_cv, val_sizes_cv = [], []
    for tr_idx, val_idx in skf.split(X, y):
        train_sizes_cv.append(len(tr_idx))
        val_sizes_cv.append(len(val_idx))

    # === Predicción out-of-fold para evaluación general ===
    y_pred_cv = cross_val_predict(clf, X, y, cv=skf)

    df_eval = pd.DataFrame({
        "genero": df["genero"],
        "y_true": y.values,
        "y_pred": y_pred_cv
    })

    def _group_report(mask):
        yt = df_eval.loc[mask, "y_true"]
        yp = df_eval.loc[mask, "y_pred"]
        if yt.empty:
            return None
        rep = classification_report(yt, yp, output_dict=True, zero_division=0)
        acc_g = accuracy_score(yt, yp)
        f1m_g = f1_score(yt, yp, average="macro")
        cm_g = confusion_matrix(yt, yp, labels=classes_sorted).tolist()
        return {
            "n": int(len(yt)),
            "accuracy": float(acc_g),
            "f1_macro": float(f1m_g),
            "classification_report": rep,
            "confusion_matrix": {"labels": classes_sorted, "matrix": cm_g},
        }

    subgroups_genero = {}
    for g in df_eval["genero"].dropna().unique():
        subgroups_genero[g] = _group_report(df_eval["genero"] == g)

    print("\n=== Subgrupos por género (cross-val) ===")
    for g, m in subgroups_genero.items():
        if m:
            print(f"Genero={g}: n={m['n']}, acc={m['accuracy']:.2f}, f1_macro={m['f1_macro']:.2f}")

    # ⬇️ NUEVO: Resumen general (una sola vez) con todo el dataset usando OOF
    correct_cv = int((y_pred_cv == y).sum())
    total_cv = int(len(y))
    acc_cv_pct = 100.0 * correct_cv / total_cv
    err_cv_pct = 100.0 - acc_cv_pct
    print("\n=== Resumen general (Validación cruzada OOF) ===")
    print(f"Predicciones correctas: {correct_cv}/{total_cv} ({acc_cv_pct:.2f}%)")
    print(f"Promedio de error: {err_cv_pct:.2f}%  (1 - accuracy)")

    # 3) Hold-out final (20%) — mantenemos reporte detallado, sin imprimir % extra
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)

    print("\n=== Reporte (test hold-out) ===")
    print(classification_report(yte, ypred))

    cm = confusion_matrix(yte, ypred, labels=classes_sorted)
    acc = accuracy_score(yte, ypred)
    f1m = f1_score(yte, ypred, average="macro")

    # === Conteos solicitados (Total / Entrenamiento / Validación / Testeo) ===
    n_total = len(X)
    n_train = len(Xtr)
    n_test  = len(Xte)
    print("\n=== Tamaños de conjuntos ===")
    print(f"Total: {n_total}")
    print(f"Entrenamiento (hold-out): {n_train}")
    print(f"Validación (CV k=5, por fold): {val_sizes_cv}")
    print(f"Testeo (hold-out): {n_test}")
    print(f"\nResumen -> Total={n_total} | Entrenamiento={n_train} | Validación(CV por fold)={val_sizes_cv} | Testeo={n_test}")

    # 4) Guardar modelo y métricas (sin cambios en el esquema)
    outdir = Path(__file__).parent / "models"
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "model_v1.joblib"
    joblib.dump(clf, model_path)

    metrics = {
        "model_version": MODEL_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_registros": int(len(df)),
        "clases": classes_sorted,
        "cv": {
            "n_splits": 5,
            "f1_macro_mean": float(cv_f1.mean()),
            "f1_macro_std": float(cv_f1.std()),
            "accuracy_mean": float(cv_acc.mean()),
            "accuracy_std": float(cv_acc.std()),
            "train_sizes_per_fold": train_sizes_cv,
            "val_sizes_per_fold": val_sizes_cv,
        },
        "subgroups": {
            "genero": subgroups_genero
        },
        "holdout": {
            "accuracy": float(acc),
            "f1_macro": float(f1m),
            "classification_report": classification_report(yte, ypred, output_dict=True),
            "confusion_matrix": cm.tolist(),
            "labels_order": classes_sorted,
            "n_test": int(len(yte)),
            "n_train": int(len(ytr)),
            "n_total": int(n_total),
        },
    }
    with open(outdir / "metrics_v1.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Modelo guardado en: {model_path.resolve()}")
    print(f"📝 Métricas guardadas en: {(outdir / 'metrics_v1.json').resolve()}")

if __name__ == "__main__":
    main()
