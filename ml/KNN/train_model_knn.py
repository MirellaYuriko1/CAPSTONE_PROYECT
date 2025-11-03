# ml/KNN/train_model_knn.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import joblib

# =========================
# CONFIGURACIÓN GLOBAL
# =========================
SEED = 42
RUTA_DATOS = Path("data/final/phq9_final.csv")

DIR_OUT = Path("ml/KNN/resultados")
DIR_OUT.mkdir(parents=True, exist_ok=True)

RUTA_CURVA_IMG             = DIR_OUT / "curva_entrenamiento_validacion_knn.png"
RUTA_CURVA_CSV             = DIR_OUT / "curva_entrenamiento_validacion_knn.csv"
RUTA_MATRIZ_CONFUSION      = DIR_OUT / "matriz_confusion_knn.png"
RUTA_METRICAS              = DIR_OUT / "metricas_knn.json"
RUTA_REPORTE_CLASIFICACION = DIR_OUT / "reporte_clasificacion_knn.txt"
RUTA_MODELO                = DIR_OUT / "modelo_knn.pkl"
RUTA_PARAMS                = DIR_OUT / "parametros_knn.json"

# =========================
# VARIABLES DEL MODELO
# =========================
FEATURES = [
    "age", "genero_bin",
    "phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"
]
TARGET = "categoryphq"

CLASSES = ["Mínimo", "Leve", "Moderada", "Moderadamente severa", "Severa"]
CLASSES_FIG = ["Mínimo", "Leve", "Moderada", "Moderadamente\nsevera", "Severa"]

def idx_to_name(arr_int):
    return [CLASSES[i-1] for i in arr_int]

# Hiperparámetros KNN (en pipeline con StandardScaler)
KNN_PARAMS = dict(
    n_neighbors=7,
    weights="distance",
    metric="minkowski",
    p=2
)

# ==========================================================
# CURVA DE APRENDIZAJE (Entrenamiento por CV interno + Validación CV externa)
# ==========================================================
def construir_modelo_knn(X_train, y_train, cv_externo, ruta_png, ruta_csv):
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(**KNN_PARAMS))
    ])

    train_sizes_rel = np.linspace(0.1, 1.0, 12)
    n = len(X_train)
    sizes_abs = np.unique(np.maximum(5, (train_sizes_rel * n).astype(int)))
    sizes_abs = [int(x) for x in sizes_abs] 

    tr_mean, tr_std, va_mean, va_std = [], [], [], []

    for m in sizes_abs:
        if m < n:
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train, train_size=m, stratify=y_train, random_state=SEED
            )
        else:
            X_sub, y_sub = X_train, y_train

        min_cls = y_sub.value_counts().min()
        k_interno = 5 if min_cls >= 5 else int(min_cls)
        if k_interno < 2:
            raise ValueError(
                f"m={m}: alguna clase tiene <2 ejemplos; no se puede CV estratificado."
            )
        cv_interno = StratifiedKFold(n_splits=k_interno, shuffle=True, random_state=SEED)

        tr_scores = cross_val_score(
            modelo, X_sub, y_sub, cv=cv_interno, scoring="accuracy", n_jobs=-1
        )
        va_scores = cross_val_score(
            modelo, X_sub, y_sub, cv=cv_externo, scoring="accuracy", n_jobs=-1
        )

        tr_mean.append(float(tr_scores.mean()))
        tr_std.append(float(tr_scores.std()))
        va_mean.append(float(va_scores.mean()))
        va_std.append(float(va_scores.std()))

    # ---- CSV con valores reales ----
    df_curve = pd.DataFrame({
        "train_size": sizes_abs,
        "acc_train_cv_mean": np.round(tr_mean, 4),
        "acc_train_cv_std":  np.round(tr_std, 4),
        "acc_valid_cv_mean": np.round(va_mean, 4),
        "acc_valid_cv_std":  np.round(va_std, 4),
    })
    df_curve.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV de curva guardado en: {ruta_csv}")

    # ---- Plot ----
    plt.figure(figsize=(7.8, 5.6))
    plt.plot(sizes_abs, tr_mean, marker="o", label="Entrenamiento")
    plt.fill_between(sizes_abs, np.array(tr_mean) - np.array(tr_std),
                                  np.array(tr_mean) + np.array(tr_std), alpha=0.15)

    plt.plot(sizes_abs, va_mean, marker="s", label="Validación")
    plt.fill_between(sizes_abs, np.array(va_mean) - np.array(va_std),
                                  np.array(va_mean) + np.array(va_std), alpha=0.15)

    plt.ylim(0.0, 1.0)
    plt.title("Curva de aprendizaje de KNN")
    plt.xlabel("Tamaño del conjunto de entrenamiento (TRAIN)")
    plt.ylabel("Accuracy (CV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=300)
    plt.close()
    print(f"[OK] Curva ENTRENAMIENTO/VALIDACIÓN guardada en: {ruta_png}")

    return list(sizes_abs), list(tr_mean), list(va_mean), list(tr_std), list(va_std)

# ==========================================================
# PROCESO PRINCIPAL
# ==========================================================
def main():
    if not RUTA_DATOS.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {RUTA_DATOS}")

    df = pd.read_csv(RUTA_DATOS)
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    assert not X.isna().any().any(), "Hay NaN en features; revisa el preprocesamiento."

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"[INFO] Entrenamiento = {len(X_train)} | Prueba = {len(X_test)}")

    # Ajuste automático de K según clase minoritaria (evita warnings)
    min_cls = y_train.value_counts().min()
    k = 5 if min_cls >= 5 else int(min_cls)
    if k < 2:
        raise ValueError("Alguna clase tiene <2 ejemplos; no es posible CV estratificado.")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)

    # Curva (Entrenamiento por CV interno + Validación por CV externo)
    sizes_abs, tr_mean, va_mean, tr_std, va_std = construir_modelo_knn(
        X_train, y_train, cv, RUTA_CURVA_IMG, RUTA_CURVA_CSV
    )

    # Rendimiento promedio en TRAIN (CV agregado)
    cv_model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(**KNN_PARAMS))
    ])
    cv_acc = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"[CV-{k}] Accuracy (TRAIN) = {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    # Entrenamiento final y evaluación en TEST
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(**KNN_PARAMS))
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    proba_test = model.predict_proba(X_test)

    acc_test   = float(accuracy_score(y_test, y_pred))
    bacc_test  = float(balanced_accuracy_score(y_test, y_pred))
    f1m_test   = float(f1_score(y_test, y_pred, average="macro"))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro  = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    try:
        roc_auc_macro = float(roc_auc_score(y_test, proba_test, multi_class="ovr", average="macro"))
    except ValueError:
        roc_auc_macro = None

    print("\n=== MÉTRICAS EN TEST (GLOBAL / PROMEDIO ENTRE CLASES) ===")
    print(f"Accuracy                = {acc_test:.4f}")
    print(f"Balanced Accuracy       = {bacc_test:.4f}")
    print(f"Precision (macro)       = {prec_macro:.4f}")
    print(f"Recall (macro)          = {rec_macro:.4f}")
    print(f"F1-Score (macro)        = {f1m_test:.4f}")
    if roc_auc_macro is not None:
        print(f"ROC-AUC (macro OVR)     = {roc_auc_macro:.4f}")
    else:
        print("ROC-AUC (macro OVR)     = N/A (clase ausente en test)")

    print("\n--- RESUMEN KNN (TEST HOLD-OUT 20%) ---")
    if roc_auc_macro is not None:
        resumen_knn = (
            f"KNN — Accuracy = {acc_test:.2f}, "
            f"Precisión = {prec_macro:.2f}, "
            f"Sensibilidad = {rec_macro:.2f}, "
            f"F1-Score = {f1m_test:.2f}, "
            f"ROC-AUC = {roc_auc_macro:.2f}"
        )
    else:
        resumen_knn = (
            f"KNN — Accuracy = {acc_test:.2f}, "
            f"Precisión = {prec_macro:.2f}, "
            f"Sensibilidad = {rec_macro:.2f}, "
            f"F1-Score = {f1m_test:.2f}, "
            f"ROC-AUC = N/A"
        )
    print(resumen_knn)

    rep = classification_report(
        idx_to_name(y_test.values), idx_to_name(y_pred), target_names=CLASSES, zero_division=0
    )
    print("\n=== REPORTE POR CLASE (TEST) ===")
    print(rep)

    cm = confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5])
    fig, ax = plt.subplots(figsize=(6.5,5))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES_FIG).plot(
        cmap="Blues", values_format="d", ax=ax, xticks_rotation=0
    )
    ax.set_title("Matriz de confusión: KNN", fontsize=14)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
    plt.tight_layout(); plt.savefig(RUTA_MATRIZ_CONFUSION, dpi=300); plt.close()
    print("[OK] Matriz de confusión guardada en:", RUTA_MATRIZ_CONFUSION)

    metricas = {
        "model": "KNN",
        "params": {"knn": KNN_PARAMS},
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "cv_train": {"k_folds": int(k), "accuracy_mean": float(cv_acc.mean()), "accuracy_std": float(cv_acc.std())},
        "curve": {
            "train_sizes": [int(x) for x in sizes_abs],
            "train_accuracy_cv_mean": [float(x) for x in tr_mean],
            "train_accuracy_cv_std":  [float(x) for x in tr_std],
            "valid_accuracy_cv_mean": [float(x) for x in va_mean],
            "valid_accuracy_cv_std":  [float(x) for x in va_std],
        },
        "test_metrics": {
            "accuracy": float(acc_test),
            "balanced_accuracy": float(bacc_test),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1m_test),
            "roc_auc_macro": (float(roc_auc_macro) if roc_auc_macro is not None else None)
        },
        "confusion_matrix": np.asarray(cm, dtype=int).tolist(),
        "labels_plot": CLASSES_FIG,
        "labels_full": CLASSES,
        "resumen": resumen_knn
    }
    with open(RUTA_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    with open(RUTA_REPORTE_CLASIFICACION, "w", encoding="utf-8") as f:
        f.write(rep)

    joblib.dump(model, RUTA_MODELO)
    with open(RUTA_PARAMS, "w", encoding="utf-8") as f:
        json.dump({"params_knn": KNN_PARAMS, "features_used": FEATURES}, f, ensure_ascii=False, indent=2)

    print("[OK] Modelo guardado en:", RUTA_MODELO)
    print("[OK] Parámetros guardados en:", RUTA_PARAMS)
    print("[OK] Métricas guardadas en:", RUTA_METRICAS)
    print("[OK] Reporte de clasificación guardado en:", RUTA_REPORTE_CLASIFICACION)

if __name__ == "__main__":
    main()
