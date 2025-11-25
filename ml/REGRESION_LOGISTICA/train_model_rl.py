# ml/LOGISTIC_REGRESSION/train_model_lr.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    learning_curve,
    cross_val_score
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

DIR_OUT = Path("ml/REGRESION_LOGISTICA/resultados")
DIR_OUT.mkdir(parents=True, exist_ok=True)

# === PUBLICACIÓN PARA DESPLIEGUE (sin afectar lo actual)
DIR_MODELS = Path("ml/models")
DIR_MODELS.mkdir(parents=True, exist_ok=True)
RUTA_MODELO_DEPLOY   = DIR_MODELS / "model_v1.joblib"
RUTA_METRICAS_DEPLOY = DIR_MODELS / "metrics_v1.json"

RUTA_CURVA_IMG             = DIR_OUT / "curva_entrenamiento_validacion_lr.png"
RUTA_CURVA_CSV             = DIR_OUT / "curva_entrenamiento_validacion_lr.csv"
RUTA_CURVA_PERDIDA_IMG     = DIR_OUT / "curva_perdida_lr.png"  
RUTA_MATRIZ_CONFUSION      = DIR_OUT / "matriz_confusion_lr.png"
RUTA_METRICAS              = DIR_OUT / "metricas_lr.json"
RUTA_REPORTE_CLASIFICACION = DIR_OUT / "reporte_clasificacion_lr.txt"
RUTA_MODELO                = DIR_OUT / "modelo_lr.pkl"
RUTA_PARAMS                = DIR_OUT / "parametros_lr.json"

# =========================
# VARIABLES DEL MODELO
# =========================
FEATURES = [
    "age", "genero_bin",
    "phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8"
]
TARGET   = "nivel_idx"

CLASSES = ["Mínimo","Leve","Moderada","Moderadamente severa","Severa"]
CLASSES_FIG = ["Mínimo","Leve","Moderada","Moderadamente\nsevera","Severa"]

def idx_to_name(arr_int):
    return [CLASSES[i-1] for i in arr_int]

# =========================
# CONSTRUCTOR DEL MODELO LR
# =========================
def make_lr():
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=5000,
        random_state=SEED,
        n_jobs=-1
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

# ----------------------------------------------------------
# Curva de aprendizaje (Accuracy y Pérdida, exporta CSV)
# ----------------------------------------------------------
def construir_modelo_rl(X_train, y_train, cv, ruta_png, ruta_csv):
    modelo = make_lr()

    train_sizes_rel = np.linspace(0.1, 1.0, 8)
    sizes_abs, train_scores, valid_scores = learning_curve(
        estimator=modelo,
        X=X_train,
        y=y_train,
        train_sizes=train_sizes_rel,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        shuffle=True,
        random_state=SEED
    )

    tr_mean = train_scores.mean(axis=1); tr_std = train_scores.std(axis=1)
    va_mean = valid_scores.mean(axis=1); va_std = valid_scores.std(axis=1)

    loss_train_mean = 1.0 - tr_mean
    loss_valid_mean = 1.0 - va_mean

    df_curve = pd.DataFrame({
        "train_size": sizes_abs,
        "acc_train_cv_mean": np.round(tr_mean, 4),
        "acc_train_cv_std":  np.round(tr_std, 4),
        "acc_valid_cv_mean": np.round(va_mean, 4),
        "acc_valid_cv_std":  np.round(va_std, 4),
        "loss_train_mean":   np.round(loss_train_mean, 4),
        "loss_valid_mean":   np.round(loss_valid_mean, 4),
    })
    df_curve.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV de curva guardado en: {ruta_csv}")

    # -------------------------
    # FIGURA 1: Accuracy
    # -------------------------
    eps = 1e-3
    tr_line = np.clip(tr_mean, 0.0, 1.0 - eps)
    va_line = np.clip(va_mean, 0.0, 1.0 - eps)
    tr_low  = np.clip(tr_mean - tr_std, 0.0, 1.0)
    tr_high = np.clip(tr_mean + tr_std, 0.0, 1.0 - eps/2)
    va_low  = np.clip(va_mean - va_std, 0.0, 1.0)
    va_high = np.clip(va_mean + va_std, 0.0, 1.0 - eps/2)

    plt.figure(figsize=(7.8, 5.6))
    plt.plot(sizes_abs, tr_line, marker="o", label="Entrenamiento", color="tab:blue")
    plt.fill_between(sizes_abs, tr_low, tr_high, alpha=0.15, color="tab:blue")
    plt.plot(sizes_abs, va_line, marker="s", label="Validación", color="tab:orange")
    plt.fill_between(sizes_abs, va_low, va_high, alpha=0.15, color="tab:orange")

    plt.ylim(0.0, 1.0)
    plt.title("Curva de aprendizaje de Regresión Logística")
    plt.xlabel("Tamaño del conjunto de entrenamiento (TRAIN)")
    plt.ylabel("Accuracy (CV 5-fold)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=300)
    plt.close()
    print(f"[OK] Curva ENTRENAMIENTO/VALIDACIÓN (Accuracy CV-5) guardada en: {ruta_png}")

    # -------------------------
    # FIGURA 2: Pérdida (1 - accuracy)
    # -------------------------
    plt.figure(figsize=(7.8, 5.6))
    plt.plot(sizes_abs, loss_train_mean, marker="o", label="Pérdida entrenamiento")
    plt.plot(sizes_abs, loss_valid_mean, marker="s", label="Pérdida validación")

    plt.ylim(0.0, 1.0)
    plt.title("Curva de pérdida de Regresión Logística")
    plt.xlabel("Tamaño del conjunto de entrenamiento (TRAIN)")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUTA_CURVA_PERDIDA_IMG, dpi=300)
    plt.close()
    print(f"[OK] Curva de PÉRDIDA (1 - accuracy) guardada en: {RUTA_CURVA_PERDIDA_IMG}")

    return sizes_abs.tolist(), tr_mean.tolist(), va_mean.tolist(), tr_std.tolist(), va_std.tolist()

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Curva CV-5 (solo TRAIN)
    sizes_abs, tr_mean, va_mean, tr_std, va_std = construir_modelo_rl(
        X_train, y_train, cv, RUTA_CURVA_IMG, RUTA_CURVA_CSV
    )

    # Rendimiento promedio en TRAIN con CV-5
    cv_model = make_lr()
    cv_acc = cross_val_score(
        cv_model, X_train, y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )
    print(f"[CV-5] Accuracy (TRAIN) = {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    # Entrenamiento final y evaluación en TEST
    model = make_lr()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    proba_test = model.predict_proba(X_test)

    acc_test   = accuracy_score(y_test, y_pred)
    bacc_test  = balanced_accuracy_score(y_test, y_pred)
    f1m_test   = f1_score(y_test, y_pred, average="macro")
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro  = recall_score(y_test, y_pred, average="macro", zero_division=0)

    try:
        roc_auc_macro = roc_auc_score(y_test, proba_test, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc_macro = None

    print("\n=== MÉTRICAS EN TEST (GLOBAL / PROMEDIO ENTRE CLASES) ===")
    print(f"Accuracy                = {acc_test:.4f}")
    print(f"Balanced Accuracy       = {bacc_test:.4f}")
    print(f"Precisión (macro)       = {prec_macro:.4f}")
    print(f"Recall (macro)          = {rec_macro:.4f}")
    print(f"F1-Score (macro)        = {f1m_test:.4f}")
    print(f"ROC-AUC (macro OVR)     = {roc_auc_macro:.4f}" if roc_auc_macro is not None else "ROC-AUC (macro OVR)     = N/A")

    print("\n--- RESUMEN REGRESIÓN LOGÍSTICA (TEST HOLD-OUT 20%) ---")
    resumen_lr = (
        f"Regresión Logística — Accuracy = {acc_test:.2f}, Precisión = {prec_macro:.2f}, "
        f"Sensibilidad = {rec_macro:.2f}, F1-Score = {f1m_test:.2f}, "
        f"ROC-AUC = {roc_auc_macro:.2f}" if roc_auc_macro is not None
        else f"Regresión Logística — Accuracy = {acc_test:.2f}, Precisión = {prec_macro:.2f}, "
             f"Sensibilidad = {rec_macro:.2f}, F1-Score = {f1m_test:.2f}, ROC-AUC = N/A"
    )
    print(resumen_lr)

    rep = classification_report(
        idx_to_name(y_test.values),
        idx_to_name(y_pred),
        target_names=CLASSES,
        zero_division=0
    )
    print("\n=== REPORTE POR CLASE (TEST) ===")
    print(rep)

    cm = confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5])
    fig, ax = plt.subplots(figsize=(6.5,5))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES_FIG).plot(
        cmap="Blues", values_format="d", ax=ax, xticks_rotation=0
    )
    ax.set_title("Matriz de confusión: Regresión Logística", fontsize=14)
    ax.set_xlabel("Predicho", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
    plt.tight_layout()
    plt.savefig(RUTA_MATRIZ_CONFUSION, dpi=300)
    plt.close()
    print("[OK] Matriz de confusión guardada en:", RUTA_MATRIZ_CONFUSION)

    # Para el JSON también guardamos las pérdidas (1 - accuracy)
    loss_train_mean = [1.0 - float(x) for x in tr_mean]
    loss_valid_mean = [1.0 - float(x) for x in va_mean]

    # Exportar hiperparámetros reales del clasificador
    clf_real = model.named_steps["clf"]
    params_export = {
        "penalty": clf_real.penalty,
        "C": clf_real.C,
        "solver": clf_real.solver,
        "multi_class": clf_real.multi_class,
        "max_iter": clf_real.max_iter,
        "random_state": clf_real.random_state,
    }

    metricas = {
        "model": "LogisticRegression",
        "params": params_export,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "cv5_train": {
            "accuracy_mean": float(cv_acc.mean()),
            "accuracy_std":  float(cv_acc.std())
        },
        "curve": {
            "train_sizes": sizes_abs,
            "train_accuracy_mean": [float(x) for x in tr_mean],
            "train_accuracy_std":  [float(x) for x in tr_std],
            "valid_accuracy_mean": [float(x) for x in va_mean],
            "valid_accuracy_std":  [float(x) for x in va_std],
            "train_loss_mean":      loss_train_mean,
            "valid_loss_mean":      loss_valid_mean,
        },
        "test_metrics": {
            "accuracy": float(acc_test),
            "balanced_accuracy": float(bacc_test),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1m_test),
            "roc_auc_macro": (float(roc_auc_macro) if roc_auc_macro is not None else None)
        },
        "confusion_matrix": cm.tolist(),
        "labels_plot": CLASSES_FIG,
        "labels_full": CLASSES,
        "resumen": resumen_lr
    }

    with open(RUTA_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print("[OK] Métricas guardadas en:", RUTA_METRICAS)

    with open(RUTA_REPORTE_CLASIFICACION, "w", encoding="utf-8") as f:
        f.write(rep)
    print("[OK] Reporte de clasificación guardado en:", RUTA_REPORTE_CLASIFICACION)

    joblib.dump(model, RUTA_MODELO)
    with open(RUTA_PARAMS, "w", encoding="utf-8") as f:
        json.dump(
            {"features_used": FEATURES, "lr_params": params_export},
            f,
            ensure_ascii=False,
            indent=2
        )
    print("[OK] Modelo guardado en:", RUTA_MODELO)
    print("[OK] Parámetros guardados en:", RUTA_PARAMS)

    # ======= COPIA PARA DESPLIEGUE (ml/models/) =======
    with open(RUTA_METRICAS_DEPLOY, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print("[OK] Métricas (deploy) guardadas en:", RUTA_METRICAS_DEPLOY)

    joblib.dump(model, RUTA_MODELO_DEPLOY)
    print("[OK] Modelo (deploy) guardado en:", RUTA_MODELO_DEPLOY)

if __name__ == "__main__":
    main()
