# ml/NAIVE_BAYES/train_model_nb.py
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
from sklearn.naive_bayes import GaussianNB
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

DIR_OUT = Path("ml/NAIVE_BAYES/resultados")
DIR_OUT.mkdir(parents=True, exist_ok=True)

RUTA_CURVA_IMG             = DIR_OUT / "curva_entrenamiento_validacion_nb.png"
RUTA_CURVA_CSV             = DIR_OUT / "curva_entrenamiento_validacion_nb.csv"
RUTA_MATRIZ_CONFUSION      = DIR_OUT / "matriz_confusion_nb.png"
RUTA_METRICAS              = DIR_OUT / "metricas_nb.json"
RUTA_REPORTE_CLASIFICACION = DIR_OUT / "reporte_clasificacion_nb.txt"
RUTA_MODELO                = DIR_OUT / "modelo_nb.pkl"
RUTA_PARAMS                = DIR_OUT / "parametros_nb.json"

# =========================
# VARIABLES DEL MODELO
# =========================
FEATURES = [
    "age", "genero_bin",
    "phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"
]
TARGET = "categoryphq"  # valores 1..5

CLASSES = ["Mínimo","Leve","Moderada","Moderadamente severa","Severa"]
CLASSES_FIG = ["Mínimo","Leve","Moderada","Moderadamente\nsevera","Severa"]

def idx_to_name(arr_int):
    return [CLASSES[i-1] for i in arr_int]

# =========================
# CONSTRUCTOR DEL MODELO NB
# =========================
def make_nb():
    return GaussianNB()

# ==========================================================
# FUNCIÓN: CURVA DE APRENDIZAJE (ENTRENAMIENTO)
# ==========================================================
def curva_nb_cv5(X_train, y_train, cv, ruta_png, ruta_csv):
    
    modelo = make_nb()
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

    tr_mean = train_scores.mean(axis=1)
    tr_std  = train_scores.std(axis=1)
    va_mean = valid_scores.mean(axis=1)
    va_std  = valid_scores.std(axis=1)

    # ----- exporte CSV -----
    df_curve = pd.DataFrame({
        "train_size": sizes_abs,
        "acc_train_cv_mean": np.round(tr_mean, 4),
        "acc_train_cv_std":  np.round(tr_std, 4),
        "acc_valid_cv_mean": np.round(va_mean, 4),
        "acc_valid_cv_std":  np.round(va_std, 4),
    })
    df_curve.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV de curva guardado en: {ruta_csv}")

    # ----- figura SIN números sobre los puntos, eje Y = 0..1 -----
    # pequeño clip para evitar que el sombreado "corte" en 1.0
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
    plt.title("Curva de aprendizaje de Naive Bayes")
    plt.xlabel("Tamaño del conjunto de entrenamiento (TRAIN)")
    plt.ylabel("Accuracy (CV 5-fold)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=300)
    plt.close()
    print(f"[OK] Curva ENTRENAMIENTO/VALIDACIÓN (CV-5) guardada en: {ruta_png}")

    return sizes_abs.tolist(), tr_mean.tolist(), va_mean.tolist(), tr_std.tolist(), va_std.tolist()

# ==========================================================
# PROCESO PRINCIPAL
# ==========================================================
def main():
    if not RUTA_DATOS.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {RUTA_DATOS}")

    df = pd.read_csv(RUTA_DATOS)
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()  # 1..5
    assert not X.isna().any().any(), "Hay NaN en features; revisa el preprocesamiento."

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"[INFO] Entrenamiento = {len(X_train)} | Prueba = {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Curva CV-5 (TRAIN)
    sizes_abs, tr_mean, va_mean, tr_std, va_std = curva_nb_cv5(
        X_train, y_train, cv, RUTA_CURVA_IMG, RUTA_CURVA_CSV
    )

    # Rendimiento promedio en TRAIN (CV-5)
    cv_model = make_nb()
    cv_acc = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"[CV-5] Accuracy (TRAIN) = {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    # Entrenamiento final y evaluación en TEST
    model = make_nb()
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
    print(f"Precision (macro)       = {prec_macro:.4f}")
    print(f"Recall (macro)          = {rec_macro:.4f}")
    print(f"F1-Score (macro)        = {f1m_test:.4f}")
    print(f"ROC-AUC (macro OVR)     = {roc_auc_macro:.4f}" if roc_auc_macro is not None else "ROC-AUC (macro OVR)     = N/A")

    resumen_nb = (
        f"Naive Bayes — Accuracy = {acc_test:.2f}, Precisión = {prec_macro:.2f}, "
        f"Sensibilidad = {rec_macro:.2f}, F1-Score = {f1m_test:.2f}, "
        f"ROC-AUC = {roc_auc_macro:.2f}" if roc_auc_macro is not None
        else f"Naive Bayes — Accuracy = {acc_test:.2f}, Precisión = {prec_macro:.2f}, "
             f"Sensibilidad = {rec_macro:.2f}, F1-Score = {f1m_test:.2f}, ROC-AUC = N/A"
    )
    print("\n--- RESUMEN NAIVE BAYES (TEST HOLD-OUT 20%) ---")
    print(resumen_nb)

    rep = classification_report(idx_to_name(y_test.values), idx_to_name(y_pred),
                                target_names=CLASSES, zero_division=0)
    print("\n=== REPORTE POR CLASE (TEST) ===")
    print(rep)

    cm = confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5])
    fig, ax = plt.subplots(figsize=(6.5,5))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES_FIG).plot(
        cmap="Blues", values_format="d", ax=ax, xticks_rotation=0
    )
    ax.set_title("Matriz de confusión: Naive Bayes", fontsize=14)
    ax.set_xlabel("Predicho", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
    plt.tight_layout()
    plt.savefig(RUTA_MATRIZ_CONFUSION, dpi=300)
    plt.close()
    print("[OK] Matriz de confusión guardada en:", RUTA_MATRIZ_CONFUSION)

    metricas = {
        "model": "NaiveBayes",
        "params": "GaussianNB (default)",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "cv5_train": {
            "accuracy_mean": float(cv_acc.mean()),
            "accuracy_std": float(cv_acc.std())
        },
        "curve": {
            "train_sizes": sizes_abs,
            "train_accuracy_mean": [float(x) for x in tr_mean],
            "valid_accuracy_mean": [float(x) for x in va_mean],
            "train_accuracy_std":  [float(x) for x in tr_std],
            "valid_accuracy_std":  [float(x) for x in va_std],
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
        "resumen": resumen_nb
    }

    with open(RUTA_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print("[OK] Métricas guardadas en:", RUTA_METRICAS)

    with open(RUTA_REPORTE_CLASIFICACION, "w", encoding="utf-8") as f:
        f.write(rep)
    print("[OK] Reporte de clasificación guardado en:", RUTA_REPORTE_CLASIFICACION)

    joblib.dump(model, RUTA_MODELO)
    with open(RUTA_PARAMS, "w", encoding="utf-8") as f:
        json.dump({"modelo": "GaussianNB", "features_used": FEATURES}, f, ensure_ascii=False, indent=2)
    print("[OK] Modelo guardado en:", RUTA_MODELO)
    print("[OK] Parámetros guardados en:", RUTA_PARAMS)

if __name__ == "__main__":
    main()
