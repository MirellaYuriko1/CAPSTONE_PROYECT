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
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance # <--- NECESARIO PARA IMPORTANCIA EN NB
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
RUTA_CURVA_PERDIDA_IMG     = DIR_OUT / "curva_perdida_nb.png"
RUTA_MATRIZ_CONFUSION      = DIR_OUT / "matriz_confusion_nb.png"
RUTA_CURVA_ROC_IMG         = DIR_OUT / "curva_roc_nb.png"
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
TARGET   = "nivel_idx"

CLASSES = ["Mínimo","Leve","Moderada","Moderadamente severa","Severa"]
CLASSES_FIG = ["Mínimo","Leve","Moderada","Moderadamente\nsevera","Severa"]

def idx_to_name(arr_int):
    return [CLASSES[int(i) - 1] for i in arr_int]

# =========================
# PARÁMETROS DEL MODELO NB (Estrategia Robusta)
# =========================
# var_smoothing: Es el "freno" de Naive Bayes.
# Default (1e-9) es muy sensible. 
# Usamos 0.05 para suavizar la curva gaussiana y evitar overfitting.
NB_PARAMS = dict(
    var_smoothing=0.05 
)  

def make_nb():
    return GaussianNB(**NB_PARAMS)

# ==========================================================
# FUNCIÓN: CURVA DE APRENDIZAJE (F1-MACRO)
# ==========================================================
def curva_nb_cv5(X_train, y_train, cv, ruta_png, ruta_csv):
    modelo = make_nb()
    train_sizes_rel = np.linspace(0.1, 1.0, 5)
    sizes_abs, train_scores, valid_scores = learning_curve(
        estimator=modelo,
        X=X_train,
        y=y_train,
        train_sizes=train_sizes_rel,
        cv=cv,
        scoring="f1_macro", 
        n_jobs=-1,
        shuffle=True,
        random_state=SEED
    )
    tr_mean = train_scores.mean(axis=1)
    tr_std  = train_scores.std(axis=1)
    va_mean = valid_scores.mean(axis=1)
    va_std  = valid_scores.std(axis=1) 
    loss_train_mean = 1.0 - tr_mean
    loss_valid_mean = 1.0 - va_mean
    df_curve = pd.DataFrame({
        "train_size":        sizes_abs,
        "f1_macro_train_cv_mean": np.round(tr_mean, 4), 
        "f1_macro_train_cv_std":  np.round(tr_std, 4),  
        "f1_macro_valid_cv_mean": np.round(va_mean, 4), 
        "f1_macro_valid_cv_std":  np.round(va_std, 4),  
        "loss_train_mean (1-F1)":    np.round(loss_train_mean, 4),
        "loss_valid_mean (1-F1)":    np.round(loss_valid_mean, 4),
    })
    df_curve.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV de curva guardado en: {ruta_csv}")

    # ----- figura de F1-Macro -----
    plt.figure(figsize=(7.8, 5.6))
    
    # Clipping visual
    tr_line = np.clip(tr_mean, 0.0, 1.0)
    va_line = np.clip(va_mean, 0.0, 1.0)

    plt.plot(sizes_abs, tr_line, marker="o", label="Entrenamiento", color="tab:blue")
    plt.fill_between(sizes_abs, np.clip(tr_mean - tr_std, 0, 1), np.clip(tr_mean + tr_std, 0, 1), alpha=0.15, color="tab:blue")
    
    plt.plot(sizes_abs, va_line, marker="s", label="Validación", color="tab:orange")
    plt.fill_between(sizes_abs, np.clip(va_mean - va_std, 0, 1), np.clip(va_mean + va_std, 0, 1), alpha=0.15, color="tab:orange")

    plt.ylim(0.0, 1.0)
    plt.title("Curva de Aprendizaje de Naive Bayes", fontsize=16, weight="bold")
    plt.xlabel("Tamaño del conjunto de entrenamiento (TRAIN)", fontsize=14)
    plt.ylabel("Macro F1-Score", fontsize=14) # <--- ETIQUETA ACTUALIZADA
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=300)
    plt.close()
    print(f"[OK] Curva ENTRENAMIENTO/VALIDACIÓN (F1-Macro) guardada en: {ruta_png}")

    # ----- figura de PÉRDIDA (Error 1-F1) -----
    plt.figure(figsize=(7.8, 5.6))
    plt.plot(sizes_abs, loss_train_mean, marker="o", label="Entrenamiento")
    plt.plot(sizes_abs, loss_valid_mean, marker="s", label="Validación")

    plt.ylim(0.0, 1.0)
    plt.title("Curva de Pérdida Naive Bayes", fontsize=16, weight="bold")
    plt.xlabel("Tamaño del conjunto de entrenamiento (TRAIN)", fontsize=14)
    plt.ylabel("Pérdida", fontsize=14) # <--- ETIQUETA ACTUALIZADA
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(RUTA_CURVA_PERDIDA_IMG, dpi=300)
    plt.close()
    print(f"[OK] Curva de ERROR (1-F1) guardada en: {RUTA_CURVA_PERDIDA_IMG}")

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

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"[INFO] Entrenamiento = {len(X_train)} | Prueba = {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Curva CV-5 (F1-MACRO)
    sizes_abs, tr_mean, va_mean, tr_std, va_std = curva_nb_cv5(
        X_train, y_train, cv, RUTA_CURVA_IMG, RUTA_CURVA_CSV
    )

    # CV promedio en TRAIN
    cv_model = make_nb()
    
    # <--- CAMBIO: cross_val_score ahora usa 'f1_macro'
    cv_metric = cross_val_score(
        cv_model, X_train, y_train,
        cv=cv,
        scoring="f1_macro", 
        n_jobs=-1
    )
    print(f"[CV-5] F1-Score Macro (TRAIN) = {cv_metric.mean():.4f} ± {cv_metric.std():.4f}")

    # Entrenamiento final
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
    print(f"Accuracy               = {acc_test:.4f}")
    print(f"Balanced Accuracy       = {bacc_test:.4f}")
    print(f"Precisión (macro)       = {prec_macro:.4f}")
    print(f"Recall (macro)          = {rec_macro:.4f}")
    print(f"--> F1-Score (macro)    = {f1m_test:.4f}") # <--- Destacado
    print(f"ROC-AUC (macro OVR)     = {roc_auc_macro:.4f}" if roc_auc_macro is not None else "ROC-AUC (macro OVR)     = N/A")

    resumen_nb = (
        f"Naive Bayes — Accuracy = {acc_test:.2f}, F1-Macro = {f1m_test:.2f}, " # <--- F1 Primero
        f"Precisión = {prec_macro:.2f}, Sensibilidad = {rec_macro:.2f}, "
        f"ROC-AUC = {roc_auc_macro:.2f}" if roc_auc_macro is not None
        else f"Naive Bayes — Accuracy = {acc_test:.2f}, F1-Macro = {f1m_test:.2f}..."
    )
    print("\n--- RESUMEN NAIVE BAYES (TEST HOLD-OUT 20%) ---")
    print(resumen_nb)

    rep = classification_report(
        idx_to_name(y_test.values),
        idx_to_name(y_pred),
        target_names=CLASSES,
        zero_division=0
    )
    print("\n=== REPORTE POR CLASE (TEST) ===")
    print(rep)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES_FIG).plot(
        cmap="Blues", values_format="d", ax=ax, xticks_rotation=0
    )
    ax.set_title("Matriz de confusión: Naive Bayes", fontsize=16, weight="bold")
    ax.set_xlabel("Predicho", fontsize=14)
    ax.set_ylabel("Real", fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=13)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=13)
    plt.tight_layout()
    plt.savefig(RUTA_MATRIZ_CONFUSION, dpi=300)
    plt.close()
    print("[OK] Matriz de confusión guardada en:", RUTA_MATRIZ_CONFUSION)

    # --------------------------------------------------
    # Curva ROC Multiclase (Micro + Macro + Fix 0,0)
    # --------------------------------------------------
    classes_int = [1, 2, 3, 4, 5]
    n_classes = len(classes_int)
    y_test_bin = label_binarize(y_test, classes=classes_int)

    # 1. Micro
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), proba_test.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # 2. Macro
    fpr_dict = dict()
    tpr_dict = dict()
    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_bin[:, i], proba_test[:, i])

    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    roc_auc_macro_curve = auc(fpr_macro, tpr_macro)

    # FIX VISUAL (0,0)
    if not (fpr_micro[0] == 0 and tpr_micro[0] == 0):
        fpr_micro = np.insert(fpr_micro, 0, 0.0)
        tpr_micro = np.insert(tpr_micro, 0, 0.0)
    if not (fpr_macro[0] == 0 and tpr_macro[0] == 0):
        fpr_macro = np.insert(fpr_macro, 0, 0.0)
        tpr_macro = np.insert(tpr_macro, 0, 0.0)

    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr_micro, tpr_micro,
             label=f'Micro-promedio (AUC = {roc_auc_micro:.2f})',
             color='darkorange', linestyle='-', linewidth=2)
    plt.plot(fpr_macro, tpr_macro,
             label=f'Macro-promedio (AUC = {roc_auc_macro_curve:.2f})',
             color='navy', linestyle='-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Clasificador aleatorio')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=14)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=14)
    plt.title('Curva ROC Multiclase — Naive Bayes', fontsize=16, weight='bold')
    plt.legend(loc="lower right", fontsize=13)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_CURVA_ROC_IMG, dpi=300)
    plt.close()
    print("[OK] Curva ROC guardada en:", RUTA_CURVA_ROC_IMG)

    # Guardar métricas
    loss_train_mean = [1.0 - float(x) for x in tr_mean]
    loss_valid_mean = [1.0 - float(x) for x in va_mean]

    metricas = {
        "model": "NaiveBayes",
        "params": NB_PARAMS,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "cv5_train": {
            "f1_macro_mean": float(cv_metric.mean()), # <-- ETIQUETA CAMBIADA
            "f1_macro_std":  float(cv_metric.std())   # <-- ETIQUETA CAMBIADA
        },
        "curve": {
            "train_sizes":         sizes_abs,
            "train_f1_macro_mean": [float(x) for x in tr_mean], # <-- ETIQUETA CAMBIADA
            "train_f1_macro_std":  [float(x) for x in tr_std],
            "valid_f1_macro_mean": [float(x) for x in va_mean], # <-- ETIQUETA CAMBIADA
            "valid_f1_macro_std":  [float(x) for x in va_std],
            "train_loss_mean":     loss_train_mean,
            "valid_loss_mean":     loss_valid_mean,
        },
        "test_metrics": {
            "accuracy":          float(acc_test),
            "balanced_accuracy": float(bacc_test),
            "precision_macro":   float(prec_macro),
            "recall_macro":      float(rec_macro),
            "f1_macro":          float(f1m_test),
            "roc_auc_macro":     (float(roc_auc_macro) if roc_auc_macro is not None else None)
        },
        "confusion_matrix": cm.tolist(),
        "labels_plot": CLASSES_FIG,
        "labels_full": CLASSES
    }

    with open(RUTA_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print("[OK] Métricas guardadas en:", RUTA_METRICAS)

    with open(RUTA_REPORTE_CLASIFICACION, "w", encoding="utf-8") as f:
        f.write(rep)
    print("[OK] Reporte de clasificación guardado en:", RUTA_REPORTE_CLASIFICACION)

    joblib.dump(model, RUTA_MODELO)
    with open(RUTA_PARAMS, "w", encoding="utf-8") as f:
        json.dump({"params": NB_PARAMS, "features_used": FEATURES}, f, ensure_ascii=False, indent=2)
    print("[OK] Modelo guardado en:", RUTA_MODELO)
    print("[OK] Parámetros guardados en:", RUTA_PARAMS)

# ==========================================================
    # GUARDAR DATOS PARA CURVA ROC COMPARATIVA (JSON)
    # ==========================================================
    roc_data = {
        "modelo": "NaiveBayes",
        "fpr": fpr_macro.tolist(),
        "tpr": tpr_macro.tolist(),
        "auc": roc_auc_macro
    }
    
    RUTA_ROC_JSON = DIR_OUT / "roc_data_nb.json"
    with open(RUTA_ROC_JSON, "w", encoding="utf-8") as f:
        json.dump(roc_data, f)
    print(f"[OK] Datos para curva ROC comparativa guardados en: {RUTA_ROC_JSON}")

if __name__ == "__main__":
    main()