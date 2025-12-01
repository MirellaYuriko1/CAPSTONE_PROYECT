from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from lightgbm import LGBMClassifier
import joblib

# =========================
# CONFIGURACIÓN GLOBAL
# =========================
SEED = 42
RUTA_DATOS = Path("data/final/phq9_final.csv")

DIR_OUT = Path("ml/LIGHTGBM/resultados")
DIR_OUT.mkdir(parents=True, exist_ok=True)

RUTA_CURVA_IMG             = DIR_OUT / "curva_entrenamiento_validacion_lgbm.png"
RUTA_CURVA_CSV             = DIR_OUT / "curva_entrenamiento_validacion_lgbm.csv"
RUTA_CURVA_PERDIDA_IMG     = DIR_OUT / "curva_perdida_lgbm.png"
RUTA_MATRIZ_CONFUSION      = DIR_OUT / "matriz_confusion_lgbm.png"
RUTA_CURVA_ROC_IMG         = DIR_OUT / "curva_roc_lgbm.png"
RUTA_METRICAS              = DIR_OUT / "metricas_lgbm.json"
RUTA_REPORTE_CLASIFICACION = DIR_OUT / "reporte_clasificacion_lgbm.txt"
RUTA_MODELO                = DIR_OUT / "modelo_lgbm.pkl"
RUTA_PARAMS                = DIR_OUT / "parametros_lgbm.json"
# --- NUEVA RUTA PARA EL CSV RESUMEN ---
RUTA_METRICAS_CSV = DIR_OUT / "metricas_modelo_lgbm.csv"
# =========================
# VARIABLES DEL MODELO
# =========================
FEATURES = [
    "age", "genero_bin",
    "phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"
]
TARGET   = "nivel_idx"

CLASSES = ["Mínimo", "Leve", "Moderada", "Moderadamente severa", "Severa"]
CLASSES_FIG = ["Mínimo", "Leve", "Moderada", "Moderadamente\nsevera", "Severa"]

def idx_to_name(arr_int):
    return [CLASSES[int(i) - 1] for i in arr_int]

# =========================
# HIPERPARÁMETROS LIGHTGBM (Configuración Anti-Overfitting)
# =========================
LGBM_PARAMS = dict(
    objective="multiclass",
    num_class=5,
    learning_rate=0.03,      
    n_estimators=250,        
    num_leaves=8,            
    max_depth=3,             
    min_child_samples=25,    
    min_split_gain=0.02,     
    subsample=0.6,           
    colsample_bytree=0.6,    
    reg_alpha=0.1,           
    reg_lambda=10.0,         
    random_state=SEED,
    n_jobs=-1,
    verbose=-1
)

def make_lgbm():
    # LightGBM maneja bien escalas, pero el StandardScaler no estorba en el Pipeline
    return Pipeline([
        ("scaler", StandardScaler()), 
        ("clf", LGBMClassifier(**LGBM_PARAMS))
    ])

# ==========================================================
# CURVA DE APRENDIZAJE (F1-MACRO)
# ==========================================================
def construir_modelo_LightGBM(X_train, y_train_enc, cv, ruta_png, ruta_csv):
    modelo = make_lgbm()
    train_sizes_rel = np.linspace(0.1, 1.0, 5)
    sizes_abs, train_scores, valid_scores = learning_curve(
        estimator=modelo,
        X=X_train,
        y=y_train_enc,
        train_sizes=train_sizes_rel,
        cv=cv,
        scoring="f1_macro", 
        n_jobs=-1,
        shuffle=True,
        random_state=SEED
    )
    tr_mean = train_scores.mean(axis=1); tr_std = train_scores.std(axis=1)
    va_mean = valid_scores.mean(axis=1); va_std = valid_scores.std(axis=1)
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

    # ---- Plot F1-Macro ----
    plt.figure(figsize=(7.8, 5.6))
    
    # Clipping visual
    tr_line = np.clip(tr_mean, 0.0, 1.0)
    va_line = np.clip(va_mean, 0.0, 1.0)

    plt.plot(sizes_abs, tr_line, marker="o", label="Entrenamiento", color="tab:blue")
    plt.fill_between(sizes_abs, np.clip(tr_mean - tr_std, 0, 1), np.clip(tr_mean + tr_std, 0, 1), alpha=0.15, color="tab:blue")

    plt.plot(sizes_abs, va_line, marker="s", label="Validación", color="tab:orange")
    plt.fill_between(sizes_abs, np.clip(va_mean - va_std, 0, 1), np.clip(va_mean + va_std, 0, 1), alpha=0.15, color="tab:orange")

    plt.ylim(0.0, 1.0)
    plt.title("Curva de Aprendizaje de LightGBM", fontsize=16, weight="bold")
    plt.xlabel("Tamaño del conjunto de entrenamiento (TRAIN)", fontsize=14)
    plt.ylabel("Macro F1-Score", fontsize=14) 
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)

    plt.tight_layout()
    plt.savefig(ruta_png, dpi=300)
    plt.close()
    print(f"[OK] Curva ENTRENAMIENTO/VALIDACIÓN (F1-Macro) guardada en: {ruta_png}")

    # ---- Plot de PÉRDIDA (Error 1-F1) ----
    plt.figure(figsize=(7.8, 5.6))
    plt.plot(sizes_abs, loss_train_mean, marker="o", label="Entrenamiento")
    plt.plot(sizes_abs, loss_valid_mean, marker="s", label="Validación")

    plt.ylim(0.0, 1.0)
    plt.title("Curva de Pérdida LightGBM", fontsize=16, weight="bold")
    plt.xlabel("Tamaño del conjunto de entrenamiento (TRAIN)", fontsize=14)
    plt.ylabel("Pérdida", fontsize=14) 
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
    y_raw = df[TARGET].astype(int).copy()  # 1..5
    assert not X.isna().any().any(), "Hay NaN en features; revisa el preprocesamiento."

    # LightGBM requiere clases 0..4 internamente
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    
    # Split 80/20
    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_enc,
        test_size=0.20,
        stratify=y_enc,
        random_state=SEED
    )
    # Recuperamos y_test_raw para métricas finales
    y_test_raw = le.inverse_transform(y_test_enc)
    
    print(f"[INFO] Entrenamiento = {len(X_train)} | Prueba = {len(X_test)}")

    # CV interno
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Curva usando F1-MACRO
    sizes_abs, tr_mean, va_mean, tr_std, va_std = construir_modelo_LightGBM(
        X_train, y_train_enc, cv, RUTA_CURVA_IMG, RUTA_CURVA_CSV
    )

    # CV promedio en TRAIN
    cv_model = make_lgbm()
    
    # <--- CAMBIO: cross_val_score ahora usa 'f1_macro'
    cv_metric = cross_val_score(
        cv_model, X_train, y_train_enc,
        cv=cv,
        scoring="f1_macro", 
        n_jobs=-1
    )
    print(f"[CV-5] F1-Score Macro (TRAIN) = {cv_metric.mean():.4f} ± {cv_metric.std():.4f}")

    # Entrenamiento final
    model = make_lgbm()
    model.fit(X_train, y_train_enc)

    y_pred_enc = model.predict(X_test)
    y_pred_raw = le.inverse_transform(y_pred_enc)
    proba_test = model.predict_proba(X_test)

    acc_test   = accuracy_score(y_test_raw, y_pred_raw)
    bacc_test  = balanced_accuracy_score(y_test_raw, y_pred_raw)
    f1m_test   = f1_score(y_test_raw, y_pred_raw, average="macro")
    prec_macro = precision_score(y_test_raw, y_pred_raw, average="macro", zero_division=0)
    rec_macro  = recall_score(y_test_raw, y_pred_raw, average="macro", zero_division=0)

    try:
        # ROC AUC necesita one-hot encoding de las etiquetas verdaderas
        roc_auc_macro = roc_auc_score(y_test_enc, proba_test, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc_macro = None

    print("\n=== MÉTRICAS EN TEST (GLOBAL / PROMEDIO ENTRE CLASES) ===")
    print(f"Accuracy               = {acc_test:.4f}")
    print(f"Balanced Accuracy       = {bacc_test:.4f}")
    print(f"Precisión (macro)       = {prec_macro:.4f}")
    print(f"Recall (macro)          = {rec_macro:.4f}")
    print(f"--> F1-Score (macro)    = {f1m_test:.4f}") # <--- Destacado
    print(f"ROC-AUC (macro OVR)     = {roc_auc_macro:.4f}" if roc_auc_macro is not None else "ROC-AUC (macro OVR)     = N/A")

    resumen_lgbm = (
        f"LightGBM — Accuracy = {acc_test:.2f}, F1-Macro = {f1m_test:.2f}, " # <--- F1 Primero
        f"Precisión = {prec_macro:.2f}, Sensibilidad = {rec_macro:.2f}, "
        f"ROC-AUC = {roc_auc_macro:.2f}" if roc_auc_macro is not None
        else f"LightGBM — Accuracy = {acc_test:.2f}, F1-Macro = {f1m_test:.2f}..."
    )
    print("\n--- RESUMEN LIGHTGBM (TEST HOLD-OUT 20%) ---")
    print(resumen_lgbm)

    rep = classification_report(
        idx_to_name(y_test_raw),
        idx_to_name(y_pred_raw),
        target_names=CLASSES,
        zero_division=0
    )
    print("\n=== REPORTE POR CLASE (TEST) ===")
    print(rep)

    cm = confusion_matrix(y_test_raw, y_pred_raw, labels=[1, 2, 3, 4, 5])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES_FIG).plot(
        cmap="Blues", values_format="d", ax=ax, xticks_rotation=0
    )
    ax.set_title("Matriz de confusión: LightGBM", fontsize=16, weight="bold")
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
    classes_int = [0, 1, 2, 3, 4] # Usamos los indices encoded 0..4
    n_classes = len(classes_int)
    y_test_bin = label_binarize(y_test_enc, classes=classes_int)

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
    plt.title('Curva ROC Multiclase — LightGBM', fontsize=16, weight='bold')
    plt.legend(loc="lower right", fontsize=13)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_CURVA_ROC_IMG, dpi=300)
    plt.close()
    print("[OK] Curva ROC guardada en:", RUTA_CURVA_ROC_IMG)

    loss_train_mean = [1.0 - float(x) for x in tr_mean]
    loss_valid_mean = [1.0 - float(x) for x in va_mean]

    # Guardar métricas
    metricas = {
        "model": "LightGBM",
        "params": LGBM_PARAMS,
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
        "labels_full": CLASSES,
        "label_encoder_map": {int(k): int(v) for k, v in enumerate(le.classes_)}
    }

    with open(RUTA_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print("[OK] Métricas guardadas en:", RUTA_METRICAS)

    with open(RUTA_REPORTE_CLASIFICACION, "w", encoding="utf-8") as f:
        f.write(rep)
    print("[OK] Reporte de clasificación guardado en:", RUTA_REPORTE_CLASIFICACION)

    joblib.dump(model, RUTA_MODELO)
    with open(RUTA_PARAMS, "w", encoding="utf-8") as f:
        json.dump({"params": LGBM_PARAMS, "features_used": FEATURES}, f, ensure_ascii=False, indent=2)
    print("[OK] Modelo guardado en:", RUTA_MODELO)
    print("[OK] Parámetros guardados en:", RUTA_PARAMS)
# ==========================================================
    # GENERACIÓN DEL CSV RESUMEN (Igual que Naive Bayes)
    # ==========================================================
    df_resumen = pd.DataFrame([{
        "Modelo": "LightGBM",
        "Accuracy": np.round(acc_test, 3),
        "BalancedAccuracy": np.round(bacc_test, 3),
        "Precision_macro": np.round(prec_macro, 3),
        "Recall_macro": np.round(rec_macro, 3),
        "F1_macro": np.round(f1m_test, 3),
        "ROC_AUC_macro": np.round(roc_auc_macro, 3) if roc_auc_macro is not None else None,
        "n": len(y_test_raw)
    }])
    
    df_resumen.to_csv(RUTA_METRICAS_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV de métricas resumen guardado en: {RUTA_METRICAS_CSV}")
# ==========================================================
    # GUARDAR DATOS PARA CURVA ROC COMPARATIVA (JSON)
    # ==========================================================
    roc_data = {
        "modelo": "LightGBM",
        "fpr": fpr_macro.tolist(),
        "tpr": tpr_macro.tolist(),
        "auc": roc_auc_macro
    }
    
    RUTA_ROC_JSON = DIR_OUT / "roc_data_lgbm.json"
    with open(RUTA_ROC_JSON, "w", encoding="utf-8") as f:
        json.dump(roc_data, f)
    print(f"[OK] Datos para curva ROC comparativa guardados en: {RUTA_ROC_JSON}")

if __name__ == "__main__":
    main()