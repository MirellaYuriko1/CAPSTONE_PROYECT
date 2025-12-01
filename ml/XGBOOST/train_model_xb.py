from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, cross_val_score
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Model imports
import joblib
from xgboost import XGBClassifier

# =========================
# CONFIGURACIÓN GLOBAL
# =========================
SEED = 42
RUTA_DATOS = Path("data/final/phq9_final.csv")

DIR_OUT = Path("ml/XGBOOST/resultados")
DIR_OUT.mkdir(parents=True, exist_ok=True)

# Rutas de salida para imágenes y archivos
RUTA_CURVA_IMG             = DIR_OUT / "curva_entrenamiento_validacion_xgb.png"
RUTA_CURVA_CSV             = DIR_OUT / "curva_entrenamiento_validacion_xgb.csv"
RUTA_CURVA_PERDIDA_IMG     = DIR_OUT / "curva_perdida_xgb.png"
RUTA_MATRIZ_CONFUSION      = DIR_OUT / "matriz_confusion_xgb.png"
RUTA_METRICAS              = DIR_OUT / "metricas_xgb.json"
RUTA_REPORTE_CLASIFICACION = DIR_OUT / "reporte_clasificacion_xgb.txt"
RUTA_MODELO                = DIR_OUT / "modelo_xgb.pkl"
RUTA_PARAMS                = DIR_OUT / "parametros_xgb.json"
RUTA_ROC_IMG               = DIR_OUT / "curva_roc_multiclase_xgb.png"

# --- NUEVA RUTA PARA EL CSV RESUMEN ---
RUTA_METRICAS_CSV          = DIR_OUT / "metricas_modelo_xgb.csv"

# Variables de entrada/salida
FEATURES = [
    "age", "genero_bin",
    "phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"
]
TARGET   = "nivel_idx"

# Etiquetas para gráficas y reportes
CLASSES = ["Mínimo","Leve","Moderada","Moderadamente severa","Severa"]
CLASSES_FIG = ["Mínimo","Leve","Moderada","Moderadamente\nsevera","Severa"]

def idx_to_name(arr_int):
    """Convierte índices 1..5 a nombres de clases para reportes."""
    return [CLASSES[int(i) - 1] for i in arr_int]

# =========================
# HIPERPARÁMETROS XGB (Anti-Overfitting)
# =========================
XGB_PARAMS = dict(
    objective="multi:softprob",   
    num_class=5,
    eval_metric="mlogloss",
    n_estimators=150,       
    learning_rate=0.05,     
    max_depth=3,            
    min_child_weight=6,     
    subsample=0.7,          
    colsample_bytree=0.7,   
    gamma=1.0,              
    reg_alpha=0.5,          
    reg_lambda=3.0,         
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1
)

# ----------------------------------------------------------
# Función: Curva de aprendizaje (F1-MACRO y Pérdida)
# ----------------------------------------------------------
def construir_modelo_xgb(X_train, y_train_0based, cv, ruta_png, ruta_csv):
    modelo = XGBClassifier(**XGB_PARAMS)
    train_sizes_rel = np.linspace(0.1, 1.0, 5)
    print("[INFO] Generando curva de aprendizaje...")
    sizes_abs, train_scores, valid_scores = learning_curve(
        estimator=modelo,
        X=X_train,
        y=y_train_0based,
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
        "loss_train_mean":    np.round(loss_train_mean, 4),
        "loss_valid_mean":    np.round(loss_valid_mean, 4),
    })
    df_curve.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV de curva guardado en: {ruta_csv}")
    
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
    plt.title("Curva de Aprendizaje de XGBoost", fontsize=16, weight="bold")
    plt.xlabel("Tamaño del conjunto de entrenamiento (TRAIN)", fontsize=14)
    plt.ylabel("F1-Score Macro", fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=300)
    plt.close()
    print("[OK] Curva ENTRENAMIENTO/VALIDACIÓN guardada en:", ruta_png)

    # ---- FIGURA 2: Pérdida ----
    plt.figure(figsize=(7.8, 5.6))
    plt.plot(sizes_abs, loss_train_mean, marker="o", label="Pérdida entrenamiento")
    plt.plot(sizes_abs, loss_valid_mean, marker="s", label="Pérdida validación")

    plt.ylim(0.0, 1.0)
    plt.title("Curva de Pérdida de XGBoost", fontsize=16, weight="bold")
    plt.xlabel("Tamaño del conjunto de entrenamiento (TRAIN)", fontsize=14)
    plt.ylabel("Pérdida", fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(RUTA_CURVA_PERDIDA_IMG, dpi=300)
    plt.close()
    print("[OK] Curva de PÉRDIDA guardada en:", RUTA_CURVA_PERDIDA_IMG)

    return sizes_abs.tolist(), tr_mean.tolist(), va_mean.tolist(), tr_std.tolist(), va_std.tolist()

# =========================
# PROCESO PRINCIPAL
# =========================
def main():
    if not RUTA_DATOS.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {RUTA_DATOS}")

    df = pd.read_csv(RUTA_DATOS)
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()  # 1..5

    assert not X.isna().any().any(), "Hay NaN en features; revisa el preprocesamiento."

    # Split 80/20 estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"[INFO] Entrenamiento = {len(X_train)} | Prueba = {len(X_test)}")

    # XGBoost necesita clases 0..(K-1) internamente
    y_train_0 = (y_train - 1).values
    y_test_0  = (y_test  - 1).values

    # CV estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1. Curva de aprendizaje
    sizes_abs, tr_mean, va_mean, tr_std, va_std = construir_modelo_xgb(
        X_train, y_train_0, cv, RUTA_CURVA_IMG, RUTA_CURVA_CSV
    )

    # 2. Validación cruzada en Train (F1-Macro)
    modelo_cv = XGBClassifier(**XGB_PARAMS)
    cv_f1 = cross_val_score(
        modelo_cv, X_train, y_train_0,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )
    print(f"[CV-5] F1-Score Macro (TRAIN) = {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # 3. Entrenamiento final
    print("[INFO] Entrenando modelo final...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train_0)

    # ===== EVALUACIÓN EN TEST =====
    y_pred_0 = model.predict(X_test)         # 0..4
    y_pred   = (y_pred_0 + 1)                # 1..5 para reportes humanos
    proba_test = model.predict_proba(X_test) # Probabilidades para ROC

    acc_test   = accuracy_score(y_test, y_pred)
    bacc_test  = balanced_accuracy_score(y_test, y_pred)
    f1m_test   = f1_score(y_test, y_pred, average="macro")
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro  = recall_score(y_test, y_pred, average="macro", zero_division=0)

    try:
        roc_auc_macro = roc_auc_score(y_test_0, proba_test, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc_macro = None

    print("\n=== MÉTRICAS EN TEST (GLOBAL / PROMEDIO ENTRE CLASES) ===")
    print(f"Accuracy            = {acc_test:.4f}")
    print(f"Balanced Accuracy   = {bacc_test:.4f}")
    print(f"Precisión (macro)   = {prec_macro:.4f}")
    print(f"Recall (macro)      = {rec_macro:.4f}")
    print(f"F1-Score (macro)    = {f1m_test:.4f}")
    if roc_auc_macro is not None:
        print(f"ROC-AUC (macro OVR) = {roc_auc_macro:.4f}")

    resumen = (
        f"XGBoost — Accuracy = {acc_test:.2f}, Precisión = {prec_macro:.2f}, "
        f"Sensibilidad = {rec_macro:.2f}, F1-Score = {f1m_test:.2f}, "
        f"ROC-AUC = {roc_auc_macro:.2f}" if roc_auc_macro is not None
        else "ROC-AUC = N/A"
    )
    print("\n--- RESUMEN XGBOOST (TEST HOLD-OUT 20%) ---")
    print(resumen)

    # 4. Reporte de Clasificación Texto
    rep = classification_report(
        idx_to_name(y_test.values),
        idx_to_name(y_pred),
        target_names=CLASSES,
        zero_division=0
    )
    print("\n=== REPORTE POR CLASE (TEST) ===")
    print(rep)

    # 5. Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5])
    fig, ax = plt.subplots(figsize=(6.5,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES_FIG)
    disp.plot(cmap="Blues", values_format="d", ax=ax, xticks_rotation=0)
    
    ax.set_title("Matriz de confusión: XGBoost", fontsize=16, weight="bold")
    ax.set_xlabel("Predicho", fontsize=14)
    ax.set_ylabel("Real", fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=13)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=13)
    plt.tight_layout()
    plt.savefig(RUTA_MATRIZ_CONFUSION, dpi=300)
    plt.close()
    print("[OK] Matriz de confusión guardada en:", RUTA_MATRIZ_CONFUSION)

    # ==========================================================
    # 6. CURVA ROC MULTICLASE (MICRO Y MACRO PROMEDIO)
    # ==========================================================
    print("\n=== GENERANDO CURVA ROC MULTICLASE ===")
    
    # Binarizar las etiquetas verdaderas para cálculo ROC (necesario para one-vs-rest)
    n_classes = 5
    y_test_bin = label_binarize(y_test_0, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc_dict = dict()

    # A) Calcular ROC para cada clase individual
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], proba_test[:, i])
        roc_auc_dict[i] = auc(fpr[i], tpr[i])

    # B) Calcular MICRO-promedio
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), proba_test.ravel())
    roc_auc_dict["micro"] = auc(fpr["micro"], tpr["micro"])

    # C) Calcular MACRO-promedio
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr["macro"], tpr["macro"])

    # D) Graficar
    plt.figure(figsize=(8, 6.5))
    plt.plot(
        fpr["micro"], tpr["micro"],
        label=f'Micro-promedio (AUC = {roc_auc_dict["micro"]:.2f})',
        color='darkorange', linestyle='-', linewidth=2.5
    )
    plt.plot(
        fpr["macro"], tpr["macro"],
        label=f'Macro-promedio (AUC = {roc_auc_dict["macro"]:.2f})',
        color='navy', linestyle='-', linewidth=2.5
    )
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Clasificador aleatorio')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=14)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=14)
    plt.title('Curva ROC Multiclase — XGBoost', fontsize=16, weight='bold')
    plt.legend(loc="lower right", fontsize=13)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_ROC_IMG, dpi=300)
    plt.close()
    print(f"[OK] Curva ROC Multiclase guardada en: {RUTA_ROC_IMG}")

    # 8. Guardar Métricas en JSON
    loss_train_mean = [1.0 - float(x) for x in tr_mean]
    loss_valid_mean = [1.0 - float(x) for x in va_mean]

    metricas = {
        "model": "XGBoost",
        "params": XGB_PARAMS,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "cv5_train": {
            "f1_macro_mean": float(cv_f1.mean()), 
            "f1_macro_std":  float(cv_f1.std())   
        },
        "curve": {
            "train_sizes":         sizes_abs,
            "train_f1_macro_mean": [float(x) for x in tr_mean],
            "train_f1_macro_std":  [float(x) for x in tr_std],
            "valid_f1_macro_mean": [float(x) for x in va_mean],
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
            "roc_auc_macro":     (float(roc_auc_macro) if roc_auc_macro is not None else None),
            "roc_auc_micro":     float(roc_auc_dict["micro"])
        },
        "confusion_matrix": cm.tolist(),
        "labels_plot": CLASSES_FIG,
        "labels_full": CLASSES,
        "resumen": resumen
    }

    with open(RUTA_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print("[OK] Métricas JSON guardadas en:", RUTA_METRICAS)

    with open(RUTA_REPORTE_CLASIFICACION, "w", encoding="utf-8") as f:
        f.write(rep)
    print("[OK] Reporte de clasificación TXT guardado en:", RUTA_REPORTE_CLASIFICACION)

    # 9. Guardar Modelo y Parámetros
    joblib.dump(model, RUTA_MODELO)
    with open(RUTA_PARAMS, "w", encoding="utf-8") as f:
        json.dump({"params": XGB_PARAMS, "features_used": FEATURES}, f, ensure_ascii=False, indent=2)
    print("[OK] Modelo guardado en:", RUTA_MODELO)
    print("[OK] Parámetros guardados en:", RUTA_PARAMS)

    # ==========================================================
    # GENERACIÓN DEL CSV RESUMEN (Igual que Naive Bayes)
    # ==========================================================
    df_resumen = pd.DataFrame([{
        "Modelo": "XGBoost",
        "Accuracy": np.round(acc_test, 3),
        "BalancedAccuracy": np.round(bacc_test, 3),
        "Precision_macro": np.round(prec_macro, 3),
        "Recall_macro": np.round(rec_macro, 3),
        "F1_macro": np.round(f1m_test, 3),
        "ROC_AUC_macro": np.round(roc_auc_macro, 3) if roc_auc_macro is not None else None,
        "n": len(y_test)
    }])
    
    df_resumen.to_csv(RUTA_METRICAS_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV de métricas resumen guardado en: {RUTA_METRICAS_CSV}")

    # ==========================================================
    # GUARDAR DATOS PARA CURVA ROC COMPARATIVA (JSON)
    # ==========================================================
    roc_data = {
        "modelo": "XGBoost",
        "fpr": fpr["macro"].tolist(), # Usamos la curva MACRO para la comparativa
        "tpr": tpr["macro"].tolist(),
        "auc": roc_auc_macro
    }
    
    RUTA_ROC_JSON = DIR_OUT / "roc_data_xgb.json"
    with open(RUTA_ROC_JSON, "w", encoding="utf-8") as f:
        json.dump(roc_data, f)
    print(f"[OK] Datos para curva ROC comparativa guardados en: {RUTA_ROC_JSON}")

    print("\n--- EJECUCIÓN FINALIZADA CON ÉXITO ---")

if __name__ == "__main__":
    main()