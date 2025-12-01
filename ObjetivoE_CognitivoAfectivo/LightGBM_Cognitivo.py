from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
)
from lightgbm import LGBMClassifier

# =========================
# CONFIGURACIÓN GENERAL
# =========================
SEED = 42
RUTA_DATOS = Path("data/final/phq9_final.csv")

# Carpeta específica para resultados de LightGBM en el objetivo cognitivo
DIR_OUT = Path("ObjetivoE_CognitivoAfectivo/resultados_LGBM")
DIR_OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# DIMENSIÓN COGNITIVO-AFECTIVA
# ---------------------------------------------------------------------
# Usamos las preguntas mentales/emocionales (NO las somáticas 3, 4, 5)
ITEMS = ["phq1", "phq2", "phq6", "phq7", "phq8", "phq9"]

ITEMS_LABELS = {
    "phq1": "Estado de ánimo (Tristeza)",
    "phq2": "Pérdida de interés (Anhedonia)",
    "phq6": "Sentimiento de Fracaso",
    "phq7": "Dificultad para concentrarse",
    "phq8": "Intranquilidad o Lentitud",
    "phq9": "Pensamientos de muerte/autolesión",
}

# Escala Likert del PHQ-9 (0–3)
LIKERT_VALUES = [0, 1, 2, 3]
LIKERT_LABELS = [
    "Nada",
    "Varios días",
    "Más de la mitad de los días",
    "Casi todos los días",
]

# Todas las columnas del PHQ para usar de contexto
PHQ_COLS = [f"phq{i}" for i in range(1, 10)]

# =========================
# HIPERPARÁMETROS LIGHTGBM (Idénticos al General)
# =========================
LGBM_PARAMS = dict(
    objective="multiclass",
    num_class=4,             # 0, 1, 2, 3
    learning_rate=0.05,      
    n_estimators=200,       
    num_leaves=10,           
    max_depth=3,             
    min_child_samples=30,   
    min_split_gain=0.01,     
    subsample=0.7,           
    colsample_bytree=0.7,    
    reg_alpha=0.5,           
    reg_lambda=5.0,          
    random_state=SEED,
    n_jobs=-1,
    verbose=-1
)

def make_lgbm():
    # Usamos Pipeline con Scaler tal como en el modelo general
    return Pipeline([
        ("scaler", StandardScaler()), 
        ("clf", LGBMClassifier(**LGBM_PARAMS))
    ])

# =========================
# PROCESO PRINCIPAL
# =========================
def main():
    if not RUTA_DATOS.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {RUTA_DATOS}")

    df = pd.read_csv(RUTA_DATOS)

    # Aseguramos tipo int
    for col in ["age", "genero_bin"] + PHQ_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    n_estudiantes = len(df)
    print(f"[INFO] Número de estudiantes = {n_estudiantes}")
    print(f"[INFO] Analizando Dimensión Cognitivo-Afectiva con LightGBM ({len(ITEMS)} ítems).")

    # Estructuras para guardar resultados temporales
    preds_por_item = {}

    # ---------------------------------------------------------
    # 1. ENTRENAMIENTO POR ÍTEM (Individual)
    # ---------------------------------------------------------
    for item in ITEMS:
        print(f"... Procesando síntoma: {item} ({ITEMS_LABELS[item]})")
        
        # DEFINICIÓN DE FEATURES: Contexto (Todo MENOS el ítem actual)
        feature_cols = ["age", "genero_bin"] + [c for c in PHQ_COLS if c != item]
        
        X = df[feature_cols].copy()
        y = df[item].astype(int).copy()

        # Usamos el Pipeline definido (Scaler + LGBM)
        clf = make_lgbm()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        # Cross Validation para obtener predicciones limpias
        y_pred = cross_val_predict(clf, X, y, cv=skf, method="predict")
        # Probabilidades para AUC
        proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")

        preds_por_item[item] = {
            "y_pred": y_pred,
            "proba": proba,
        }

    # ---------------------------------------------------------
    # 2. CONSTRUCCIÓN DE LA MÉTRICA DE DIMENSIÓN
    # ---------------------------------------------------------
    # Nivel Real = Máximo valor registrado entre los síntomas cognitivos
    y_oe_true = df[ITEMS].max(axis=1).astype(int).values

    # Nivel Predicho = Máximo valor predicho por los modelos
    preds_matrix = np.column_stack([preds_por_item[item]["y_pred"] for item in ITEMS])
    y_oe_pred = preds_matrix.max(axis=1)

    # Probabilidad promedio (aproximación para AUC)
    proba_stack = np.stack([preds_por_item[item]["proba"] for item in ITEMS], axis=0)
    proba_oe = proba_stack.mean(axis=0)

    # ---------------------------------------------------------
    # 3. GENERACIÓN DE ENTREGABLES (Matriz y Métricas)
    # ---------------------------------------------------------

    # --- A) Matriz de Confusión ---
    cm = confusion_matrix(y_oe_true, y_oe_pred, labels=LIKERT_VALUES)
    
    xtick_labels = ["Nada", "Varios\ndías", "Más de la\nmitad", "Casi todos\nlos días"]
    ytick_labels = ["Nada", "Varios días", "Más de la mitad", "Casi todos los días"]

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5), dpi=160)
    im = ax_cm.imshow(cm, cmap="Blues", aspect="auto")

    ax_cm.set_title("Matriz de Confusión: Dimensión Cognitivo-Afectiva (LightGBM)", fontsize=12, weight='bold')
    ax_cm.set_xlabel("Predicción del Nivel Cognitivo", fontsize=10)
    ax_cm.set_ylabel("Nivel Cognitivo Real (Max)", fontsize=10)

    ax_cm.set_xticks(range(len(xtick_labels)))
    ax_cm.set_xticklabels(xtick_labels, rotation=0, fontsize=9)
    ax_cm.set_yticks(range(len(ytick_labels)))
    ax_cm.set_yticklabels(ytick_labels, fontsize=9)

    # Poner números en las celdas
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, int(cm[i, j]), ha="center", va="center", 
                     color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.colorbar(im, ax=ax_cm)
    plt.tight_layout()
    ruta_cm = DIR_OUT / "matriz_confusion_cognitivo_lgbm.png"
    plt.savefig(ruta_cm, dpi=300)
    plt.close()
    print(f"[OK] Matriz de confusión guardada en: {ruta_cm}")

    # --- B) Métricas y CSV ---
    acc = accuracy_score(y_oe_true, y_oe_pred)
    
    # Reporte completo para extraer medias macro
    rep = classification_report(
        y_oe_true, y_oe_pred, labels=LIKERT_VALUES, target_names=LIKERT_LABELS, output_dict=True, zero_division=0
    )
    
    try:
        auc_macro = roc_auc_score(y_oe_true, proba_oe, multi_class="ovr", average="macro")
    except ValueError:
        auc_macro = None

    # Crear DataFrame resumen para la tesis
    df_resumen = pd.DataFrame([{
        "Modelo": "LightGBM (Dimensión Cognitivo-Afectiva)",
        "Accuracy": round(acc, 3),
        "Precision_macro": round(rep["macro avg"]["precision"], 3),
        "Recall_macro": round(rep["macro avg"]["recall"], 3),
        "F1_macro": round(rep["macro avg"]["f1-score"], 3),
        "ROC-AUC": round(auc_macro, 3) if auc_macro else "N/A"
    }])

    ruta_csv = DIR_OUT / "metricas_cognitivo_lgbm_tesis.csv"
    df_resumen.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV resumen guardado en: {ruta_csv}")

    print("\n=== RESULTADOS FINALES LIGHTGBM (COGNITIVO-AFECTIVO) ===")
    print(df_resumen.to_string(index=False))

if __name__ == "__main__":
    main()