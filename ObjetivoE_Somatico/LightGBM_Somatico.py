# LGBM_O2/LGBM_O2.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

# =========================
# CONFIGURACIÓN GENERAL
# =========================
SEED = 42
RUTA_DATOS = Path("data/final/phq9_final.csv")

# Carpeta para resultados de LightGBM en el objetivo somático
DIR_OUT = Path("ObjetivoE_Somatico/resultados_LGBM")
DIR_OUT.mkdir(parents=True, exist_ok=True)

# Síntomas que quieres analizar en este OE2-Somático (PHQ-9)
ITEMS = ["phq3", "phq4", "phq5"]
ITEMS_LABELS = {
    "phq3": "Problemas de sueño",
    "phq4": "Cansancio",
    "phq5": "Cambios en el apetito",
}

# Escala Likert del PHQ-9 (0–3)
LIKERT_VALUES = [0, 1, 2, 3]
LIKERT_LABELS = [
    "Nada",
    "Varios días",
    "Más de la mitad de los días",
    "Casi todos los días",
]

PHQ_COLS = [f"phq{i}" for i in range(1, 10)]

# Hiperparámetros LightGBM (basados en tu modelo general)
LGBM_PARAMS = dict(
    objective="multiclass",
    num_class=4,              # 4 clases: 0,1,2,3
    learning_rate=0.05,
    n_estimators=200,
    num_leaves=15,
    max_depth=4,
    min_child_samples=20,
    min_split_gain=0.01,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1
)


def make_lgbm():
    return LGBMClassifier(**LGBM_PARAMS)


# =========================
# FUNCIONES AUXILIARES
# =========================
def _brier_multiclass(y_true, proba, classes):
    """Brier score multicategoría."""
    y_true = np.array(y_true)
    proba = np.array(proba)
    Y = np.zeros_like(proba)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for i, y in enumerate(y_true):
        Y[i, class_to_idx[y]] = 1.0
    return float(np.mean(np.sum((proba - Y) ** 2, axis=1)))


def ovr_counts_and_metrics(y_true, y_pred, labels):
    filas = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for lbl, nombre in zip(labels, LIKERT_LABELS):
        tp = int(((y_true == lbl) & (y_pred == lbl)).sum())
        fp = int(((y_true != lbl) & (y_pred == lbl)).sum())
        fn = int(((y_true == lbl) & (y_pred != lbl)).sum())
        tn = int(((y_true != lbl) & (y_pred != lbl)).sum())

        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        filas.append(
            {
                "Nivel": nombre,
                "f00": tn,
                "f01": fp,
                "f10": fn,
                "f11": tp,
                "Accuracy": round(acc, 3),
                "Recall": round(rec, 3),
                "Precision": round(prec, 3),
                "F1-score": round(f1, 3),
            }
        )

    return pd.DataFrame(filas)


# =========================
# PROCESO PRINCIPAL OE2 LGBM
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

    # ---------- TABLA TIPO “TABLA 13” ----------
    tabla = pd.DataFrame({"Nivel": LIKERT_LABELS})

    # ---------- FIGURA CON 2 ZONAS: CABECERA (líneas + textos) Y GRÁFICO ----------
    fig = plt.figure(figsize=(8.5, 5.0), dpi=160)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.05)

    ax_top = fig.add_subplot(gs[0])   # cabecera tipo “Figura 31”
    ax = fig.add_subplot(gs[1])       # gráfico de líneas

    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")

    handles_exp = []
    handles_pred = []

    # Para construir OE2 a nivel estudiante
    preds_por_item = {}

    # ---------- MODELOS POR ÍTEM Y GRÁFICO PRINCIPAL ----------
    for item in ITEMS:
        if item not in df.columns:
            raise ValueError(f"No se encontró la columna {item} en {RUTA_DATOS}")

        nombre_bonito = ITEMS_LABELS.get(item, item)

        feature_cols = ["age", "genero_bin"] + [c for c in PHQ_COLS if c != item]
        X = df[feature_cols].copy()
        y = df[item].astype(int).copy()  # 0..3

        clf = make_lgbm()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        # Predicciones con CV 5-fold
        y_pred = cross_val_predict(clf, X, y, cv=skf, method="predict")
        proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")

        preds_por_item[item] = {
            "y_true": y.values,
            "y_pred": y_pred,
            "proba": proba,
        }

        # Conteos por categoría para la tabla tipo Tabla 13
        counts_exp = pd.Series(y).value_counts().reindex(LIKERT_VALUES, fill_value=0)
        counts_pred = (
            pd.Series(y_pred).value_counts().reindex(LIKERT_VALUES, fill_value=0)
        )

        col_pred = f"Predicción {nombre_bonito}"
        col_exp = nombre_bonito
        tabla[col_pred] = counts_pred.values
        tabla[col_exp] = counts_exp.values

        x = np.arange(len(LIKERT_VALUES))

        # Línea esperada
        line_exp, = ax.plot(
            x,
            counts_exp.values,
            marker="o",
            linewidth=2,
        )
        # Línea predicha
        line_pred, = ax.plot(
            x,
            counts_pred.values,
            marker="o",
            linewidth=2,
            linestyle="--",
        )

        handles_exp.append((nombre_bonito, line_exp))
        handles_pred.append((f"Predicción {nombre_bonito}", line_pred))

    # Fila TOTAL de la tabla
    tot = {"Nivel": "TOTAL"}
    for col in tabla.columns[1:]:
        tot[col] = int(tabla[col].sum())
    tabla = pd.concat([tabla, pd.DataFrame([tot])], ignore_index=True)

    ruta_tabla = DIR_OUT / "tabla_oe2_phq9_pred_vs_esp.csv"
    tabla.to_csv(ruta_tabla, index=False, encoding="utf-8-sig")
    print(f"[OK] Tabla guardada en: {ruta_tabla}")

    # ---------- CONFIGURACIÓN DEL GRÁFICO PRINCIPAL ----------
    ax.set_xticks(np.arange(len(LIKERT_LABELS)))
    ax.set_xticklabels(LIKERT_LABELS)
    ax.set_xlabel("Nivel de frecuencia del síntoma")
    ax.set_ylabel("Número de estudiantes")
    ax.grid(axis="y", alpha=0.2)

    sintomas_titulo = ", ".join(ITEMS_LABELS[i] for i in ITEMS)
    fig.suptitle(
        f"Gráfico comparación entre diagnósticos predichos y diagnósticos esperados\n"
        f"con respecto a los síntomas de {sintomas_titulo}.",
        fontsize=12,
        y=0.98,
    )

    # ---------- “LEYENDA” PERSONALIZADA ARRIBA ----------
    ys = [0.80, 0.55, 0.30]  # tres filas

    # Columna izquierda: observados
    for (texto, line), y in zip(handles_exp, ys):
        color = line.get_color()
        ax_top.plot([0.05, 0.13], [y, y], color=color, linewidth=2)
        ax_top.text(0.135, y, texto, va="center", fontsize=8)

    # Columna derecha: predicciones
    for (texto, line), y in zip(handles_pred, ys):
        color = line.get_color()
        ax_top.plot([0.65, 0.73], [y, y], color=color,
                    linewidth=2, linestyle="--")
        ax_top.text(0.735, y, texto, va="center", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.88))

    ruta_fig = DIR_OUT / "fig_oe2_phq9_sintomas_pred_vs_esp.png"
    fig.savefig(ruta_fig, dpi=300)
    plt.close(fig)
    print(f"[OK] Figura guardada en:", ruta_fig)

    # ======================================================
    #   MÉTRICAS GLOBALES DEL OE2 A NIVEL ESTUDIANTE
    # ======================================================

    # Nivel esperado OE2 = máximo de los 3 síntomas (0–3)
    y_oe2_true = df[ITEMS].max(axis=1).astype(int).values

    # Nivel predicho OE2 = máximo de las predicciones de los 3 modelos
    preds_matrix = np.column_stack(
        [preds_por_item[item]["y_pred"] for item in ITEMS]
    )
    y_oe2_pred = preds_matrix.max(axis=1)

    # Probabilidad OE2 = promedio de las probabilidades de los 3 modelos
    proba_stack = np.stack(
        [preds_por_item[item]["proba"] for item in ITEMS], axis=0
    )  # (3, n, 4)
    proba_oe2 = proba_stack.mean(axis=0)  # (n, 4)

    # --- Matriz de confusión ---
    cm = confusion_matrix(y_oe2_true, y_oe2_pred, labels=LIKERT_VALUES)

    xtick_labels = [
        "Nada",
        "Varios\ndías",
        "Más de la\nmitad de los días",
        "Casi todos\nlos días",
    ]
    ytick_labels = [
        "Nada",
        "Varios días",
        "Más de la mitad\nde los días",
        "Casi todos los días",
    ]

    fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.8), dpi=160)
    im = ax_cm.imshow(cm, cmap="Blues", aspect="auto")

    ax_cm.set_title("Matriz de confusión", fontsize=12)
    ax_cm.set_xlabel("Predicho", fontsize=10)
    ax_cm.set_ylabel("Esperado", fontsize=10)

    ax_cm.set_xticks(range(len(xtick_labels)))
    ax_cm.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax_cm.set_yticks(range(len(ytick_labels)))
    ax_cm.set_yticklabels(ytick_labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, int(cm[i, j]),
                       ha="center", va="center", fontsize=9)

    cbar = plt.colorbar(im, ax=ax_cm)
    cbar.set_label("Casos")

    fig_cm.tight_layout()
    ruta_cm = DIR_OUT / "matriz_confusion_oe2_phq9.png"
    fig_cm.savefig(ruta_cm, dpi=300)
    plt.close(fig_cm)
    print(f"[OK] Matriz de confusión guardada en:", ruta_cm)

    # --- Métricas globales ---
    acc = accuracy_score(y_oe2_true, y_oe2_pred)
    bal_acc = balanced_accuracy_score(y_oe2_true, y_oe2_pred)
    f1_macro = f1_score(y_oe2_true, y_oe2_pred, average="macro")

    try:
        auc_macro = roc_auc_score(
            y_oe2_true, proba_oe2, multi_class="ovr", average="macro"
        )
    except ValueError:
        auc_macro = None

    brier = _brier_multiclass(y_oe2_true, proba_oe2, LIKERT_VALUES)

    rep = classification_report(
        y_oe2_true,
        y_oe2_pred,
        labels=LIKERT_VALUES,
        target_names=LIKERT_LABELS,
        output_dict=True,
        zero_division=0,
    )

    # ---- CSV tipo tabla12_ovr PER SÍNTOMA (solo f00, f01, f10, f11) ----
    df_ovr_multi = None
    for item in ITEMS:
        nombre_bonito = ITEMS_LABELS.get(item, item)
        y_true_i = preds_por_item[item]["y_true"]
        y_pred_i = preds_por_item[item]["y_pred"]

        df_item = ovr_counts_and_metrics(y_true_i, y_pred_i, LIKERT_VALUES)
        # nos quedamos SOLO con los conteos
        df_item = df_item[["Nivel", "f00", "f01", "f10", "f11"]]

        # renombrar columnas excepto "Nivel"
        rename_map = {
            col: f"{nombre_bonito}_{col}"
            for col in df_item.columns
            if col != "Nivel"
        }
        df_item = df_item.rename(columns=rename_map)

        if df_ovr_multi is None:
            df_ovr_multi = df_item
        else:
            df_ovr_multi = df_ovr_multi.merge(df_item, on="Nivel")

    ruta_ovr = DIR_OUT / "tabla_oe2_phq9_ovr.csv"
    df_ovr_multi.to_csv(ruta_ovr, index=False, encoding="utf-8-sig")
    print(f"[OK] Tabla OVR por síntoma guardada en:", ruta_ovr)

    # ---- CSV DE MÉTRICAS POR SÍNTOMA (resumen tipo metricas_por_sintoma) ----
    filas_sintomas = []
    for item in ITEMS:
        nombre_bonito = ITEMS_LABELS.get(item, item)
        y_true_i = preds_por_item[item]["y_true"]
        y_pred_i = preds_por_item[item]["y_pred"]
        proba_i = preds_por_item[item]["proba"]

        acc_i = accuracy_score(y_true_i, y_pred_i)
        bal_acc_i = balanced_accuracy_score(y_true_i, y_pred_i)
        prec_i = precision_score(y_true_i, y_pred_i, average="macro", zero_division=0)
        rec_i = recall_score(y_true_i, y_pred_i, average="macro", zero_division=0)
        f1_i = f1_score(y_true_i, y_pred_i, average="macro", zero_division=0)

        try:
            auc_i = roc_auc_score(
                y_true_i, proba_i, multi_class="ovr", average="macro"
            )
        except ValueError:
            auc_i = None

        brier_i = _brier_multiclass(y_true_i, proba_i, LIKERT_VALUES)

        filas_sintomas.append(
            {
                "Sintoma": nombre_bonito,
                "Precision": round(prec_i, 3),
                "Recall": round(rec_i, 3),
                "F1": round(f1_i, 3),
                "Accuracy": round(acc_i, 3),
                "BalancedAccuracy": round(bal_acc_i, 3),
                "AUC_OVR": None if auc_i is None else round(auc_i, 3),
                "Brier": round(brier_i, 3),
                "n": int(len(y_true_i)),
            }
        )

    df_sintomas = pd.DataFrame(filas_sintomas)
    ruta_sintomas = DIR_OUT / "metricas_oe2_phq9_por_sintoma.csv"
    df_sintomas.to_csv(ruta_sintomas, index=False, encoding="utf-8-sig")
    print(f"[OK] Métricas por síntoma guardadas en:", ruta_sintomas)

    # ---- CSV tipo metricas_UpperBound GLOBAL ----
    filas = []
    for lbl, nombre in zip(LIKERT_VALUES, LIKERT_LABELS):
        d = rep.get(nombre, {})
        filas.append(
            {
                "Clase": nombre,
                "Precision": round(d.get("precision", 0), 3),
                "Recall": round(d.get("recall", 0), 3),
                "F1": round(d.get("f1-score", 0), 3),
                "Soporte": int(d.get("support", 0)),
            }
        )

    filas.append(
        {
            "Clase": "GLOBAL",
            "Precision": round(rep["macro avg"]["precision"], 3),
            "Recall": round(rep["macro avg"]["recall"], 3),
            "F1": round(rep["macro avg"]["f1-score"], 3),
            "Soporte": int(len(y_oe2_true)),
            "Accuracy": round(acc, 3),
            "BalancedAccuracy": round(bal_acc, 3),
            "AUC_OVR": None if auc_macro is None else round(auc_macro, 3),
            "Brier": round(brier, 3),
            "n": int(len(y_oe2_true)),
        }
    )

    df_metricas = pd.DataFrame(filas)
    ruta_metricas = DIR_OUT / "metricas_oe2_phq9.csv"
    df_metricas.to_csv(ruta_metricas, index=False, encoding="utf-8-sig")
    print(f"[OK] Métricas globales guardadas en:", ruta_metricas)

    # ======================================================
    #   CSV RESUMEN POR DIMENSIÓN FISIOLÓGICA (LightGBM)
    # ======================================================
    prec_macro_global = rep["macro avg"]["precision"]
    rec_macro_global = rep["macro avg"]["recall"]
    f1_macro_global = rep["macro avg"]["f1-score"]

    df_global_modelo = pd.DataFrame(
        [
            {
                "Modelo": "LightGBM",
                "Accuracy": round(acc, 3),
                "BalancedAccuracy": round(bal_acc, 3),
                "Precision_macro": round(prec_macro_global, 3),
                "Recall_macro": round(rec_macro_global, 3),
                "F1_macro": round(f1_macro_global, 3),
                "AUC_OVR": None if auc_macro is None else round(auc_macro, 3),
                "Brier": round(brier, 3),
                "n": int(len(y_oe2_true)),
            }
        ]
    )

    ruta_global = DIR_OUT / "metricas_dimension_fisiologica_lgbm.csv"
    df_global_modelo.to_csv(ruta_global, index=False, encoding="utf-8-sig")
    print(f"[OK] Métricas globales por dimensión (LGBM) guardadas en:", ruta_global)


if __name__ == "__main__":
    main()
