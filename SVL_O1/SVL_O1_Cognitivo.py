# SVL_O1/SVL_O1.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

# =========================
# CONFIGURACIÓN GENERAL
# =========================
SEED = 42
RUTA_DATOS = Path("data/final/phq9_final.csv")

DIR_OUT = Path("SVL_O1/resultados")
DIR_OUT.mkdir(parents=True, exist_ok=True)

# Síntomas que quieres analizar en este OE (PHQ-9)
ITEMS = ["phq1", "phq2", "phq6"]
ITEMS_LABELS = {
    "phq1": "Pérdida de interés",
    "phq2": "Estado de ánimo",
    "phq6": "Sentimientos de fracaso",
}

# Escala Likert del PHQ-9
LIKERT_VALUES = [0, 1, 2, 3]
LIKERT_LABELS = [
    "Nada",
    "Varios días",
    "Más de la mitad de los días",
    "Casi todos los días",
]

PHQ_COLS = [f"phq{i}" for i in range(1, 10)]

# Hiperparámetros del SVM lineal (tu modelo ganador)
SVM_PARAMS = dict(
    kernel="linear",
    C=1.0,
    class_weight="balanced",
    probability=True,
    random_state=SEED,
)


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


# =========================
# PROCESO PRINCIPAL OE1
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

    # ---------- FIGURA CON 2 ZONAS ----------
    fig = plt.figure(figsize=(8.5, 5.0), dpi=160)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.05)

    ax_top = fig.add_subplot(gs[0])   # cabecera tipo “Figura 31”
    ax = fig.add_subplot(gs[1])       # gráfico de líneas

    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")

    handles_exp = []
    handles_pred = []

    # Para construir OE1 a nivel estudiante
    preds_por_item = {}  # item -> dict con y_pred y proba

    # ---------- MODELOS POR ÍTEM Y GRÁFICO PRINCIPAL ----------
    for item in ITEMS:
        if item not in df.columns:
            raise ValueError(f"No se encontró la columna {item} en {RUTA_DATOS}")

        nombre_bonito = ITEMS_LABELS.get(item, item)

        feature_cols = ["age", "genero_bin"] + [c for c in PHQ_COLS if c != item]
        X = df[feature_cols].copy()
        y = df[item].astype(int).copy()

        clf = SVC(**SVM_PARAMS)
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

    ruta_tabla = DIR_OUT / "tabla_oe1_phq9_pred_vs_esp.csv"
    tabla.to_csv(ruta_tabla, index=False, encoding="utf-8-sig")
    print(f"[OK] Tabla guardada en: {ruta_tabla}")

    # ---------- CONFIGURACIÓN DEL GRÁFICO ----------
    ax.set_xticks(np.arange(len(LIKERT_LABELS)))
    ax.set_xticklabels(LIKERT_LABELS)
    ax.set_xlabel("Nivel de frecuencia del síntoma (PHQ-9)")
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

    ruta_fig = DIR_OUT / "fig_oe1_phq9_sintomas_pred_vs_esp.png"
    fig.savefig(ruta_fig, dpi=300)
    plt.close(fig)
    print(f"[OK] Figura guardada en: {ruta_fig}")

    # ======================================================
    #   MÉTRICAS GLOBALES DEL OE1 A NIVEL ESTUDIANTE
    # ======================================================

    # Nivel esperado OE1 = máximo de los 3 síntomas (0–3)
    y_oe1_true = df[ITEMS].max(axis=1).astype(int).values

    # Nivel predicho OE1 = máximo de las predicciones de los 3 modelos
    preds_matrix = np.column_stack(
        [preds_por_item[item]["y_pred"] for item in ITEMS]
    )
    y_oe1_pred = preds_matrix.max(axis=1)

    # Probabilidad OE1 = promedio de las probabilidades de los 3 modelos
    proba_stack = np.stack(
        [preds_por_item[item]["proba"] for item in ITEMS], axis=0
    )  # (3, n, 4)
    proba_oe1 = proba_stack.mean(axis=0)  # (n, 4)

    # --- Matriz de confusión global OE1 ---
    cm = confusion_matrix(y_oe1_true, y_oe1_pred, labels=LIKERT_VALUES)

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
    ruta_cm = DIR_OUT / "matriz_confusion_oe1_phq9.png"
    fig_cm.savefig(ruta_cm, dpi=300)
    plt.close(fig_cm)
    print(f"[OK] Matriz de confusión guardada en: {ruta_cm}")

    # --- Métricas globales (OE1 completo) ---
    acc = accuracy_score(y_oe1_true, y_oe1_pred)
    bal_acc = balanced_accuracy_score(y_oe1_true, y_oe1_pred)
    f1_macro = f1_score(y_oe1_true, y_oe1_pred, average="macro")

    try:
        auc_macro = roc_auc_score(
            y_oe1_true, proba_oe1, multi_class="ovr", average="macro"
        )
    except ValueError:
        auc_macro = None

    brier = _brier_multiclass(y_oe1_true, proba_oe1, LIKERT_VALUES)

    rep = classification_report(
        y_oe1_true,
        y_oe1_pred,
        labels=LIKERT_VALUES,
        target_names=LIKERT_LABELS,
        output_dict=True,
        zero_division=0,
    )

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
            "Soporte": int(len(y_oe1_true)),
            "Accuracy": round(acc, 3),
            "BalancedAccuracy": round(bal_acc, 3),
            "AUC_OVR": None if auc_macro is None else round(auc_macro, 3),
            "Brier": round(brier, 3),
            "n": int(len(y_oe1_true)),
        }
    )

    df_metricas_global = pd.DataFrame(filas)
    ruta_metricas_global = DIR_OUT / "metricas_oe1_phq9.csv"
    df_metricas_global.to_csv(ruta_metricas_global, index=False, encoding="utf-8-sig")
    print(f"[OK] Métricas globales guardadas en: {ruta_metricas_global}")

    # ======================================================
    #   NUEVA TABLA DE MÉTRICAS POR SÍNTOMA
    # ======================================================
    filas_sintomas = []
    for item in ITEMS:
        nombre = ITEMS_LABELS[item]
        y_true_item = preds_por_item[item]["y_true"]
        y_pred_item = preds_por_item[item]["y_pred"]
        proba_item = preds_por_item[item]["proba"]

        acc_item = accuracy_score(y_true_item, y_pred_item)
        bal_acc_item = balanced_accuracy_score(y_true_item, y_pred_item)
        f1_item = f1_score(y_true_item, y_pred_item, average="macro")

        try:
            auc_item = roc_auc_score(
                y_true_item, proba_item, multi_class="ovr", average="macro"
            )
        except ValueError:
            auc_item = None

        brier_item = _brier_multiclass(y_true_item, proba_item, LIKERT_VALUES)

        rep_item = classification_report(
            y_true_item,
            y_pred_item,
            labels=LIKERT_VALUES,
            target_names=LIKERT_LABELS,
            output_dict=True,
            zero_division=0,
        )

        filas_sintomas.append(
            {
                "Sintoma": nombre,
                "Precision": round(rep_item["macro avg"]["precision"], 3),
                "Recall": round(rep_item["macro avg"]["recall"], 3),
                "F1": round(rep_item["macro avg"]["f1-score"], 3),
                "Accuracy": round(acc_item, 3),
                "BalancedAccuracy": round(bal_acc_item, 3),
                "AUC_OVR": None if auc_item is None else round(auc_item, 3),
                "Brier": round(brier_item, 3),
                "n": int(len(y_true_item)),
            }
        )

    df_metricas_sintomas = pd.DataFrame(filas_sintomas)
    ruta_metricas_sintomas = DIR_OUT / "metricas_oe1_phq9_por_sintoma.csv"
    df_metricas_sintomas.to_csv(
        ruta_metricas_sintomas, index=False, encoding="utf-8-sig"
    )
    print(f"[OK] Métricas por síntoma guardadas en: {ruta_metricas_sintomas}")


if __name__ == "__main__":
    main()
