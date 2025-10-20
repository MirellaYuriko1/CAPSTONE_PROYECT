# ml/ADABOOST/train_model_adaboost.py
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

# =========================
# Configuración y rutas
# =========================
SEED = 42
np.random.seed(SEED); random.seed(SEED)

RUTA_DATOS = Path("data/final/phq9_final.csv")

DIR_OUT = Path("ml/ADABOOST/resultados")
DIR_OUT.mkdir(parents=True, exist_ok=True)

RUTA_CURVA                 = DIR_OUT / "curva_entrenamiento_validacion_adaboost.png"
RUTA_CURVA_CSV             = DIR_OUT / "learning_curve_adaboost.csv"
RUTA_METRICAS              = DIR_OUT / "metricas_adaboost.json"
RUTA_REPORTE_CLASIFICACION = DIR_OUT / "reporte_clasificacion_adaboost.txt"
RUTA_MODELO                = DIR_OUT / "modelo_adaboost.pkl"
RUTA_PARAMS                = DIR_OUT / "parametros_adaboost.json"

EVAL_TEST = True  # pon False si quieres mostrar/guardar solo ENTRENAMIENTO

FEATURES = ["age","genero_bin","phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"]
TARGET   = "categoryphq"

CLASSES = ["Mínimo","Leve","Moderada","Moderadamente severa","Severa"]
def idx_to_name(arr_int): return [CLASSES[i-1] for i in arr_int]

# =========================
# Hiperparámetros AdaBoost + árbol base
# =========================
# Árbol base (stump) regularizado; class_weight para compensar desbalance
BASE_TREE_PARAMS = dict(
    max_depth=2,
    min_samples_leaf=5,
    min_samples_split=12,
    class_weight="balanced",
    random_state=SEED
)

ADA_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    algorithm="SAMME.R",  # para multiclase con probabilidades
    random_state=SEED
)

def make_adaboost():
    base_tree = DecisionTreeClassifier(**BASE_TREE_PARAMS)
    # Compatibilidad con distintas versiones de scikit-learn
    try:
        return AdaBoostClassifier(estimator=base_tree, **ADA_PARAMS)
    except TypeError:
        return AdaBoostClassifier(base_estimator=base_tree, **ADA_PARAMS)

# =========================
# Curva de aprendizaje (F1-macro, CV 5-fold en TRAIN)
# =========================
def construir_modelo_AB(estimator, X_train, y_train, cv, ruta_png, ruta_csv, titulo="Modelo AdaBoost"):
    train_sizes_rel = np.linspace(0.1, 1.0, 8)

    sizes_abs, train_scores, valid_scores = learning_curve(
        estimator=estimator,
        X=X_train, y=y_train,
        train_sizes=train_sizes_rel,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        shuffle=True,
        random_state=SEED
    )

    tr_mean, tr_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    va_mean, va_std = valid_scores.mean(axis=1), valid_scores.std(axis=1)

    # Gráfica (±1σ)
    plt.figure(figsize=(7,5))
    plt.plot(sizes_abs, tr_mean, marker="o", label="Entrenamiento")
    plt.fill_between(sizes_abs, tr_mean-tr_std, tr_mean+tr_std, alpha=0.2)
    plt.plot(sizes_abs, va_mean, marker="s", label="Validación")
    plt.fill_between(sizes_abs, va_mean-va_std, va_mean+va_std, alpha=0.2)
    plt.title(titulo)
    plt.xlabel("Tamaño del conjunto de entrenamiento (TRAIN)")
    plt.ylabel("F1-macro (CV)")
    plt.legend(); plt.tight_layout()
    plt.savefig(ruta_png, dpi=200); plt.close()

    # CSV
    pd.DataFrame({
        "train_size": sizes_abs,
        "f1_train_mean": tr_mean, "f1_train_std": tr_std,
        "f1_valid_mean": va_mean, "f1_valid_std": va_std
    }).to_csv(ruta_csv, index=False)

    return sizes_abs.tolist(), tr_mean.tolist(), va_mean.tolist()

# =========================
# Entrenar y evaluar
# =========================
def main():
    if not RUTA_DATOS.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {RUTA_DATOS}")

    df = pd.read_csv(RUTA_DATOS)
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    # sanity checks
    assert not X.isna().any().any(), "Hay NaN en features; revisa el preprocesamiento."
    assert TARGET not in FEATURES, "El target no puede estar en FEATURES."

    # split 80/20 estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"[INFO] Entrenamiento = {len(X_train)} | Prueba = {len(X_test)}")

    ada = make_adaboost()

    # CV estratificada SOLO en TRAIN
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1) Curva de aprendizaje (CV en TRAIN)
    sizes_abs, tr_scores, va_scores = construir_modelo_AB(
        ada, X_train, y_train, cv, RUTA_CURVA, RUTA_CURVA_CSV, titulo="Modelo AdaBoost"
    )

    # 2) Puntaje global de CV en TRAIN
    cv_f1 = cross_val_score(ada, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"[CV-5] F1-macro (TRAIN) = {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # 3) Entrenamiento final con TODO TRAIN
    ada.fit(X_train, y_train)

    # 4) (Opcional) Evaluación FINAL en TEST
    rep = ""
    test_metrics = None
    if EVAL_TEST:
        y_pred = ada.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        f1m  = f1_score(y_test, y_pred, average="macro")

        print("\n=== MÉTRICAS EN TEST ===")
        print(f"accuracy        = {acc:.4f}")
        print(f"balanced_acc    = {bacc:.4f}")
        print(f"macroF1         = {f1m:.4f}")

        rep = classification_report(
            idx_to_name(y_test.values),
            idx_to_name(y_pred),
            target_names=CLASSES,
            zero_division=0
        )
        print("\n=== REPORTE POR CLASE (TEST) ===")
        print(rep)

        test_metrics = {"accuracy": float(acc), "balanced_accuracy": float(bacc), "macro_f1": float(f1m)}

    # 5) Guardar artefactos
    metricas = {
        "model": "AdaBoostClassifier",
        "base_tree_params": BASE_TREE_PARAMS,
        "adaboost_params": ADA_PARAMS,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "cv5_train": {
            "f1_macro_mean": float(cv_f1.mean()),
            "f1_macro_std": float(cv_f1.std())
        },
        "curve": {
            "train_sizes": sizes_abs,
            "f1_train_mean": [float(x) for x in tr_scores],
            "f1_valid_mean": [float(x) for x in va_scores]
        },
        "test_metrics": test_metrics,
        "labels": CLASSES
    }
    with open(RUTA_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print("[OK] Métricas guardadas en:", RUTA_METRICAS)

    if EVAL_TEST and rep:
        with open(RUTA_REPORTE_CLASIFICACION, "w", encoding="utf-8") as f:
            f.write(rep)
        print("[OK] Reporte de clasificación guardado en:", RUTA_REPORTE_CLASIFICACION)

    joblib.dump(ada, RUTA_MODELO)
    with open(RUTA_PARAMS, "w", encoding="utf-8") as f:
        json.dump({
            "base_tree_params": BASE_TREE_PARAMS,
            "adaboost_params": ADA_PARAMS,
            "features_used": FEATURES
        }, f, ensure_ascii=False, indent=2)
    print("[OK] Modelo guardado en:", RUTA_MODELO)
    print("[OK] Parámetros guardados en:", RUTA_PARAMS)

if __name__ == "__main__":
    main()
