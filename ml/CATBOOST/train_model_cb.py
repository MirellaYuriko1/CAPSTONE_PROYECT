# ml/CATBOOST/train_model_catboost.py
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
import joblib

# --- CatBoost (wrapper sklearn) ---
try:
    from catboost import CatBoostClassifier
except ImportError:
    raise SystemExit("CatBoost no está instalado. Ejecuta: pip install catboost")

# =========================
# Configuración y rutas
# =========================
SEED = 42
np.random.seed(SEED); random.seed(SEED)

RUTA_DATOS = Path("data/final/phq9_final.csv")

DIR_OUT = Path("ml/CATBOOST/resultados")
DIR_OUT.mkdir(parents=True, exist_ok=True)

RUTA_CURVA                 = DIR_OUT / "curva_entrenamiento_validacion_catb.png"
RUTA_CURVA_CSV             = DIR_OUT / "learning_curve_catb.csv"
RUTA_METRICAS              = DIR_OUT / "metricas_catb.json"
RUTA_REPORTE_CLASIFICACION = DIR_OUT / "reporte_clasificacion_catb.txt"
RUTA_MODELO                = DIR_OUT / "modelo_catb.pkl"
RUTA_PARAMS                = DIR_OUT / "parametros_catb.json"

EVAL_TEST = True  # pon False si deseas omitir evaluación en TEST

FEATURES = ["age","genero_bin","phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"]
TARGET   = "categoryphq"

CLASSES = ["Mínimo","Leve","Moderada","Moderadamente severa","Severa"]
def idx_to_name0(arr_idx): return [CLASSES[i] for i in arr_idx]  # y en 0..4

# =========================
# Hiperparámetros CatBoost
# =========================
CAT_PARAMS = dict(
    loss_function="MultiClass",
    eval_metric="TotalF1",         
    learning_rate=0.05,
    iterations=1000,
    depth=6,                      
    l2_leaf_reg=3.0,               
    random_strength=1.0,
    bootstrap_type="Bayesian",
    bagging_temperature=1.0,
    rsm=0.8,                       
    thread_count=-1,
    random_state=SEED,
    verbose=False
)

# =========================
# Utilidad: pesos balanceados por clase (opcional)
# =========================
def class_weights_from_counts(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    inv = 1.0 / np.maximum(counts, 1.0)
    weights = (inv / inv.sum()) * n_classes  # normaliza alrededor de 1.0
    return weights.tolist()

# =========================
# Curva de aprendizaje
# =========================
def construir_modelo_CB(estimator, X_train, y_train, cv, ruta_png, ruta_csv,
                      titulo="Modelo CatBoost"):
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

    # CatBoost requiere etiquetas 0..num_class-1
    X = df[FEATURES].astype(np.float32).copy()
    y = df[TARGET].astype(int).values - 1

    assert not np.isnan(X.values).any(), "Hay NaN en features; revisa el preprocesamiento."
    assert set(np.unique(y)) == set(range(5)), "Las etiquetas deben estar en 0..4."

    # Split 80/20 estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"[INFO] Entrenamiento = {len(X_train)} | Prueba = {len(X_test)}")

    # Pesos por clase (balanceo)
    class_weights = class_weights_from_counts(y_train, n_classes=5)

    catb = CatBoostClassifier(**CAT_PARAMS, class_weights=class_weights)

    # CV estratificada SOLO en TRAIN
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1) Curva de aprendizaje (CV en TRAIN)
    sizes_abs, tr_scores, va_scores = construir_modelo_CB(
        catb, X_train, y_train, cv, RUTA_CURVA, RUTA_CURVA_CSV, titulo="Modelo CatBoost"
    )

    # 2) Puntaje global CV en TRAIN
    cv_f1 = cross_val_score(catb, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"[CV-5] F1-macro (TRAIN) = {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # 3) Entrenamiento final con TODO TRAIN
    catb.fit(X_train, y_train)

    # 4) (Opcional) Evaluación FINAL en TEST
    rep = ""
    test_metrics = None
    if EVAL_TEST:
        y_pred = catb.predict(X_test)
        # y_pred sale shape (n,1); convertir a 1D
        y_pred = y_pred.astype(int).ravel()

        acc  = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        f1m  = f1_score(y_test, y_pred, average="macro")

        print("\n=== MÉTRICAS EN TEST ===")
        print(f"accuracy        = {acc:.4f}")
        print(f"balanced_acc    = {bacc:.4f}")
        print(f"macroF1         = {f1m:.4f}")

        rep = classification_report(
            idx_to_name0(y_test),
            idx_to_name0(y_pred),
            target_names=CLASSES,
            zero_division=0
        )
        print("\n=== REPORTE POR CLASE (TEST) ===")
        print(rep)

        test_metrics = {"accuracy": float(acc), "balanced_accuracy": float(bacc), "macro_f1": float(f1m)}

    # 5) Guardar artefactos
    metricas = {
        "model": "CatBoostClassifier",
        "params": CAT_PARAMS,
        "class_weights": class_weights,
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

    joblib.dump(catb, RUTA_MODELO)
    with open(RUTA_PARAMS, "w", encoding="utf-8") as f:
        json.dump({"params": CAT_PARAMS, "class_weights": class_weights, "features_used": FEATURES}, f, ensure_ascii=False, indent=2)
    print("[OK] Modelo guardado en:", RUTA_MODELO)
    print("[OK] Parámetros guardados en:", RUTA_PARAMS)

if __name__ == "__main__":
    main()
