# ml/KNN/train_model_knn.py
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
import joblib

# =========================
# Configuración y rutas
# =========================
SEED = 42
np.random.seed(SEED); random.seed(SEED)

RUTA_DATOS = Path("data/final/phq9_final.csv")

DIR_OUT = Path("ml/KNN/resultados")
DIR_OUT.mkdir(parents=True, exist_ok=True)

RUTA_CURVA                 = DIR_OUT / "curva_entrenamiento_validacion_knn.png"
RUTA_CURVA_CSV             = DIR_OUT / "learning_curve_knn.csv"
RUTA_METRICAS              = DIR_OUT / "metricas_knn.json"
RUTA_REPORTE_CLASIFICACION = DIR_OUT / "reporte_clasificacion_knn.txt"
RUTA_MODELO                = DIR_OUT / "modelo_knn.pkl"
RUTA_PARAMS                = DIR_OUT / "parametros_knn.json"

EVAL_TEST = True  # pon False si quieres mostrar/guardar solo ENTRENAMIENTO

FEATURES = ["age","genero_bin","phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"]
TARGET   = "categoryphq"

CLASSES = ["Mínimo","Leve","Moderada","Moderadamente severa","Severa"]
def idx_to_name(arr_int): return [CLASSES[i-1] for i in arr_int]

# =========================
# Función curva de aprendizaje (misma que usas en otros modelos)
# =========================
def construir_modelo_KNN(estimator, X_train, y_train, cv, ruta_png, ruta_csv, titulo="Modelo KNN"):
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

    # Gráfica
    plt.figure(figsize=(7,5))
    plt.plot(sizes_abs, tr_mean, marker="o", label="Entrenamiento")
    plt.fill_between(sizes_abs, tr_mean-tr_std, tr_mean+tr_std, alpha=0.2)
    plt.plot(sizes_abs, va_mean, marker="s", label="Validación")
    plt.fill_between(sizes_abs, va_mean-va_std, va_mean+va_std, alpha=0.2)
    plt.title(titulo)
    plt.xlabel("Tamaño del conjunto de entrenamiento (TRAIN)")
    plt.ylabel("F1-macro (CV)")
    plt.legend(); plt.tight_layout(); plt.savefig(ruta_png, dpi=200); plt.close()

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

    # --------- Pipeline: escalado de continuas (obligatorio en KNN) ---------
    cont_cols = ["age","phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"]
    cat_cols  = ["genero_bin"]  # no escalar

    pre = ColumnTransformer([
        ("cont", StandardScaler(), cont_cols),
        ("cat",  "passthrough",    cat_cols),
    ], remainder="drop")

    knn = KNeighborsClassifier()
    pipe = Pipeline([("pre", pre), ("clf", knn)])

    # --------- Búsqueda de hiperparámetros ---------
    param_grid = {
        "clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["minkowski"],  # p=2 → euclidiana
        "clf__p": [2],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    grid.fit(X_train, y_train)
    best_pipe = grid.best_estimator_
    print("[CV-Grid] Mejor combinación:", grid.best_params_, " | F1-macro:", round(grid.best_score_, 4))

    # --------- Curva de aprendizaje con los mejores hiperparámetros ---------
    sizes_abs, tr_scores, va_scores = construir_modelo_KNN(
        best_pipe, X_train, y_train, cv, RUTA_CURVA, RUTA_CURVA_CSV, titulo="Modelo KNN"
    )

    # --------- Entrenamiento final ---------
    best_pipe.fit(X_train, y_train)

    # --------- Evaluación opcional en TEST ---------
    rep = ""
    test_metrics = None
    if EVAL_TEST:
        y_pred = best_pipe.predict(X_test)
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

    # --------- Guardar artefactos ---------
    metricas = {
        "model": "KNeighborsClassifier",
        "best_params": grid.best_params_,
        "cv5_best_f1_macro": float(grid.best_score_),
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

    joblib.dump(best_pipe, RUTA_MODELO)  # guarda pipeline completo (escalado + KNN)
    with open(RUTA_PARAMS, "w", encoding="utf-8") as f:
        json.dump({"best_params": grid.best_params_, "features_used": FEATURES}, f, ensure_ascii=False, indent=2)
    print("[OK] Modelo guardado en:", RUTA_MODELO)
    print("[OK] Parámetros guardados en:", RUTA_PARAMS)

if __name__ == "__main__":
    main()
