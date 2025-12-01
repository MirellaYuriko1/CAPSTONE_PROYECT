import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# ==========================================
# CONFIGURACIÓN
# ==========================================
BASE_DIR = Path(__file__).parent 
DIR_COMPARATIVAS = BASE_DIR / "COMPARATIVAS_FINALES"
DIR_COMPARATIVAS.mkdir(parents=True, exist_ok=True)

# Diccionario para nombres más bonitos en la leyenda
NOMBRES_LINDOS = {
    "cb": "CatBoost",
    "rf": "Random Forest",
    "dt": "Decision Tree",
    "xgb": "XGBoost",
    "svm": "SVM",
    "knn": "KNN",
    "lr": "Regresión Logística",
    "nb": "Naive Bayes",
    "gb": "Gradient Boosting"
}

# --- ESTILO VISUAL ---
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

# ==========================================================
# 1. BARRAS COMPARATIVAS (Métricas CSV)
# ==========================================================
print("--- 1. Generando Gráficos de Barras (Métricas) ---")
archivos_metricas = list(BASE_DIR.glob("**/resultados/metricas_modelo_*.csv"))

if archivos_metricas:
    dfs = []
    for archivo in archivos_metricas:
        try:
            df_temp = pd.read_csv(archivo)
            dfs.append(df_temp)
        except Exception as e:
            print(f"[WARN] Error leyendo CSV {archivo.name}: {e}")

    if dfs:
        df_global = pd.concat(dfs, ignore_index=True)
        df_global.to_csv(DIR_COMPARATIVAS / "tabla_resumen_metricas.csv", index=False)

        def plot_bar(df, col_metrica, titulo, nombre_archivo, color_map):
            df_sorted = df.sort_values(by=col_metrica, ascending=False)
            plt.figure(figsize=(11, 7)) 
            cmap = plt.get_cmap(color_map)
            colors = cmap(np.linspace(0.2, 0.8, len(df_sorted)))
            bars = plt.bar(df_sorted["Modelo"], df_sorted[col_metrica], color=colors, edgecolor='grey', linewidth=0.5)
            plt.title(titulo, fontsize=16, weight='bold', pad=15)
            plt.ylabel("Puntación", fontsize=12)
            plt.ylim(0, 1.15)
            plt.xticks(rotation=45, ha='right', fontsize=11)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
            plt.tight_layout()
            plt.savefig(DIR_COMPARATIVAS / nombre_archivo, dpi=300)
            plt.close()

        cols = df_global.columns.tolist()
        if "F1_macro" in cols: plot_bar(df_global, "F1_macro", "Comparativa F1-Score (Macro)", "barras_f1.png", "viridis")
        if "Accuracy" in cols: plot_bar(df_global, "Accuracy", "Comparativa Accuracy", "barras_accuracy.png", "plasma")
        if "Recall_macro" in cols: plot_bar(df_global, "Recall_macro", "Comparativa Sensibilidad (Recall)", "barras_recall.png", "magma")
        if "Precision_macro" in cols: plot_bar(df_global, "Precision_macro", "Comparativa Precisión", "barras_precision.png", "cividis")
        print("[OK] Gráficos de barras generados.")

# ==========================================================
# 2. CURVAS ROC COMPARATIVAS (Líneas Superpuestas)
# ==========================================================
print("\n--- 2. Generando Curvas ROC (Estilo Final) ---")

archivos_roc = list(BASE_DIR.glob("**/resultados/roc_data_*.json"))

if archivos_roc:
    plt.close('all')
    
    # Volvemos a un tamaño estándar ya que la leyenda estará dentro
    fig, ax = plt.subplots(figsize=(10, 8))
    
    datos_curvas = []

    for archivo in archivos_roc:
        try:
            with open(archivo, 'r') as f:
                data = json.load(f)
                nombre_modelo = data.get("modelo")
                if not nombre_modelo:
                    stem = archivo.stem
                    codigo = stem.split("_")[-1]
                    nombre_modelo = NOMBRES_LINDOS.get(codigo, codigo.upper())
                data["nombre_display"] = nombre_modelo
                datos_curvas.append(data)
        except Exception as e: pass

    # Ordenar por AUC
    datos_curvas.sort(key=lambda x: x.get("auc", 0), reverse=True)

    if len(datos_curvas) > 10:
        colores = plt.cm.tab20(np.linspace(0, 1, len(datos_curvas)))
    else:
        colores = plt.cm.tab10(np.linspace(0, 1, len(datos_curvas)))

    for i, data in enumerate(datos_curvas):
        fpr = data.get("fpr")
        tpr = data.get("tpr")
        auc = data.get("auc", 0)
        nombre = data.get("nombre_display")
        
        if fpr and tpr:
            ax.plot(fpr, tpr, lw=2, color=colores[i], alpha=0.8,
                    label=f'{nombre} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Azar (0.5)')

    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=13)
    ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=13)
    ax.set_title('Rendimiento de Modelos: Curvas ROC', fontsize=16, weight='bold', pad=15)
    
    # --- CAMBIO AQUÍ: LEYENDA DENTRO (ABAJO A LA DERECHA) ---
    ax.legend(loc="lower right", fontsize=10, frameon=True, shadow=True, fancybox=True)
    
    ax.grid(True, linestyle=':', alpha=0.6)

    ruta_roc = DIR_COMPARATIVAS / "comparativa_roc_final.png"
    plt.savefig(ruta_roc, dpi=300) # Ya no hace falta bbox_inches='tight' estricto
    
    print(f"[OK] Curva ROC guardada en: {ruta_roc}")
    
else:
    print("[ERROR] No se encontraron archivos JSON.")

print("\n--- PROCESO FINALIZADO ---")