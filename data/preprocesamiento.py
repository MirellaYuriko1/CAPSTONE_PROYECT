# ml/preprocesamiento.py
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =========================
# Rutas (NO MODIFICADAS)
# =========================
RUTA_CRUDO = Path("data/crudo/Test_inicial.xlsx")
RUTA_SALIDA_SIN_DUPLICADOS = Path("data/transformado/Test_sin_duplicados.csv")

RUTA_SIN_DUPLICADOS = Path("data/transformado/Test_sin_duplicados.csv")
RUTA_SALIDA_SUBCONJUNTO = Path("data/transformado/phq9_subconjunto_v1.csv")

RUTA_SUBCONJUNTO = Path("data/transformado/phq9_subconjunto_v1.csv")
RUTA_SALIDA_RANGOS_VALIDOS = Path("data/transformado/phq9_rangos_validos.csv")

RUTA_RANGOS_VALIDOS = Path("data/transformado/phq9_rangos_validos.csv")
RUTA_SALIDA_FINAL = Path("data/final/phq9_final.csv")

# =========================
# Config EDA
# =========================
EDA_DIR = Path("data/analisis EDA")
EDA_DIR.mkdir(parents=True, exist_ok=True)

PHQ_ITEMS = [f"phq{i}" for i in range(1, 10)]

def _save_simple_bar(labels, values, title, filename, xlabel="", ylabel="Conteo"):
    """Gráfico simple de barras con anotaciones (para duplicados, etc.)."""
    plt.figure(figsize=(6,4))
    plt.bar(labels, values)
    for i, v in enumerate(values):
        try:
            txt = str(int(v))
        except Exception:
            txt = f"{v}"
        plt.text(i, v, txt, ha="center", va="bottom")
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    out = EDA_DIR / filename
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[FIG] {out}")

# =========================
# ETAPA 0: Nulos (solo heatmap)
# =========================
def eliminar_nulos():
    """Genera únicamente: 00_mapa_missing.png (todas las columnas)."""
    if not RUTA_CRUDO.exists():
        raise FileNotFoundError(f"No encuentro el archivo crudo en: {RUTA_CRUDO}")

    df = pd.read_excel(RUTA_CRUDO)
    print(f"[NULOS] Filas: {len(df)} | Columnas: {len(df.columns)}")

    mask = df.isna().values
    ancho = max(10, 0.45 * len(df.columns))
    fig, ax = plt.subplots(figsize=(ancho, 6))
    ax.imshow(mask, aspect="auto", interpolation="nearest")
    ax.set_title("Mapa de valores faltantes", fontsize=14, weight="bold")
    ax.set_xlabel("Columnas"); ax.set_ylabel("Filas")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns.astype(str), rotation=90, ha="right", fontsize=9)
    fig.tight_layout()

    out = EDA_DIR / "00_mapa_missing.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] {out}")

# =========================
# ETAPA 1: Eliminar duplicados (1 gráfico simple)
# =========================
def eliminar_duplicados():
    if not RUTA_CRUDO.exists():
        raise FileNotFoundError(f"No encuentro el archivo crudo en: {RUTA_CRUDO}")

    df = pd.read_excel(RUTA_CRUDO)
    duplicados = int(df.duplicated().sum())
    unicos = len(df) - duplicados

    _save_simple_bar(
        ["Duplicados", "Únicos"],
        [duplicados, unicos],
        "",  # sin título (lo pones en el pie de figura del paper)
        "01_duplicados.png"
    )

    df_sin_dup = df.drop_duplicates()
    RUTA_SALIDA_SIN_DUPLICADOS.parent.mkdir(parents=True, exist_ok=True)
    df_sin_dup.to_csv(RUTA_SALIDA_SIN_DUPLICADOS, index=False, encoding="utf-8")
    print(f"[OK] Archivo sin duplicados guardado en: {RUTA_SALIDA_SIN_DUPLICADOS}")

# =========================
# ETAPA 2: Subconjunto PHQ-9
# =========================
COLUMNAS_UTILIZADAS = [
    "age", "gender",
    "phq1", "phq2", "phq3", "phq4", "phq5", "phq6", "phq7", "phq8", "phq9",
    "totalphq", "categoryphq"
]

def crear_subconjunto_phq9():
    if not RUTA_SIN_DUPLICADOS.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {RUTA_SIN_DUPLICADOS}")

    df = pd.read_csv(RUTA_SIN_DUPLICADOS)
    faltantes = [c for c in COLUMNAS_UTILIZADAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el archivo: {faltantes}")

    df_sub = df[COLUMNAS_UTILIZADAS].copy()
    RUTA_SALIDA_SUBCONJUNTO.parent.mkdir(parents=True, exist_ok=True)
    df_sub.to_csv(RUTA_SALIDA_SUBCONJUNTO, index=False, encoding="utf-8")

    print(f"[OK] Subconjunto PHQ-9 creado: {RUTA_SALIDA_SUBCONJUNTO} "
          f"(Filas {len(df_sub)} | Columnas {len(df_sub.columns)})")

# --- EDA: Selección de características (SOLO gráfica VERTICAL)
def eda_seleccion_caracteristicas():
    """
    Genera una única figura VERTICAL con las variables seleccionadas
    (ordenadas como en COLUMNAS_UTILIZADAS). Guarda: 02_seleccion_caracteristicas.png
    """
    if RUTA_SIN_DUPLICADOS.exists():
        df_in = pd.read_csv(RUTA_SIN_DUPLICADOS)
        columnas_iniciales = set(df_in.columns.astype(str))
        seleccionadas = [c for c in COLUMNAS_UTILIZADAS if c in columnas_iniciales]
    elif RUTA_CRUDO.exists():
        df_in = pd.read_excel(RUTA_CRUDO)
        columnas_iniciales = set(df_in.columns.astype(str))
        seleccionadas = [c for c in COLUMNAS_UTILIZADAS if c in columnas_iniciales]
    else:
        seleccionadas = COLUMNAS_UTILIZADAS[:]

    n = len(seleccionadas)
    x = np.arange(n)

    fig_w = max(10, 0.6 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    ax.bar(x, np.ones(n))
    ax.set_xticks(x)
    ax.set_xticklabels(seleccionadas, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("Características")
    ax.set_title("", fontsize=18, fontweight="bold", pad=10)

    for spine in ["right", "top", "left"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out = EDA_DIR / "02_seleccion_caracteristicas.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[FIG] {out}")

# =========================
# ETAPA 3: Validar rangos (SIN gráfico)
# =========================
def validar_rangos():
    if not RUTA_SUBCONJUNTO.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {RUTA_SUBCONJUNTO}")

    df = pd.read_csv(RUTA_SUBCONJUNTO)

    phq_cols = ["phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"]
    mask_age   = df["age"].between(12, 19)
    mask_phq   = df[phq_cols].apply(lambda s: s.between(0, 3)).all(axis=1)
    mask_total = df["totalphq"].between(0, 27)
    mask_cat   = df["categoryphq"].between(1, 5)

    mask_final = mask_age & mask_phq & mask_total & mask_cat
    df_valid = df[mask_final].copy()

    RUTA_SALIDA_RANGOS_VALIDOS.parent.mkdir(parents=True, exist_ok=True)
    df_valid.to_csv(RUTA_SALIDA_RANGOS_VALIDOS, index=False, encoding="utf-8")
    print(f"[OK] Archivo validado guardado en: {RUTA_SALIDA_RANGOS_VALIDOS}")

# =========================
# ETAPA 4: Normalizar género (con figura Antes vs Después)
# =========================
def normalizar_genero():
    """
    Normaliza 'gender' y crea 'genero_bin' (Masculino->0, Femenino->1).
    Además genera: data/analisis EDA/04_normalizacion_genero.png con la
    comparación de conteos Antes vs Después y el mapeo explícito.
    """
    if not RUTA_RANGOS_VALIDOS.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {RUTA_RANGOS_VALIDOS}")

    df = pd.read_csv(RUTA_RANGOS_VALIDOS)

    # --- Conteos "ANTES"
    gender_raw = df["gender"].astype(str).str.strip()
    vc_antes = gender_raw.value_counts()

    # --- Normalización a etiquetas homogéneas
    df["gender"] = (
        gender_raw.str.lower()
                  .replace({
                      "male": "Masculino", "masculino": "Masculino",
                      "female": "Femenino", "femenino": "Femenino"
                  })
    )

    # Advertir si hay valores no mapeados
    valores_inusuales = sorted(set(df["gender"].unique()) - {"Masculino", "Femenino"})
    if valores_inusuales:
        print(f"[WARN] Valores no estandarizados en 'gender': {valores_inusuales}")

    # --- Codificación binaria
    df["genero_bin"] = df["gender"].map({"Masculino": 0, "Femenino": 1})

    # --- Guardar dataset final
    RUTA_SALIDA_FINAL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUTA_SALIDA_FINAL, index=False, encoding="utf-8")
    print(f"[OK] Dataset final guardado en: {RUTA_SALIDA_FINAL}")

    # --- Figura: Antes vs Después
    vc_despues_texto = df["gender"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Antes (texto crudo)
    axes[0].bar(vc_antes.index.astype(str), vc_antes.values)
    for i, v in enumerate(vc_antes.values):
        axes[0].text(i, v, str(int(v)), ha="center", va="bottom")
    axes[0].set_title("Antes")
    axes[0].set_xlabel("Género (crudo)")
    axes[0].set_ylabel("Conteo")

    # Después (texto normalizado / codificado)
    axes[1].bar(["0","1"], [vc_despues_texto.get("Masculino",0), vc_despues_texto.get("Femenino",0)])
    for i, v in enumerate([vc_despues_texto.get("Masculino",0), vc_despues_texto.get("Femenino",0)]):
        axes[1].text(i, v, str(int(v)), ha="center", va="bottom")
    axes[1].set_title("Después")
    axes[1].set_xlabel("Género (normalizado)")

    fig.suptitle("", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_fig = EDA_DIR / "04_normalizacion_genero.png"
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[FIG] {out_fig}")

# =========================
# ==== EDA ADICIONAL
# =========================
def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def eda_subconjunto_phq9():
    """EDA del subconjunto PHQ-9 (boxplots de ítems)."""
    if not RUTA_SALIDA_SUBCONJUNTO.exists():
        print("[WARN] eda_subconjunto_phq9: no existe el subconjunto; omito.")
        return

    df = pd.read_csv(RUTA_SALIDA_SUBCONJUNTO)
    df = _coerce_numeric(df, ["age", "totalphq", "categoryphq", *PHQ_ITEMS])

    # (ELIMINADO el histograma de totalphq)

    # Boxplot ítems PHQ-9
    if set(PHQ_ITEMS).issubset(df.columns):
        plt.figure(figsize=(8,4))
        datos = [df[c].dropna().values for c in PHQ_ITEMS]
        plt.boxplot(datos, showfliers=True)
        plt.xticks(range(1, 10), PHQ_ITEMS)
        plt.title("Distribución de ítems PHQ-9 (0–3)")
        plt.ylabel("Puntaje (0–3)")
        plt.tight_layout()
        out = EDA_DIR / "13_boxitems_phq9.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[FIG] {out}")

def eda_bivariado_phq9():
    """(Vacío intencionalmente: quitamos scatter/box antiguos)."""
    if not RUTA_SALIDA_SUBCONJUNTO.exists():
        print("[WARN] eda_bivariado_phq9: no existe el subconjunto; omito.")
        return
    # Sin figuras en esta etapa

def eda_multivariado_phq9():
    """Heatmap de correlaciones y figura conjunta totalphq vs ítems PHQ-9."""
    if not RUTA_SALIDA_SUBCONJUNTO.exists():
        print("[WARN] eda_multivariado_phq9: no existe el subconjunto; omito.")
        return

    # Cargamos el subconjunto (antes de normalizar género final)
    df = pd.read_csv(RUTA_SALIDA_SUBCONJUNTO)
    df = _coerce_numeric(df, ["age", "totalphq", "categoryphq", *PHQ_ITEMS])

    # =====================
    # 1. Heatmap de correlación (pearson)
    # =====================
    cols_corr = [c for c in (PHQ_ITEMS + ["totalphq"]) if c in df.columns]
    if len(cols_corr) >= 2:
        corr = df[cols_corr].corr(method="pearson")
        fig, ax = plt.subplots(figsize=(0.6*len(cols_corr)+4, 0.6*len(cols_corr)+4))
        im = ax.imshow(corr, vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label="correlación")
        ax.set_title("Correlación PHQ-9 y totalphq")
        ax.set_xticks(range(len(cols_corr))); ax.set_yticks(range(len(cols_corr)))
        ax.set_xticklabels(cols_corr, rotation=45, ha="right")
        ax.set_yticklabels(cols_corr)
        for i in range(len(cols_corr)):
            for j in range(len(cols_corr)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=8, color="black")
        fig.tight_layout()
        out = EDA_DIR / "19_heatmap_corr_phq.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"[FIG] {out}")

    # =====================
    # 2. Figura conjunta tipo strip/jitter:
    #    totalphq frente a ítems phq1..phq9, todos en una gráfica
    # =====================
    cols_necesarias = ["totalphq"] + PHQ_ITEMS
    faltantes_items = [c for c in cols_necesarias if c not in df.columns]
    if not faltantes_items:
        # Formato largo: cada fila = (item, valor_item, totalphq)
        registros = []
        for item in PHQ_ITEMS:
            sub = df[[item, "totalphq"]].dropna()
            for v_item, v_total in zip(sub[item].values, sub["totalphq"].values):
                registros.append({"item": item, "valor_item": v_item, "totalphq": v_total})

        if len(registros) == 0:
            print("[WARN] eda_multivariado_phq9: no hay datos para figura conjunta; se omite.")
        else:
            df_long = pd.DataFrame(registros)

            plt.figure(figsize=(12, 6))

            # posición base por ítem en el eje X
            items_orden = PHQ_ITEMS[:]  # ["phq1", ..., "phq9"]
            x_positions = {item: ix for ix, item in enumerate(items_orden)}

            rng = np.random.default_rng(42)  # jitter reproducible
            xs_plot = []
            ys_plot = []

            for item in items_orden:
                base_x = x_positions[item]
                datos_item = df_long[df_long["item"] == item]

                for _, fila in datos_item.iterrows():
                    # desplazamiento interno según el valor del ítem (0..3)
                    offset_por_valor = (fila["valor_item"] - 1.5) * 0.15
                    jitter_peq = rng.normal(0, 0.03)
                    x_final = base_x + offset_por_valor + jitter_peq

                    xs_plot.append(x_final)
                    ys_plot.append(fila["totalphq"])

            plt.scatter(xs_plot, ys_plot, alpha=0.6, s=20)
            plt.xticks(
                [x_positions[it] for it in items_orden],
                items_orden,
                rotation=45,
                ha="right"
            )
            plt.ylabel("totalphq")
            plt.title("totalphq frente a los ítems PHQ-9 (distribución conjunta)")

            plt.tight_layout()
            out_fig = EDA_DIR / "22_strip_total_vs_items.png"
            plt.savefig(out_fig, dpi=200)
            plt.close()
            print(f"[FIG] {out_fig}")
    else:
        print(f"[WARN] eda_multivariado_phq9: faltan columnas para figura conjunta: {faltantes_items}")

def eda_dispersion_items_vs_total():
    """
    Genera una figura 3x3 con la relación entre cada ítem del PHQ-9 (phq1..phq9)
    y la puntuación totalphq. Guarda: 21_dispersion_items_vs_total.png
    """
    if not RUTA_SALIDA_SUBCONJUNTO.exists():
        print("[WARN] eda_dispersion_items_vs_total: no existe el subconjunto; omito.")
        return

    # cargamos el subconjunto antes de normalizar género final
    df = pd.read_csv(RUTA_SALIDA_SUBCONJUNTO)
    df = _coerce_numeric(df, ["totalphq", *PHQ_ITEMS])

    # asegurarnos de que las columnas necesarias estén
    necesarios = ["totalphq"] + PHQ_ITEMS
    faltantes = [c for c in necesarios if c not in df.columns]
    if faltantes:
        print(f"[WARN] eda_dispersion_items_vs_total: faltan columnas {faltantes}; omito.")
        return

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle("", fontsize=14, fontweight="bold")

    # iterar sobre los 9 ítems y graficar contra totalphq
    for idx, item in enumerate(PHQ_ITEMS):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        ax.scatter(df[item], df["totalphq"], alpha=0.6, s=20)
        ax.set_xlabel(item)
        ax.set_ylabel("totalphq")
        ax.set_title(f"{item} vs totalphq", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # deja espacio para el título general
    out_path = EDA_DIR / "21_dispersion_items_vs_total.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[FIG] {out_path}")


# === LLAMADAS SECUENCIALES ===
if __name__ == "__main__":
    # CRUDO
    eliminar_nulos()             # Heatmap nulos (00_mapa_missing.png)

    # TRANSFORMADO 1
    eliminar_duplicados()        # Crea Test_sin_duplicados.csv (01_duplicados.png)

    # TRANSFORMADO 2 (subconjunto)
    crear_subconjunto_phq9()     # Crea phq9_subconjunto_v1.csv
    eda_seleccion_caracteristicas()  # 02_seleccion_caracteristicas.png
    eda_subconjunto_phq9()       # Boxplots ítems PHQ-9 (13_boxitems_phq9.png)
    eda_bivariado_phq9()         # (sin figuras)
    eda_multivariado_phq9()      # Heatmap + strip conjunta (19_heatmap_corr_phq.png / 22_strip_total_vs_items.png)
    eda_dispersion_items_vs_total()  # Figura 3x3 clásica (21_dispersion_items_vs_total.png)

    # TRANSFORMADO 3
    validar_rangos()             # Guarda phq9_rangos_validos.csv
    normalizar_genero()          # Guarda phq9_final.csv + 04_normalizacion_genero.png
