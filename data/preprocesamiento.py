# ml/preprocesamiento.py
from __future__ import annotations
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# =========================
# Rutas (no cambies nombres de salida)
# =========================
RUTA_CRUDO = Path("data/crudo/Test_inicial.xlsx")

# Intermedios explícitos de la fase de limpieza
RUTA_SALIDA_SIN_NULOS       = Path("data/transformado/Test_sin_nulos.csv")
RUTA_SALIDA_SIN_DUPLICADOS  = Path("data/transformado/Test_sin_duplicados.csv")
RUTA_SALIDA_NORMALIZADO     = Path("data/transformado/Test_normalizado.csv")
RUTA_SEL_CARAC              = Path("data/transformado/seleccion_caracteristicas.csv")
RUTA_SALIDA_FINAL           = Path("data/final/phq9_final.csv")  # <- final oficial

# =========================
# Config EDA
# =========================
EDA_DIR = Path("data/analisis EDA")
EDA_DIR.mkdir(parents=True, exist_ok=True)

# Ítems canónicos (tras renombrar)
PHQ_ITEMS = [f"phq{i}" for i in range(1, 10)]

# Conjunto FINAL que quieres conservar (orden explícito)
KEEP_COLS = ["age", "genero_bin", *PHQ_ITEMS, "nivel_idx"]

# Mapeo español -> canónico
MAP_CRUDO = {
    "edad": "age",
    "genero": "gender",
    "p1": "phq1", "p2": "phq2", "p3": "phq3", "p4": "phq4", "p5": "phq5",
    "p6": "phq6", "p7": "phq7", "p8": "phq8", "p9": "phq9",
    "puntaje_total": "puntaje_total",
    "nivel": "nivel",
    "grado": "grado",
}

# Columnas clave esperadas
COLS_REQUERIDAS = ["age", "gender", *PHQ_ITEMS, "puntaje_total", "nivel"]

# Orden de severidad (para nivel_idx)
CLASES_ORDEN = ["Mínimo", "Leve", "Moderado", "Moderadamente grave", "Grave"]

# =========================
# Utilidades
# =========================
def _renombrar_a_canonico(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: MAP_CRUDO[c] for c in df.columns if c in MAP_CRUDO})

def _canon_str(s: str) -> str:
    s = (s or "").strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s

def _estandariza_nulos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj = df.select_dtypes(include="object").columns
    for c in obj:
        s = df[c].astype(str)
        s = (s.str.replace("\u00a0", " ", regex=False)
               .str.replace("\u200b", "", regex=False)
               .str.strip())
        df[c] = s
    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    df.replace(to_replace=r"(?i)^(nan|null|na|n/?a|none)$", value=np.nan, regex=True, inplace=True)
    df.replace(to_replace=r"^(?:-+|—)$", value=np.nan, regex=True, inplace=True)
    return df

def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _save_bar(labels, values, title, filename, xlabel="", ylabel="Conteo"):
    plt.figure(figsize=(6,4))
    plt.bar(labels, values)
    for i, v in enumerate(values):
        try:
            txt = str(int(v))
        except Exception:
            txt = f"{v}"
        plt.text(i, v, txt, ha="center", va="bottom")
    if title:
        plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    out = EDA_DIR / filename
    plt.savefig(out, dpi=200); plt.close()
    print(f"[FIG] {out}")

# =========================
# A) EDA inicial + eliminación TEMPRANA de nulos
# =========================
def eda_mapa_faltantes_y_drop(df_in: pd.DataFrame) -> pd.DataFrame:
    df_norm = _estandariza_nulos(df_in)

    present = df_norm.notna().values.astype(int)
    ancho = max(10, 0.45 * len(df_norm.columns))
    fig, ax = plt.subplots(figsize=(ancho, 6))
    cmap = ListedColormap(["#ffffff", "#6a51a3"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.imshow(present, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_title("Mapa de valores faltantes", fontsize=16, weight="bold")
    ax.set_xlabel("Columnas"); ax.set_ylabel("Filas")
    ax.set_xticks(range(len(df_norm.columns)))
    ax.set_xticklabels(df_norm.columns.astype(str), rotation=90, ha="right", fontsize=9)
    fig.tight_layout()
    out_fig = EDA_DIR / "00_mapa_missing.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[FIG] {out_fig} (morado = dato, blanco = vacío)")

    if not set(COLS_REQUERIDAS).issubset(df_norm.columns):
        falt = [c for c in COLS_REQUERIDAS if c not in df_norm.columns]
        raise ValueError(f"Faltan columnas clave para validar nulos tempranos: {falt}")

    print("[EARLY NULOS] NaN por columna requerida:")
    print(df_norm[COLS_REQUERIDAS].isna().sum().to_string())

    mask_notnull_req = df_norm[COLS_REQUERIDAS].notna().all(axis=1)
    n_eliminadas = int((~mask_notnull_req).sum())
    print(f"[EARLY NULOS] Filas eliminadas por nulos (temprano): {n_eliminadas}")

    df_clean = df_norm.loc[mask_notnull_req].copy()
    RUTA_SALIDA_SIN_NULOS.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(RUTA_SALIDA_SIN_NULOS, index=False, encoding="utf-8")
    print(f"[OK] Guardado sin nulos -> {RUTA_SALIDA_SIN_NULOS}")
    return df_clean

# =========================
# EDA: Distribuciones y Boxplots
# =========================
def eda_distribuciones(df_can: pd.DataFrame):
    df = _coerce_numeric(df_can, ["age", "puntaje_total", *PHQ_ITEMS])

    if "age" in df.columns:
        edades = df["age"].dropna().astype(int)
        plt.figure(figsize=(7, 4))
        bins_edad = np.arange(11.5, 18.5 + 1, 1)
        plt.hist(edades, bins=bins_edad, edgecolor="black", linewidth=0.6, alpha=0.9)
        plt.title("Distribución de las edades")
        plt.xlabel("Edades"); plt.ylabel("Frecuencia")
        plt.xticks(np.arange(12, 19, 1)); plt.xlim(12, 18)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        out = EDA_DIR / "11_hist_edad.png"
        plt.savefig(out, dpi=300); plt.close(); print(f"[FIG] {out}")

    if "puntaje_total" in df.columns:
        punt = df["puntaje_total"].dropna().astype(int)
        plt.figure(figsize=(7, 4))
        bins_punt = np.arange(-0.5, 27.5 + 1, 1)
        plt.hist(punt, bins=bins_punt, edgecolor="black", linewidth=0.6, alpha=0.9)
        plt.title("Distribución de los puntajes totales")
        plt.xlabel("Puntaje"); plt.ylabel("Frecuencia")
        plt.xticks(np.arange(0, 28, 3)); plt.xlim(0, 27)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        out = EDA_DIR / "12_hist_puntaje_total.png"
        plt.savefig(out, dpi=300); plt.close(); print(f"[FIG] {out}")

    # --- Barras: NIVELES (SIN normalizar etiquetas) ---
    if "nivel" in df_can.columns:
        niveles_raw = df_can["nivel"].dropna().astype(str)
        conteo = niveles_raw.value_counts()

        # Si tus etiquetas coinciden con CLASES_ORDEN, respeta ese orden; si no, usa el natural
        if set(conteo.index).issubset(set(CLASES_ORDEN)):
            conteo = conteo.reindex([c for c in CLASES_ORDEN if c in conteo.index])

        plt.figure(figsize=(7, 4))
        plt.bar(conteo.index, conteo.values, edgecolor="black", linewidth=0.6, alpha=0.9)
        for i, v in enumerate(conteo.values):
            plt.text(i, v + 1, str(int(v)), ha="center", va="bottom", fontsize=9)
        plt.title("Distribución de los niveles de depresión")
        plt.xlabel("Niveles de depresión"); plt.ylabel("Frecuencia")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        out = EDA_DIR / "13_hist_niveles_depresion.png"
        plt.savefig(out, dpi=300); plt.close(); print(f"[FIG] {out}")

        # Info en consola
        total_df = len(df_can)
        total_conteo = int(conteo.sum())
        nan_nivel = total_df - len(niveles_raw)  # filas donde 'nivel' es NaN
        print("[INFO] Frecuencia de niveles (sin normalizar):")
        print(conteo.to_string())
        print(f"[INFO] Total filas DF: {total_df} | Contadas en gráfico: {total_conteo} | NaN en 'nivel': {nan_nivel}")

        

def eda_boxplots_items(df_can: pd.DataFrame,
                       items: list[str] = PHQ_ITEMS,
                       outname: str = "13_boxitems_phq9.png"):
    if not set(items).issubset(df_can.columns):
        falt = [c for c in items if c not in df_can.columns]
        print(f"[BOXPLOTS] Omitido. Faltan columnas: {falt}")
        return
    df_num = df_can.copy()
    for c in items:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    datos = [df_num[c].dropna().values for c in items]
    plt.figure(figsize=(9, 5))
    plt.boxplot(datos, showfliers=True)
    plt.xticks(range(1, len(items) + 1), items)
    plt.ylim(-0.1, 3.1)
    plt.ylabel("Puntaje"); plt.title("Distribución de puntajes por ítem")
    plt.tight_layout()
    out = EDA_DIR / outname
    plt.savefig(out, dpi=300); plt.close()
    print(f"[FIG] {out}")

# =========================
# EDA: Correlaciones (Pearson)
# =========================
def _plot_corr_heatmap(corr: pd.DataFrame, title: str, outfile: Path):
    labels = corr.columns.tolist()
    n = len(labels)
    fig_w = max(8, 0.6 * n + 4)
    fig_h = max(6, 0.6 * n + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    cbar = plt.colorbar(im, ax=ax, label="")
    cbar.ax.tick_params(labelsize=14)

    ax.set_title(title, fontsize=21, weight="bold")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(labels, fontsize=14)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=13,weight="500")
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] {outfile}")

def eda_correlaciones_pearson(df_can: pd.DataFrame, thr_abs: float = 0.8):
    df = df_can.copy()
    num_cols = ["age", *PHQ_ITEMS]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    cols_pred = [c for c in ["age", "genero_bin", "grado_ordinal", *PHQ_ITEMS] if c in df.columns]
    cols_full = cols_pred + [c for c in ["puntaje_total", "nivel_idx"] if c in df.columns]

    if len(cols_full) >= 2:
        corr_full = df[cols_full].corr(method="pearson")
        _plot_corr_heatmap(corr_full,
                           "Matriz de correlación de las variables del estudio",
                           EDA_DIR / "20_corr_full_pearson.png")

    if len(cols_pred) >= 2:
        corr_pred = df[cols_pred].corr(method="pearson")
        abs_corr = corr_pred.abs()
        pairs = []
        for i in range(len(cols_pred)):
            for j in range(i+1, len(cols_pred)):
                r = abs_corr.iloc[i, j]
                if pd.notna(r) and r >= thr_abs:
                    pairs.append({"var_1": cols_pred[i],
                                  "var_2": cols_pred[j],
                                  "abs_r": float(r),
                                  "r": float(corr_pred.iloc[i, j])})
        pd.DataFrame(pairs).sort_values("abs_r", ascending=False)\
            .to_csv(EDA_DIR / "21_corr_pairs_ge08.csv", index=False, encoding="utf-8")
        print(f"[OK] Pares con |r| >= {thr_abs:.2f} -> {EDA_DIR / '21_corr_pairs_ge08.csv'}")

# =========================
# B) Limpieza informada por EDA
# =========================
def eliminar_duplicados(df_in: pd.DataFrame) -> pd.DataFrame:
    duplicados_mask = df_in.duplicated(keep=False)
    dups_detalle = df_in.loc[duplicados_mask].copy()
    if not dups_detalle.empty:
        dups_detalle.to_csv(EDA_DIR / "duplicados_detalle_full.csv", index=False, encoding="utf-8")
        print(f"[CHECK] Detalle de duplicados -> {EDA_DIR / 'duplicados_detalle_full.csv'}")

    duplicados = int(df_in.duplicated().sum())
    unicos = len(df_in) - duplicados
    _save_bar(["Duplicados", "Únicos"], [duplicados, unicos], "", "01_duplicados.png")

    df_sin_dup = df_in.drop_duplicates().copy()
    RUTA_SALIDA_SIN_DUPLICADOS.parent.mkdir(parents=True, exist_ok=True)
    df_sin_dup.to_csv(RUTA_SALIDA_SIN_DUPLICADOS, index=False, encoding="utf-8")
    print(f"[OK] Guardado sin duplicados: {RUTA_SALIDA_SIN_DUPLICADOS}")
    return df_sin_dup

# =========================
# C) Normalización ligera (sin escalado)
# =========================
def normalizar_genero(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    g = df["gender"].astype(str).str.strip().str.lower()
    df["gender"] = g.replace({
        "male":"Masculino","masculino":"Masculino",
        "female":"Femenino","femenino":"Femenino"
    })
    df["genero_bin"] = df["gender"].map({"Masculino":0, "Femenino":1})
    return df

def normalizar_grado(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    def _to_ord(x: str):
        if pd.isna(x): return np.nan
        s = _canon_str(str(x))
        s = (s.replace("primero","1").replace("segundo","2").replace("tercero","3")
               .replace("cuarto","4").replace("quinto","5").replace("sexto","6"))
        m = re.search(r"([1-6])", s)
        return int(m.group(1)) if m else np.nan
    if "grado" in df.columns:
        df["grado_ordinal"] = df["grado"].map(_to_ord)
    return df

def normalizar_nivel(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    def _canon_nivel(x: str) -> str:
        s = _canon_str(x)
        m = {
            "minimo":"Mínimo", "mínimo":"Mínimo",
            "leve":"Leve",
            "moderado":"Moderado",
            "moderadamente grave":"Moderadamente grave",
            "moderadamente severa":"Moderadamente grave",
            "moderadamente severo":"Moderadamente grave",
            "moderadamente_grave":"Moderadamente grave",
            "grave":"Grave",
        }
        return m.get(s, x)
    df["nivel"] = df["nivel"].astype(str).map(_canon_nivel)
    mapa_idx = {n:i+1 for i,n in enumerate(CLASES_ORDEN)}
    df["nivel_idx"] = df["nivel"].map(mapa_idx)
    if df["nivel_idx"].isna().any():
        desconocidos = sorted(df.loc[df["nivel_idx"].isna(), "nivel"].astype(str).unique())
        raise ValueError(f"Valores de 'nivel' no reconocidos: {desconocidos}.")
    df["nivel_idx"] = df["nivel_idx"].astype(int)
    return df

# =========================
# === PIPELINE (hasta normalizado + correlaciones + selección estricta)
# =========================
if __name__ == "__main__":
    if not RUTA_CRUDO.exists():
        raise FileNotFoundError(f"No encuentro el archivo crudo: {RUTA_CRUDO}")

    # 0) Leer crudo y renombrar a canónico
    df_raw = pd.read_excel(RUTA_CRUDO)
    print(f"[CRUDO] Filas: {len(df_raw)} | Columnas: {len(df_raw.columns)}")
    df_can0 = _renombrar_a_canonico(df_raw)

    # 1) Missing + drop (guarda Test_sin_nulos.csv)
    df1 = eda_mapa_faltantes_y_drop(df_can0)

    # 2) EDA esencial
    eda_distribuciones(df1)
    eda_boxplots_items(df1)

    # 3) Duplicados (guarda Test_sin_duplicados.csv)
    df2 = eliminar_duplicados(df1)

    # 4) Normalización ligera (sin escalado): gender, grado, nivel
    df3 = normalizar_genero(df2)
    df4 = normalizar_grado(df3)
    df5 = normalizar_nivel(df4)

    # 5) Correlaciones (figuras + pares |r|>=0.8) — informativo
    eda_correlaciones_pearson(df5, thr_abs=0.8)

    # 6) Guardar dataset normalizado (trazabilidad)
    RUTA_SALIDA_NORMALIZADO.parent.mkdir(parents=True, exist_ok=True)
    df5.to_csv(RUTA_SALIDA_NORMALIZADO, index=False, encoding="utf-8")
    print(f"[OK] Test_normalizado -> {RUTA_SALIDA_NORMALIZADO}")

    # 7) SELECCIÓN ESTRICTA: solo las 12 columnas finales
    keep_cols = [c for c in KEEP_COLS if c in df5.columns]
    df_sel = df5[keep_cols].copy()

    # Guardar en transformado y final
    RUTA_SEL_CARAC.parent.mkdir(parents=True, exist_ok=True)
    df_sel.to_csv(RUTA_SEL_CARAC, index=False, encoding="utf-8")
    RUTA_SALIDA_FINAL.parent.mkdir(parents=True, exist_ok=True)
    df_sel.to_csv(RUTA_SALIDA_FINAL, index=False, encoding="utf-8")

    print(f"[OK] seleccion_caracteristicas -> {RUTA_SEL_CARAC}")
    print(f"[OK] phq9_final -> {RUTA_SALIDA_FINAL}")
    print(f"[INFO] Columnas finales ({len(df_sel.columns)}): {list(df_sel.columns)}")
