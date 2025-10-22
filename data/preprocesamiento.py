# ml/preprocesamiento.py
import pandas as pd
from pathlib import Path


RUTA_CRUDO = Path("data/crudo/Test_inicial.xlsx")
RUTA_SALIDA_SIN_DUPLICADOS = Path("data/transformado/Test_sin_duplicados.csv")

def eliminar_duplicados():
    if not RUTA_CRUDO.exists():
        raise FileNotFoundError(f"No encuentro el archivo crudo en: {RUTA_CRUDO}")

    df = pd.read_excel(RUTA_CRUDO)
    
    duplicados = df.duplicated().sum()
    print(f"Duplicados detectados: {duplicados}")

    df_sin_dup = df.drop_duplicates()
    print(f"Filas después de eliminar duplicados: {len(df_sin_dup)}")

    RUTA_SALIDA_SIN_DUPLICADOS.parent.mkdir(parents=True, exist_ok=True)
    df_sin_dup.to_csv(RUTA_SALIDA_SIN_DUPLICADOS, index=False, encoding="utf-8")

    print(f"[OK] Archivo sin duplicados guardado en: {RUTA_SALIDA_SIN_DUPLICADOS}")







# === NUEVO BLOQUE: CREAR SUBCONJUNTO PHQ-9 ===
RUTA_SIN_DUPLICADOS = Path("data/transformado/Test_sin_duplicados.csv")
RUTA_SALIDA_SUBCONJUNTO = Path("data/transformado/phq9_subconjunto_v1.csv")

COLUMNAS_UTILIZADAS = [
    "age", "gender",
    "phq1", "phq2", "phq3", "phq4", "phq5", "phq6", "phq7", "phq8", "phq9",
    "totalphq", "categoryphq"
]

def crear_subconjunto_phq9():
    if not RUTA_SIN_DUPLICADOS.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {RUTA_SIN_DUPLICADOS}")

    df = pd.read_csv(RUTA_SIN_DUPLICADOS)
    print(f"\n[INFO] Filas después de eliminar duplicados: {len(df)}")

    faltantes = [c for c in COLUMNAS_UTILIZADAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el archivo: {faltantes}")

    df_sub = df[COLUMNAS_UTILIZADAS].copy()

    RUTA_SALIDA_SUBCONJUNTO.parent.mkdir(parents=True, exist_ok=True)
    df_sub.to_csv(RUTA_SALIDA_SUBCONJUNTO, index=False, encoding="utf-8")

    print(f"[OK] Subconjunto PHQ-9 creado correctamente.")
    print(f" - Archivo: {RUTA_SALIDA_SUBCONJUNTO}")
    print(f" - Filas: {len(df_sub)} | Columnas: {len(df_sub.columns)}")
    print(f" - Columnas: {list(df_sub.columns)}")





# === NUEVO BLOQUE: VALIDAR RANGOS ===
RUTA_SUBCONJUNTO = Path("data/transformado/phq9_subconjunto_v1.csv")
RUTA_SALIDA_RANGOS_VALIDOS = Path("data/transformado/phq9_rangos_validos.csv")

def validar_rangos():
    if not RUTA_SUBCONJUNTO.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {RUTA_SUBCONJUNTO}")

    df = pd.read_csv(RUTA_SUBCONJUNTO)
    print(f"\n[INFO] Filas de entrada: {len(df)}")

    phq_cols = ["phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"]
    mask_age = df["age"].between(12, 19)
    mask_phq = df[phq_cols].apply(lambda s: s.between(0, 3)).all(axis=1)
    mask_total = df["totalphq"].between(0, 27)
    mask_cat = df["categoryphq"].between(1, 5)

    print(f"[CHK] Fuera de rango -> edad: {(~mask_age).sum()} | phq: {(~mask_phq).sum()} | totalphq: {(~mask_total).sum()} | categoría: {(~mask_cat).sum()}")

    mask_final = mask_age & mask_phq & mask_total & mask_cat
    df_valid = df[mask_final].copy()
    print(f"[OK] Filas válidas: {len(df_valid)} (eliminadas: {len(df) - len(df_valid)})")

    RUTA_SALIDA_RANGOS_VALIDOS.parent.mkdir(parents=True, exist_ok=True)
    df_valid.to_csv(RUTA_SALIDA_RANGOS_VALIDOS, index=False, encoding="utf-8")
    print(f"[OK] Archivo validado guardado en: {RUTA_SALIDA_RANGOS_VALIDOS}")







# === BLOQUE FINAL: NORMALIZAR GÉNERO ===
RUTA_RANGOS_VALIDOS = Path("data/transformado/phq9_rangos_validos.csv")
RUTA_SALIDA_FINAL = Path("data/final/phq9_final.csv")

def normalizar_genero():
    if not RUTA_RANGOS_VALIDOS.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {RUTA_RANGOS_VALIDOS}")
    df = pd.read_csv(RUTA_RANGOS_VALIDOS)
    print(f"\n[INFO] Filas recibidas para normalización de género: {len(df)}")

    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"male": "Masculino","masculino": "Masculino",
                  "female": "Femenino","femenino": "Femenino"})
    )
    df["genero_bin"] = df["gender"].map({
        "Masculino": 0,
        "Femenino": 1
    })
    print(f"[CHK] Valores únicos en 'gender': {df['gender'].unique().tolist()}")
    RUTA_SALIDA_FINAL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUTA_SALIDA_FINAL, index=False, encoding="utf-8")
    print(f"[OK] Dataset final guardado en: {RUTA_SALIDA_FINAL}")
    print(f"[OK] Columnas finales: {list(df.columns)}")
    
# === LLAMADAS SECUENCIALES ===
if __name__ == "__main__":
    eliminar_duplicados()      
    crear_subconjunto_phq9()   
    validar_rangos()            
    normalizar_genero()        