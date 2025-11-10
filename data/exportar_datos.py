# datos_iniciales.py
# Exporta datos unidos (usuario + cuestionario + resultado) a data/crudo/Test_inicial.xlsx
from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv
import mysql.connector

# --- CONFIGURACIÓN ---
LATEST_ONLY = False   # False = histórico; True = solo el último cuestionario por usuario
OUT_DIR = Path("data/crudo")
OUT_XLSX = OUT_DIR / "Test_inicial.xlsx"
SHEET_NAME = "datos"

# Cargar variables de entorno (.env)
load_dotenv(override=True)
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

def get_conn():
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME
    )

# ============== SQL (SIN IDs y SIN created_at) ==============

# Histórico
SQL_HIST = """
SELECT
    u.nombre,
    u.apellido,
    u.grado,
    u.edad,
    u.genero,
    u.rol,
    c.p1, c.p2, c.p3, c.p4, c.p5, c.p6, c.p7, c.p8, c.p9,
    r.puntaje_total,
    r.nivel
FROM usuario u
JOIN cuestionario c
  ON c.id_usuario = u.id_usuario
LEFT JOIN resultado r
  ON r.id_cuestionario = c.id_cuestionario
ORDER BY u.id_usuario ASC, c.created_at DESC;
"""

# Último cuestionario
SQL_LAST = """
WITH ult AS (
  SELECT id_usuario, MAX(created_at) AS mx
  FROM cuestionario
  GROUP BY id_usuario
)
SELECT
    u.nombre,
    u.apellido,
    u.grado,
    u.edad,
    u.genero,
    u.rol,
    c.p1, c.p2, c.p3, c.p4, c.p5, c.p6, c.p7, c.p8, c.p9,
    r.puntaje_total,
    r.nivel
FROM usuario u
JOIN ult
  ON ult.id_usuario = u.id_usuario
JOIN cuestionario c
  ON c.id_usuario = u.id_usuario AND c.created_at = ult.mx
LEFT JOIN resultado r
  ON r.id_cuestionario = c.id_cuestionario
ORDER BY u.id_usuario ASC, c.created_at DESC;
"""

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sql = SQL_LAST if LATEST_ONLY else SQL_HIST
    print(f"[INFO] Conectando a MySQL {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME} ...")
    cn = get_conn()
    try:
        df = pd.read_sql(sql, cn)
    finally:
        cn.close()

    # Guardar Excel limpio (sin columnas ID)
    df.to_excel(OUT_XLSX, sheet_name=SHEET_NAME, index=False, engine="openpyxl")
    print(f"[OK] Exportado {len(df):,} filas a: {OUT_XLSX}")

if __name__ == "__main__":
    main()
