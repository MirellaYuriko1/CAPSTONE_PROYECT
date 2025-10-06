from urllib.parse import quote_plus

# PARA QUE GUARDE MI ML A MI BASE DE DATOS MYSQL 
import json  # <--- nuyyevo
MODEL_VERSION = "v1"  # <--- nuevo (versiona tu modelo)

#Framework web para mostrar el formulario y manejar las respuestas.
from flask import Flask, render_template, request, redirect, Response, jsonify
from io import BytesIO

#==================================================
#==========INTEGRACION MACHINE LEARNING
#Manipulación de datos 
import pandas as pd
import os

# --- ML: cargar modelo y utilidades ---
from joblib import load
from pathlib import Path

PREGUNTAS = [f"p{i}" for i in range(1, 39)]
MODEL_PATH = Path(__file__).parent / "ml" / "models" / "model_v1.joblib"
_model = None

# === HELPERS PARA LA ML
def get_model():
    """Carga el modelo una sola vez (lazy)."""
    global _model
    if _model is None:
        try:
            _model = load(MODEL_PATH)
            print(f"[ML] Modelo cargado: {MODEL_PATH}")
        except Exception as e:
            print(f"[ML] No se pudo cargar el modelo: {e}")
            _model = None
    return _model

def ml_predict_from_answers(respuestas: dict, edad: int, genero: str):
    """
    Usa el pipeline entrenado (con edad y genero).
    Retorna (pred_label, proba_dict | None)
    """
    clf = get_model()
    if clf is None:
        return None, None

    # construir un dataframe con EXACTOS nombres de columnas de entrenamiento
    row = {f"p{i}": float(respuestas.get(f"p{i}", 0)) for i in range(1, 39)}
    row["edad"] = float(edad)
    row["genero"] = str(genero or "")

    X = pd.DataFrame([row])  # el Pipeline se encarga del one-hot

    pred = clf.predict(X)[0]

    proba = None
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
        # Obtener clases del modelo dentro del Pipeline
        classes = getattr(clf, "classes_", None)
        if classes is None and hasattr(clf, "named_steps"):
            classes = clf.named_steps["model"].classes_
        proba = {c: round(float(p) * 100, 1) for c, p in zip(classes, probs)}

    return pred, proba

def _conf_label_from_pct(top_pct: float) -> str:  #PARA LA CONFIANZA PARA QUE SE MUESTRE EN MI MYSQL
    if top_pct >= 70:
        return "Alta"
    if top_pct >= 50:
        return "Media"
    return "Baja"
#========================================================

# importa tu conexión BD y reglas desde el paquete scas
from Scas.configuracion import get_db
from Scas.regla_puntuaciones import (
    DIMENSIONES, DIM_NOMBRES, PRETTY,
    interpreta_normas,
)

#----------------------------------------------
#Inicializar la app Flask
app = Flask(__name__)


#----------------------------------------------
# === 8) Rutas ===
@app.route('/')
def home():
    return render_template("index.html")

# Ruta para mostrar el formulario registro
@app.route('/form_registro')
def form_registro():
    return render_template("registro.html")

# Ruta para login
@app.route('/form_login')
def form_login():
    return render_template("login.html")


# Ruta para mostrar el formulario cuestionario
@app.route('/cuestionario')
def cuestionario():
    uid = request.args.get('uid', type=int)
    if not uid:
        return redirect('/form_login')
    # Traer el nombre del usuario para mostrarlo en el navbar
    cn = get_db()
    cur = cn.cursor()
    cur.execute("SELECT nombre,apellido FROM usuario WHERE id_usuario=%s", (uid,))
    row = cur.fetchone()
    cur.close(); cn.close()

    usuario_nombre = row[0] if row else None
    usuario_apellido = row[1] if row else None

    return render_template(
        'cuestionario.html', 
        uid=uid, 
        usuario_nombre=usuario_nombre,
        usuario_apellido=usuario_apellido)

#RUTA PARA VER EL PANEL DE ADMIN
@app.route('/form_panel')
def form_panel():
    uid = request.args.get('uid', type=int)
    q = (request.args.get('q') or '').strip()
    if not uid:
        return redirect('/form_login')

    cn = get_db()
    cur = cn.cursor(dictionary=True)
    try:
        cur.execute("SELECT nombre, rol FROM usuario WHERE id_usuario=%s", (uid,))
        admin = cur.fetchone()
        if not admin:
            return "Usuario no encontrado.", 404
        if (admin.get('rol') or '').lower() != 'admin':
            return redirect(f'/cuestionario?uid={uid}')

        where_like = ""
        params = [MODEL_VERSION]        # <-- aquí empezamos con la versión del modelo
        if q:
            where_like = " AND u.nombre LIKE %s "
            params.append(f"%{q}%")

        sql = f"""
            SELECT 
                u.id_usuario,
                u.nombre,
                c.genero,
                c.edad,
                r.puntaje_total,
                r.nivel,
                -- ML:
                pm.pred_label AS ml_label,
                pm.conf_label AS ml_conf,
                pm.conf_pct   AS ml_conf_pct,
                COALESCE(r.created_at, c.created_at) AS created_at
            FROM usuario u
            JOIN (
                SELECT c1.*
                FROM cuestionario c1
                JOIN (
                    SELECT id_usuario, MAX(created_at) AS mx
                    FROM cuestionario
                    GROUP BY id_usuario
                ) ult
                  ON ult.id_usuario = c1.id_usuario AND ult.mx = c1.created_at
            ) c ON c.id_usuario = u.id_usuario
            LEFT JOIN resultado r 
                   ON r.id_cuestionario = c.id_cuestionario
            LEFT JOIN prediccion_ml pm
                   ON pm.id_cuestionario = c.id_cuestionario
                  AND pm.model_version = %s
            WHERE u.rol = 'estudiante' {where_like}
            ORDER BY COALESCE(r.created_at, c.created_at) DESC
        """
        cur.execute(sql, params)
        rows = cur.fetchall()
    finally:
        cur.close(); cn.close()

    return render_template('panel.html',
                           admin_nombre=admin['nombre'],
                           rows=rows, uid=uid, q=q)

# Ruta para Resultado
@app.get('/resultado')
def resultado():
    uid = request.args.get('uid', type=int)
    if not uid:
        return "Falta el parámetro uid.", 400

    cn = get_db()
    cur = cn.cursor(dictionary=True)
    cur.execute(f"""
        SELECT 
               c.id_cuestionario, c.edad, c.genero, c.created_at,
               {", ".join([f"c.p{i}" for i in range(1,39)])}, 
               r.puntaje_Dim1, r.puntaje_Dim2, r.puntaje_Dim3,
               r.puntaje_Dim4, r.puntaje_Dim5, r.puntaje_Dim6,
               r.puntaje_total, r.nivel,
               u.nombre
        FROM (
            SELECT *
            FROM cuestionario
            WHERE id_usuario=%s
            ORDER BY created_at DESC
            LIMIT 1
        ) c
        JOIN usuario u        ON u.id_usuario = c.id_usuario
        LEFT JOIN resultado r ON r.id_cuestionario = c.id_cuestionario
    """, (uid,))
    row = cur.fetchone()
    cur.close(); cn.close()

    # Si no hay cuestionario o aún no hay fila en 'resultado'
    if not row or row.get('puntaje_total') is None:
        return render_template('resultado.html', notfound=True, uid=uid)

    sumas_dim = {
        "Dim1": row['puntaje_Dim1'],
        "Dim2": row['puntaje_Dim2'],
        "Dim3": row['puntaje_Dim3'],
        "Dim4": row['puntaje_Dim4'],
        "Dim5": row['puntaje_Dim5'],
        "Dim6": row['puntaje_Dim6'],
    }

    #===PARA QUE MUESTRE NIVEL Y CONFIANZA PRECISION
    # features p1..p38 para ML
    respuestas = {f"p{i}": row.get(f"p{i}") for i in range(1, 39)} #NUEVO ML#
    pred_ml, proba_ml = ml_predict_from_answers(respuestas, row['edad'], row['genero']) #NUEVO ML#
    # === Confianza del modelo (según prob. más alta) ===
    conf_ml = None
    conf_pct = None
    if proba_ml:
        top = max(proba_ml.values())# p.ej. 40.0
        conf_pct = top
        if top >= 70:
            conf_ml = "Alta"
        elif top >= 50:
            conf_ml = "Media"
        else:
            conf_ml = "Baja"
    #======================================================

    # Etiquetas por norma (para las chapitas de cada subescala)
    inter_sub, inter_total = interpreta_normas(
        row['genero'], row['edad'], sumas_dim, row['puntaje_total']
    )

    dims_order = ["Dim1","Dim2","Dim3","Dim4","Dim5","Dim6"]
    rows_view = []
    for d in dims_order:
        key = DIM_NOMBRES[d]
        rows_view.append({
            "code": key,
            "label": PRETTY[key],
            "score": sumas_dim[d],
            "level": inter_sub.get(key) or "-"
        })

    nivel_total = inter_total or row['nivel']

    return render_template(
        'resultado.html',
        notfound=False,
        uid=uid,
        nombre=row['nombre'],
        edad=row['edad'],
        rows=rows_view,
        total=row['puntaje_total'],
        nivel_total=nivel_total,
        pred_ml=pred_ml,   #AGREGADO ML
        proba_ml=proba_ml, #AGREGADO ML
        conf_ml=conf_ml,   #AGREGADO ML
        conf_pct=conf_pct  #AGREGADO ML
    )

# Ruta para que guarde el registro de usuario (GET y POST)
@app.route('/registro', methods=['GET', 'POST'])
def registro():
    if request.method == 'GET':
        return render_template("registro.html")  # sin exito/error por Jinja

    nombre   = request.form.get("nombre")
    apellido = request.form.get("apellido")
    grado    = (request.form.get("grado") or "").strip()
    edad     = (request.form.get("edad") or "").strip()          # ← solo 'edad'
    genero   = (request.form.get("genero") or "").strip().lower()
    password = request.form.get("password")

    cn = get_db(); cur = cn.cursor()
    try:
        cur.execute(
            "INSERT INTO usuario (nombre, apellido, grado, edad, genero, contraseña) VALUES (%s, %s, %s, %s, %s, %s)",
            (nombre, apellido, grado, edad, genero, password)
        )
        cn.commit()
        # 👉 redirige con querystring para que el front muestre el modal
        return redirect("/form_registro?exito=1")
    except Exception as e:
        cn.rollback()
        # 👉 redirige con error en querystring
        from urllib.parse import quote_plus
        return redirect(f"/form_registro?error={quote_plus(str(e))}")
    finally:
        cur.close(); cn.close()

# === Editar perfil (reusa registro.html en modo edición) ===
@app.route('/perfil', methods=['GET', 'POST'])
def perfil():
    if request.method == 'GET':
        uid = request.args.get('uid', type=int)
        if not uid:
            return redirect('/form_login')

        cn = get_db(); cur = cn.cursor()
        try:
            cur.execute("SELECT nombre, apellido FROM usuario WHERE id_usuario=%s", (uid,))
            row = cur.fetchone()
        finally:
            cur.close(); cn.close()

        nombre   = row[0] if row else ''
        apellido = row[1] if row and len(row) > 1 else ''

        # 👉 Redirige al mismo formulario pero en modo edición
        return redirect(
            f"/form_registro?mode=editar&uid={uid}"
            f"&nombre={quote_plus(nombre)}&apellido={quote_plus(apellido)}"
        )

    # POST -> guardar cambios (igual que ya lo tienes)
    uid = request.form.get('uid', type=int)
    if not uid:
        return redirect('/form_login')

    nombre    = (request.form.get('nombre') or '').strip()
    apellido  = (request.form.get('apellido') or '').strip()
    password  = (request.form.get('password') or '').strip()
    password2 = (request.form.get('confirm_password') or request.form.get('password2') or '').strip()

    if not nombre or not apellido:
        return redirect(f"/form_registro?mode=editar&uid={uid}&nombre={quote_plus(nombre)}&apellido={quote_plus(apellido)}&error=Nombre%20y%20apellido%20son%20obligatorios")
    if (password or password2) and password != password2:
        return redirect(f"/form_registro?mode=editar&uid={uid}&nombre={quote_plus(nombre)}&apellido={quote_plus(apellido)}&error=Las%20contrase%C3%B1as%20no%20coinciden")
    if password and not (3 <= len(password) <= 6):
        return redirect(f"/form_registro?mode=editar&uid={uid}&nombre={quote_plus(nombre)}&apellido={quote_plus(apellido)}&error=La%20contrase%C3%B1a%20debe%20tener%20entre%203%20y%206%20caracteres")

    cn = get_db(); cur = cn.cursor()
    try:
        if password:
            cur.execute(
                "UPDATE usuario SET nombre=%s, apellido=%s, contraseña=%s WHERE id_usuario=%s",
                (nombre, apellido, password, uid)
            )
        else:
            cur.execute(
                "UPDATE usuario SET nombre=%s, apellido=%s WHERE id_usuario=%s",
                (nombre, apellido, uid)
            )
        cn.commit()
    finally:
        cur.close(); cn.close()

    return redirect(f"/cuestionario?uid={uid}")

# === Login (GET/POST) ===
# IMPORTANTE: tu login.html debe postear a /login (action="/login")
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html', error=None)

    # POST: validar contra BD (sin hash)
    nombre = request.form.get('nombre')
    password = request.form.get('password')

    cn = get_db()
    cur = cn.cursor()
    try:
        cur.execute(
            "SELECT id_usuario FROM usuario WHERE nombre=%s AND contraseña=%s",
            (nombre, password)
        )
        row = cur.fetchone()
    finally:
        cur.close()
        cn.close()

    if row:
        uid = row[0]

        # >>> NUEVO: consultar rol y redirigir según sea admin o no
        cn = get_db()
        cur = cn.cursor()
        try:
            cur.execute("SELECT rol FROM usuario WHERE id_usuario=%s", (uid,))
            rol_row = cur.fetchone()
        finally:
            cur.close()
            cn.close()

        rol = (rol_row[0] if rol_row and rol_row[0] else '').lower()
        if rol == 'admin':
            return redirect(f'/form_panel?uid={uid}')              # admin -> panel
        else:
            return redirect(f'/cuestionario?uid={uid}') # estudiante/otros -> cuestionario

    # credenciales incorrectas
    return render_template('login.html', error="Nombre de usuario o contraseña incorrectos.")

#========================================
# === Guardar/Actualizar cuestionario ===
@app.post('/guardar')
def guardar():
    try:
        # Validar que venga id_usuario (oculto en el form)
        id_usuario_raw = (request.form.get("id_usuario") or "").strip()
        if not id_usuario_raw.isdigit():
            return "Falta id_usuario. Vuelve a iniciar sesión.", 400
        id_usuario = int(id_usuario_raw)

        # Datos demográficos
        edad = int(request.form.get("edad"))
        genero = request.form.get("genero")  # "Femenino" / "Masculino"

        # Respuestas p1..p38 (cada una 0..3)
        respuestas = {f"p{i}": int(request.form.get(f"p{i}", 0)) for i in range(1, 39)}

        # Sumas por dimensión y total
        sumas = {dim: sum(respuestas[f"p{i}"] for i in items) for dim, items in DIMENSIONES.items()}
        puntaje_total = sum(respuestas.values())

        # Interpretación por reglas
        inter_sub, inter_total = interpreta_normas(genero, edad, sumas, puntaje_total)
        nivel_txt = inter_total or "N/A"

        cn = get_db()
        cur = cn.cursor()

        # ¿Ya tiene cuestionario? Tomar el más reciente si existiera
        cur.execute(
            "SELECT id_cuestionario FROM cuestionario WHERE id_usuario=%s ORDER BY created_at DESC LIMIT 1",
            (id_usuario,)
        )
        row = cur.fetchone()

        if row:
            # UPDATE sobre el existente
            id_cuest = row[0]
            set_cols_p = ", ".join([f"p{i}=%s" for i in range(1, 39)])
            sql = f"""
                UPDATE cuestionario
                SET edad=%s, genero=%s, {set_cols_p}
                WHERE id_cuestionario=%s
            """
            valores = [
                edad, genero,
                *[respuestas[f"p{i}"] for i in range(1, 39)],
                id_cuest
            ]
            cur.execute(sql, valores)
        else:
            # INSERT nuevo
            columnas_p = ", ".join([f"p{i}" for i in range(1, 39)])
            placeholders_p = ", ".join(["%s"] * 38)
            sql = f"""
                INSERT INTO cuestionario
                (id_usuario, edad, genero, {columnas_p})
                VALUES (%s, %s, %s, {placeholders_p})
            """
            valores = [
                id_usuario, edad, genero,
                *[respuestas[f"p{i}"] for i in range(1, 39)],
            ]
            cur.execute(sql, valores)
            id_cuest = cur.lastrowid

        # 6) INSERT o UPDATE en 'resultado' (como NO hay UNIQUE, lo controlamos por código)
        cur.execute("SELECT id_resultado FROM resultado WHERE id_cuestionario=%s LIMIT 1", (id_cuest,))
        row_res = cur.fetchone()

        if row_res:
            # UPDATE
            cur.execute("""
                UPDATE resultado
                SET puntaje_Dim1=%s, puntaje_Dim2=%s, puntaje_Dim3=%s,
                    puntaje_Dim4=%s, puntaje_Dim5=%s, puntaje_Dim6=%s,
                    puntaje_total=%s, nivel=%s
                WHERE id_cuestionario=%s
            """, (
                sumas["Dim1"], sumas["Dim2"], sumas["Dim3"],
                sumas["Dim4"], sumas["Dim5"], sumas["Dim6"],
                puntaje_total, nivel_txt,
                id_cuest
            ))
        else:
            # INSERT
            cur.execute("""
                INSERT INTO resultado(
                  id_cuestionario, puntaje_Dim1, puntaje_Dim2, puntaje_Dim3,
                  puntaje_Dim4, puntaje_Dim5, puntaje_Dim6, puntaje_total, nivel
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                id_cuest,
                sumas["Dim1"], sumas["Dim2"], sumas["Dim3"],
                sumas["Dim4"], sumas["Dim5"], sumas["Dim6"],
                puntaje_total, nivel_txt
            ))

        # === ML: calcular y guardar/actualizar predicción del modelo en mi MYSQL===
        try:
            # usa tus mismas respuestas + edad + genero
            pred_ml, proba_ml = ml_predict_from_answers(respuestas, edad, genero)

            if pred_ml is not None:
                conf_pct = None
                conf_label = None
                proba_json = None

                if proba_ml:
                    conf_pct = float(max(proba_ml.values()))
                    conf_label = _conf_label_from_pct(conf_pct)
                    proba_json = json.dumps(proba_ml, ensure_ascii=False)

                # UPSERT: una sola predicción por (id_cuestionario, model_version)
                cur.execute("""
                    INSERT INTO prediccion_ml
                        (id_cuestionario, model_version, pred_label, conf_pct, conf_label, proba_json)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        pred_label = VALUES(pred_label),
                        conf_pct   = VALUES(conf_pct),
                        conf_label = VALUES(conf_label),
                        proba_json = VALUES(proba_json)
                """, (
                    id_cuest, MODEL_VERSION, pred_ml, conf_pct, conf_label, proba_json
                ))
        except Exception as e:
            # no rompas el flujo por un error de ML; solo lo logueas
            print(f"[ML] Error guardando predicción: {e}")

        cn.commit()
        cur.close(); cn.close()

        # 7) Redirigir al resultado
        return redirect(f"/resultado?uid={id_usuario}")

    except Exception as e:
        return f"Error al guardar: {e}", 400
        


# === 9) Run ===
if __name__ == "__main__":
    # Solo local
    app.run(host="127.0.0.1", port=5000, debug=True)  # pon debug=False si no quieres recarga/tracebacks
