from dotenv import load_dotenv
load_dotenv(override=True)

from urllib.parse import quote_plus

# PARA QUE GUARDE MI ML A MI BASE DE DATOS MYSQL 
import json  # <--- nuevo
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

PREGUNTAS = [f"p{i}" for i in range(1, 10)]
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
    clf = get_model()
    if clf is None:
        return None, None

    row = {f"p{i}": float(respuestas.get(f"p{i}", 0)) for i in range(1, 10)}
    row["edad"] = float(edad)
    # 👇 Normaliza exactamente como en el entrenamiento
    row["genero"] = (str(genero or "").strip().lower())

    X = pd.DataFrame([row])
    pred = clf.predict(X)[0]

    proba = None
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
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

# importa tu conexión BD 
from Scas.configuracion import get_db

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

    # --- helper para clasificar PHQ-A con tus puntos de corte ---
    def interpreta_phqa(total: int) -> str:
        if 0 <= total <= 4:   return "Mínimo"
        if 5 <= total <= 9:   return "Leve"
        if 10 <= total <= 14: return "Moderado"
        if 15 <= total <= 19: return "Moderadamente grave"
        return "Grave"  # 20–27

    cn = get_db()
    cur = cn.cursor(dictionary=True)

    # Trae el último cuestionario del usuario (solo p1..p9)
    cur.execute("""
        SELECT 
               c.id_cuestionario,
               c.created_at,
               c.p1, c.p2, c.p3, c.p4, c.p5, c.p6, c.p7, c.p8, c.p9,
               r.puntaje_total, r.nivel,
               u.nombre, u.edad, u.genero
          FROM (
                SELECT *
                  FROM cuestionario
                 WHERE id_usuario=%s
                 ORDER BY created_at DESC
                 LIMIT 1
          ) c
          JOIN usuario u        ON u.id_usuario = %s
     LEFT JOIN resultado r      ON r.id_cuestionario = c.id_cuestionario
    """, (uid, uid))
    row = cur.fetchone()
    cur.close(); cn.close()

    if not row:
        return render_template('resultado.html', notfound=True, uid=uid)

    # Si aún no hay fila en 'resultado', calcula aquí (solo p1..p9)
    total = row.get('puntaje_total')
    nivel = row.get('nivel')
    if total is None:
        total = sum(int(row.get(f"p{i}", 0) or 0) for i in range(1, 10))
        nivel = interpreta_phqa(total)

    # ====== PREDICCIÓN ML SOLO PARA MOSTRAR (NO SE GUARDA) ======
    respuestas = {f"p{i}": int(row.get(f"p{i}", 0) or 0) for i in range(1, 10)}
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
    # ============================================================

    return render_template(
        'resultado.html',
        notfound=False,
        uid=uid,
        nombre=row.get('nombre'),
        edad=row.get('edad'),
        total=total,
        nivel_total=nivel,
        # compat:
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
            # ⬇️ trae también grado, edad y genero
            cur.execute(
                "SELECT nombre, apellido, grado, edad, genero FROM usuario WHERE id_usuario=%s",
                (uid,)
            )
            row = cur.fetchone()
        finally:
            cur.close(); cn.close()

        nombre   = row[0] if row else ''
        apellido = row[1] if row and len(row) > 1 else ''
        grado    = row[2] if row and len(row) > 2 else ''
        edad     = row[3] if row and len(row) > 3 else ''
        genero   = row[4] if row and len(row) > 4 else ''

        # 👉 Redirige al mismo formulario pero en modo edición (pasa los nuevos campos)
        return redirect(
            f"/form_registro?mode=editar&uid={uid}"
            f"&nombre={quote_plus(str(nombre))}"
            f"&apellido={quote_plus(str(apellido))}"
            f"&grado={quote_plus(str(grado))}"
            f"&edad={quote_plus(str(edad))}"
            f"&genero={quote_plus(str(genero))}"
        )

    # POST -> guardar cambios
    uid = request.form.get('uid', type=int)
    if not uid:
        return redirect('/form_login')

    nombre    = (request.form.get('nombre') or '').strip()
    apellido  = (request.form.get('apellido') or '').strip()
    grado     = (request.form.get('grado') or '').strip()
    edad_str  = (request.form.get('edad') or '').strip()
    genero    = (request.form.get('genero') or '').strip().lower()  # DB espera 'femenino'/'masculino'
    password  = (request.form.get('password') or '').strip()
    password2 = (request.form.get('confirm_password') or request.form.get('password2') or '').strip()

    # Si quieres, puedes validar edad a entero (la columna es INT NOT NULL)
    try:
        edad = int(edad_str) if edad_str != '' else None
    except ValueError:
        edad = None

    if not nombre or not apellido:
        return redirect(
            f"/form_registro?mode=editar&uid={uid}"
            f"&nombre={quote_plus(nombre)}&apellido={quote_plus(apellido)}"
            f"&error=Nombre%20y%20apellido%20son%20obligatorios"
        )
    if (password or password2) and password != password2:
        return redirect(
            f"/form_registro?mode=editar&uid={uid}"
            f"&nombre={quote_plus(nombre)}&apellido={quote_plus(apellido)}"
            f"&error=Las%20contrase%C3%B1as%20no%20coinciden"
        )
    if password and not (3 <= len(password) <= 6):
        return redirect(
            f"/form_registro?mode=editar&uid={uid}"
            f"&nombre={quote_plus(nombre)}&apellido={quote_plus(apellido)}"
            f"&error=La%20contrase%C3%B1a%20debe%20tener%20entre%203%20y%206%20caracteres"
        )

    cn = get_db(); cur = cn.cursor()
    try:
        if password:
            cur.execute(
                "UPDATE usuario "
                "SET nombre=%s, apellido=%s, grado=%s, edad=%s, genero=%s, contraseña=%s "
                "WHERE id_usuario=%s",
                (nombre, apellido, grado, edad, genero, password, uid)
            )
        else:
            cur.execute(
                "UPDATE usuario "
                "SET nombre=%s, apellido=%s, grado=%s, edad=%s, genero=%s "
                "WHERE id_usuario=%s",
                (nombre, apellido, grado, edad, genero, uid)
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
#========================================
# === Guardar/Actualizar cuestionario (PHQ-A/PHQ-9: 9 ítems) ===
@app.post('/guardar')
def guardar():
    try:
        # 1) id_usuario obligado
        id_usuario_raw = (request.form.get("id_usuario") or "").strip()
        if not id_usuario_raw.isdigit():
            return "Falta id_usuario. Vuelve a iniciar sesión.", 400
        id_usuario = int(id_usuario_raw)

        # 2) Respuestas PHQ-A p1..p9 (0..3)
        respuestas = {f"p{i}": int(request.form.get(f"p{i}", 0)) for i in range(1, 10)}
        puntaje_total = sum(respuestas.values())

        # 3) Nivel PHQ-A (0–4, 5–9, 10–14, 15–19, 20–27)
        def interpreta_phqa(total: int) -> str:
            if 0 <= total <= 4:   return "Mínimo"
            if 5 <= total <= 9:   return "Leve"
            if 10 <= total <= 14: return "Moderado"
            if 15 <= total <= 19: return "Moderadamente grave"
            return "Grave"  # 20–27

        nivel_txt = interpreta_phqa(puntaje_total)

        cn = get_db()
        cur = cn.cursor()

        # 4) ¿Tiene cuestionario previo? (tomar el último)
        cur.execute(
            "SELECT id_cuestionario FROM cuestionario WHERE id_usuario=%s ORDER BY created_at DESC LIMIT 1",
            (id_usuario,)
        )
        row = cur.fetchone()

        if row:
            # UPDATE del último SOLO con p1..p9
            id_cuest = row[0]
            sql = """
                UPDATE cuestionario
                   SET 
                       p1=%s,p2=%s,p3=%s,p4=%s,p5=%s,p6=%s,p7=%s,p8=%s,p9=%s
                 WHERE id_cuestionario=%s
            """
            valores = [
                respuestas["p1"],respuestas["p2"],respuestas["p3"],
                respuestas["p4"],respuestas["p5"],respuestas["p6"],
                respuestas["p7"],respuestas["p8"],respuestas["p9"],
                id_cuest
            ]
            cur.execute(sql, valores)
        else:
            # INSERT nuevo SOLO con p1..p9
            sql = """
                INSERT INTO cuestionario
                    (id_usuario,p1,p2,p3,p4,p5,p6,p7,p8,p9)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """
            valores = [
                id_usuario, 
                respuestas["p1"],respuestas["p2"],respuestas["p3"],
                respuestas["p4"],respuestas["p5"],respuestas["p6"],
                respuestas["p7"],respuestas["p8"],respuestas["p9"],
            ]
            cur.execute(sql, valores)
            id_cuest = cur.lastrowid

        # 5) UPSERT en resultado (solo total + nivel)
        cur.execute("SELECT id_resultado FROM resultado WHERE id_cuestionario=%s LIMIT 1", (id_cuest,))
        row_res = cur.fetchone()

        if row_res:
            cur.execute(
                "UPDATE resultado SET puntaje_total=%s, nivel=%s WHERE id_cuestionario=%s",
                (puntaje_total, nivel_txt, id_cuest)
            )
        else:
            cur.execute(
                "INSERT INTO resultado (id_cuestionario, puntaje_total, nivel) VALUES (%s,%s,%s)",
                (id_cuest, puntaje_total, nivel_txt)
            )

        # === ML: calcular y guardar/actualizar predicción del modelo en mi MYSQL===
        try:
            # Traer edad y genero del usuario
            cur.execute("SELECT edad, genero FROM usuario WHERE id_usuario=%s", (id_usuario,))
            urow = cur.fetchone()
            edad = urow[0] if urow else None
            genero = urow[1] if urow else None

            pred_ml, proba_ml = ml_predict_from_answers(respuestas, edad, genero)

            # --- Normalizar al ENUM EXACTO de la BD ---
            label_map = {
                0: "Mínimo", 1: "Leve", 2: "Moderado",
                3: "Moderadamente grave", 4: "Grave",
                "minimo": "Mínimo", "mínimo": "Mínimo",
                "leve": "Leve",
                "moderado": "Moderado",
                "moderadamente grave": "Moderadamente grave",
                "grave": "Grave",
            }
            def canon_label(y):
                if isinstance(y, (int, float)) and int(y) in label_map:
                    return label_map[int(y)]
                s = str(y).strip().lower()
                return label_map.get(s)

            pred_ml_canon = canon_label(pred_ml)
            if not pred_ml_canon:
                raise ValueError(f"Clase del modelo no mapea al ENUM: {pred_ml!r}")

            # Confianza
            conf_pct = float(max(proba_ml.values())) if proba_ml else None  # ya viene en %
            conf_label = _conf_label_from_pct(conf_pct) if conf_pct is not None else None
            proba_json = json.dumps(proba_ml, ensure_ascii=False) if proba_ml else None

            # UPSERT (recomendado crear índice único ver nota abajo)
            cur.execute("""
                INSERT INTO prediccion_ml
                    (id_cuestionario, model_version, pred_label, conf_pct, conf_label, proba_json)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    pred_label = VALUES(pred_label),
                    conf_pct   = VALUES(conf_pct),
                    conf_label = VALUES(conf_label),
                    proba_json = VALUES(proba_json)
            """, (id_cuest, MODEL_VERSION, pred_ml_canon, conf_pct, conf_label, proba_json))

        except Exception as e:
            import traceback
            print("[ML] Error guardando predicción:", e)
            traceback.print_exc()

        cn.commit()
        cur.close(); cn.close()

        # 6) Ir a resultados
        return redirect(f"/resultado?uid={id_usuario}")

    except Exception as e:
        return f"Error al guardar: {e}", 400

@app.get("/_ml_health")
def ml_health():
    try:
        ok = get_model() is not None
        return {"loaded": ok}, (200 if ok else 500)
    except Exception as e:
        return {"loaded": False, "err": str(e)}, 500
