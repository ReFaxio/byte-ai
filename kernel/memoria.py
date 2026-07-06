"""Persistencia SQLite — conversaciones y sesión.
Sin dependencias, sin if/else, solo datos."""
import sqlite3, os, time, json, re
from datetime import datetime, timedelta

RUTA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos')
RUTA_DB = os.path.join(RUTA, 'memoria.db')


class Memoria:
    def __init__(self):
        os.makedirs(RUTA, exist_ok=True)
        self.db = sqlite3.connect(RUTA_DB, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS conversaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entrada TEXT, respuesta TEXT,
            timestamp REAL, fecha TEXT
        )""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS kv (
            key TEXT PRIMARY KEY, value TEXT
        )""")
        self.db.commit()
        self.sesion = self._cargar('sesion', {'historial': [], 'palabras': []})

    def _cargar(self, key, default):
        cur = self.db.execute("SELECT value FROM kv WHERE key=?", (key,))
        row = cur.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except Exception:
                pass
        return default

    def _guardar(self, key, valor):
        self.db.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?,?)",
                        (key, json.dumps(valor, ensure_ascii=False)))
        self.db.commit()

    def guardar(self, entrada, respuesta):
        ahora = time.time()
        hoy = datetime.now().strftime('%Y-%m-%d')
        self.db.execute("INSERT INTO conversaciones (entrada, respuesta, timestamp, fecha) VALUES (?,?,?,?)",
                        (entrada, respuesta, ahora, hoy))
        self.db.commit()
        self.sesion['historial'].append({
            't': datetime.now().strftime('%H:%M'),
            'u': entrada[:80],
        })
        if len(self.sesion['historial']) > 30:
            self.sesion['historial'] = self.sesion['historial'][-30:]
        pals = set(self.sesion.get('palabras', []))
        for p in re.findall(r'[a-záéíóúñ]{4,}', entrada.lower()):
            if p not in pals and len(pals) < 100:
                pals.add(p)
        self.sesion['palabras'] = list(pals)[-100:]
        self._guardar('sesion', self.sesion)

    def ultimos(self, n=5):
        cur = self.db.execute(
            "SELECT entrada, respuesta FROM conversaciones ORDER BY id DESC LIMIT ?", (n,))
        return list(reversed(cur.fetchall()))

    def buscar(self, termino):
        cur = self.db.execute(
            "SELECT entrada, respuesta FROM conversaciones WHERE entrada LIKE ? OR respuesta LIKE ? ORDER BY id DESC LIMIT 5",
            (f'%{termino}%', f'%{termino}%'))
        return cur.fetchall()
