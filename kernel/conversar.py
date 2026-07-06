"""Conversación asociativa — busca pares en tabla SQLite.
0 if/else, 0 textos fijos."""
import os, re, sqlite3

RUTA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos')
_STOP = frozenset('que de la el en y a los las un por con no se me te le lo su al del es fue era son sus mis tus este esa eso'.split())

class Conversar:
    def __init__(self):
        self._db = None
        self._cargar()

    def _cargar(self):
        ruta = os.path.join(RUTA, 'asociaciones.db')
        if os.path.exists(ruta):
            self._db = sqlite3.connect(ruta, check_same_thread=False)
            self._db.execute("PRAGMA query_only=1")

    def _normalizar(self, t):
        t = t.lower()
        for a, b in [('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u'),('ü','u')]:
            t = t.replace(a, b)
        return re.sub(r'[^a-zñ ]', ' ', t).strip()

    def tokenizar(self, texto):
        return self._normalizar(texto).split()

    def _clave(self, texto):
        norm = self._normalizar(texto)
        for w in norm.split():
            if len(w) >= 4 and w not in _STOP:
                return w
        return norm.split()[0] if norm.split() else ''

    def responder(self, entrada):
        if not self._db:
            return None
        toks = self.tokenizar(entrada)
        if not toks:
            return None
        entrada_norm = ' '.join(toks)
        # Buscar por palabra clave
        for t in toks:
            if len(t) < 4:
                continue
            cur = self._db.execute(
                "SELECT linea, respuesta FROM conversaciones WHERE clave=? ORDER BY id LIMIT 200",
                (t,))
            for linea, respuesta in cur.fetchall():
                if not respuesta or len(respuesta) < 5:
                    continue
                if entrada_norm in self._normalizar(linea):
                    punt = 100 + len(respuesta.split())
                    return respuesta[:160]
        # Búsqueda secundaria: LIKE en linea (solo para primera palabra)
        if toks:
            w = toks[0]
            if len(w) >= 3:
                cur = self._db.execute(
                    "SELECT respuesta FROM conversaciones WHERE clave=? LIMIT 100",
                    (w,))
                for (respuesta,) in cur.fetchall():
                    if respuesta and len(respuesta) > 10:
                        return respuesta[:160]
        return None

    def _formatear(self, t):
        if not t:
            return ''
        t = t.strip()
        t = t[0].upper() + t[1:]
        if t[-1] not in '.!?¿¡':
            t += '.'
        return t
