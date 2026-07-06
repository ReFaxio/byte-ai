"""Conversación asociativa.
Busca en subtítulos reales usando FTS5. 0 if/else, 0 textos fijos."""
import os, re, sqlite3

RUTA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos')
_STOP = frozenset('de la el en y a e i o u su tu mis sus al del'.split())

class Conversar:
    def __init__(self):
        self._db = None
        self._fts = None
        self._cargar()

    def _cargar(self):
        ruta = os.path.join(RUTA, 'asociaciones.db')
        if not os.path.exists(ruta):
            return
        self._db = sqlite3.connect(ruta, check_same_thread=False)
        self._db.execute("PRAGMA query_only=1")
        # Detectar si la tabla FTS5 existe
        cur = self._db.execute(
            "SELECT name FROM sqlite_master WHERE name='c_fts'")
        if cur.fetchone():
            self._fts = True
        else:
            self._fts = False

    def _norm(self, t):
        t = t.lower()
        for a, b in [('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u'),('ü','u')]:
            t = t.replace(a, b)
        return re.sub(r'[^a-zñ ]', ' ', t).strip()

    def _tokens(self, texto):
        raw = [w for w in self._norm(texto).split()
               if len(w) >= 2 and w not in _STOP]
        variantes = set(raw)
        for t in raw:
            if len(t) > 4:
                variantes.add(t[:4])
            if len(t) > 5:
                variantes.add(t[:5])
        return list(variantes)

    def responder(self, entrada):
        if not self._db:
            return None
        toks = self._tokens(entrada)
        if not toks:
            return None

        # 1. FTS5 si disponible
        if self._fts:
            query = ' OR '.join(f'linea:{t}' for t in toks)
            cur = self._db.execute(
                "SELECT respuesta FROM c_fts WHERE c_fts MATCH ? LIMIT 30",
                (query,))
            for (respuesta,) in cur.fetchall():
                if len(respuesta) >= 8 and respuesta.count(' ') >= 1:
                    return respuesta[:300]

        # 2. Fallback: LIKE sobre clave (primeros 4+ chars)
        for intento in [toks[0], toks[0][:4], toks[0][:3]]:
            if len(intento) < 3:
                continue
            cur = self._db.execute(
                "SELECT linea, respuesta FROM conversaciones WHERE clave LIKE ? ORDER BY id LIMIT 100",
                (intento + '%',))
            for linea, respuesta in cur.fetchall():
                if not respuesta or len(respuesta) < 8:
                    continue
                if self._norm(entrada) in self._norm(linea) or len(intento) <= 4:
                    return respuesta[:200]
        # 3. Fallback extremo: cualquier respuesta con esa clave
        cur = self._db.execute(
            "SELECT respuesta FROM conversaciones WHERE clave LIKE ? LIMIT 30",
            (toks[0][:4] + '%',))
        for (respuesta,) in cur.fetchall():
            if respuesta and len(respuesta) > 10:
                return respuesta[:200]
        return None
