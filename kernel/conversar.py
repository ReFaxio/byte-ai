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
        cur = self._db.execute(
            "SELECT name FROM sqlite_master WHERE name='c_fts'")
        if cur.fetchone():
            self._fts = True

    def _norm(self, t):
        t = t.lower()
        for a, b in [('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u'),('ü','u')]:
            t = t.replace(a, b)
        return re.sub(r'[^a-zñ ]', ' ', t).strip()

    def _palabras(self, texto):
        return [w for w in self._norm(texto).split()
                if len(w) >= 2 and w not in _STOP]

    def _variantes(self, palabras):
        v = set(palabras)
        for t in palabras:
            if len(t) > 4:
                v.add(t[:4])
            if len(t) > 5:
                v.add(t[:5])
        return list(v)

    def _cuantas_matchean(self, palabras, linea):
        nl = self._norm(linea)
        return sum(1 for p in palabras if p in nl)

    def responder(self, entrada):
        if not self._db:
            return None
        raw = self._palabras(entrada)
        if not raw:
            return None
        variantes = self._variantes(raw)
        if self._fts:
            # AND: todas las palabras originales en la misma línea
            if len(raw) >= 2:
                q_and = ' AND '.join(f'linea:{t}' for t in raw)
                cur = self._db.execute(
                    "SELECT linea, respuesta FROM c_fts WHERE c_fts MATCH ? LIMIT 5",
                    (q_and,))
                for linea_fts, respuesta in cur.fetchall():
                    if len(respuesta) >= 8:
                        return respuesta[:300]
            # OR: buscar con variantes, elegir la que más palabras originales contenga
            q_or = ' OR '.join(f'linea:{t}' for t in variantes)
            cur = self._db.execute(
                "SELECT linea, respuesta FROM c_fts WHERE c_fts MATCH ? LIMIT 30",
                (q_or,))
            mejor_linea, mejor_resp, mejor_punt = None, None, 0
            for linea_fts, respuesta in cur.fetchall():
                if len(respuesta) < 8 or respuesta.count(' ') < 1:
                    continue
                punt = self._cuantas_matchean(raw, linea_fts)
                if punt > mejor_punt:
                    mejor_punt = punt
                    mejor_resp = respuesta
                    mejor_linea = linea_fts
            if mejor_punt > 0 and mejor_resp:
                return mejor_resp[:300]
        # LIKE fallback
        for intento in [raw[0], raw[0][:4], raw[0][:3]]:
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
        cur = self._db.execute(
            "SELECT respuesta FROM conversaciones WHERE clave LIKE ? LIMIT 30",
            (raw[0][:4] + '%',))
        for (respuesta,) in cur.fetchall():
            if respuesta and len(respuesta) > 10:
                return respuesta[:200]
        return None
