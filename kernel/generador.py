"""Generador n-grama híbrido — 4-gramas con fallback a 3-gramas.
0 if/else, 0 textos fijos, 0 categorías."""
import random, re, os, sqlite3

RUTA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos')
_CLITICOS = ['me','te','se','lo','la','le','nos','os','les']

class Generador:
    def __init__(self):
        self._db = None
        self._ultimas = []
        self._cargar_db()

    def _cargar_db(self):
        ruta = os.path.join(RUTA, 'asociaciones.db')
        if os.path.exists(ruta):
            self._db = sqlite3.connect(ruta, check_same_thread=False)
            self._db.execute("PRAGMA query_only=1")

    def tokenizar(self, texto):
        if not texto:
            return []
        texto = texto.lower()
        for a, b in [('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u'),('ü','u')]:
            texto = texto.replace(a, b)
        return re.sub(r'[^a-zñ ]', ' ', texto).split()

    def _peso3(self, w1, w2, lim=15):
        if not self._db:
            return []
        cur = self._db.execute(
            "SELECT w3, freq FROM ngramas WHERE w1=? AND w2=? ORDER BY freq DESC LIMIT ?",
            (w1, w2, lim))
        return cur.fetchall()

    def _peso4(self, w1, w2, w3, lim=15):
        if not self._db:
            return []
        try:
            cur = self._db.execute(
                "SELECT w4, freq FROM ngramas4 WHERE w1=? AND w2=? AND w3=? ORDER BY freq DESC LIMIT ?",
                (w1, w2, w3, lim))
            return cur.fetchall()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return []

    def _sig3(self, w1, w2):
        filas = self._peso3(w1, w2)
        if not filas:
            return None
        total = sum(f for _, f in filas)
        if total == 0:
            return None
        r = random.randint(0, total - 1)
        acum = 0
        for w3, f in filas:
            acum += f
            if r < acum:
                return w3
        return filas[-1][0]

    def _sig4(self, w1, w2, w3):
        filas = self._peso4(w1, w2, w3)
        if not filas:
            return None
        total = sum(f for _, f in filas)
        if total == 0:
            return None
        r = random.randint(0, total - 1)
        acum = 0
        for w4, f in filas:
            acum += f
            if r < acum:
                return w4
        return filas[-1][0]

    def _sig2(self, w):
        cur = self._db.execute(
            "SELECT w2, freq FROM ngramas WHERE w1=? ORDER BY freq DESC LIMIT 5", (w,))
        filas = cur.fetchall()
        if not filas:
            return None
        total = sum(f for _, f in filas)
        r = random.randint(0, total - 1)
        acum = 0
        for w2, f in filas:
            acum += f
            if r < acum:
                return w2
        return filas[-1][0]

    def _desclitic(self, w):
        for c in _CLITICOS:
            if w.endswith(c) and len(w) >= len(c) + 4:
                return w[:-len(c)], c
        return w, None

    def _buscar_bigrama(self, w1, w2):
        if self._peso3(w1, w2):
            return w1, w2
        w2f = self._sig2(w1)
        if w2f:
            return w1, w2f
        base, cl = self._desclitic(w1)
        if cl:
            w2f = self._sig2(base)
            if w2f:
                return base, w2f
        w2f = self._sig2(w2)
        if w2f:
            return w2, w2f
        base2, cl2 = self._desclitic(w2)
        if cl2:
            w2f = self._sig2(base2)
            if w2f:
                return base2, w2f
        for largo in range(len(w1)-1, 2, -1):
            cur = self._db.execute(
                "SELECT w1 FROM ngramas WHERE w1 LIKE ? GROUP BY w1 ORDER BY SUM(freq) DESC LIMIT 3",
                (w1[:largo] + '%',))
            for (w1p,) in cur.fetchall():
                w2f = self._sig2(w1p)
                if w2f:
                    return w1p, w2f
        # Último recurso: bigrama más común de toda la DB
        cur = self._db.execute(
            "SELECT w1, w2 FROM ngramas ORDER BY freq DESC LIMIT 1")
        row = cur.fetchone()
        if row:
            return row[0], row[1]
        return w1, w2

    def _caminar(self, semilla, max_pal=12):
        pal = list(semilla)
        vistos = set()
        for _ in range(max_pal - len(semilla)):
            ctx3 = len(pal) >= 3
            if ctx3:
                w4 = self._sig4(pal[-3], pal[-2], pal[-1])
                if w4 and w4 not in vistos and w4 not in pal[-2:] and w4 not in _CLITICOS:
                    vistos.add(w4)
                    pal.append(w4)
                    continue
            w3 = self._sig3(pal[-2], pal[-1])
            if not w3:
                w2f = self._sig2(pal[-1])
                if w2f and w2f not in vistos:
                    pal.append(w2f)
                    continue
                break
            if w3 in vistos or w3 in pal[-2:] or w3 in _CLITICOS or w3 == 'prnl' or len(w3) > 20:
                ok = [(w, f) for w, f in self._peso3(pal[-2], pal[-1], lim=25)
                      if w not in vistos and w not in pal[-2:] and w not in _CLITICOS and w != 'prnl' and len(w) <= 20]
                if not ok:
                    break
                total = sum(f for _, f in ok)
                r = random.randint(0, total - 1)
                acum = 0
                for w3n, f in ok:
                    acum += f
                    if r < acum:
                        vistos.add(w3n)
                        pal.append(w3n)
                        break
                else:
                    vistos.add(ok[-1][0])
                    pal.append(ok[-1][0])
                continue
            vistos.add(w3)
            pal.append(w3)
        return ' '.join(pal)

    def responder(self, entrada, contexto=''):
        toks = self.tokenizar(entrada)
        if not toks or all(len(t) <= 2 for t in toks):
            toks = ['hola']
        if len(toks) >= 2:
            w1, w2 = toks[-2], toks[-1]
        else:
            w = toks[-1]
            w2 = self._sig2(w)
            if not w2:
                base, _ = self._desclitic(w)
                w2 = self._sig2(base)
                if not w2:
                    w1, w2 = w, w
                else:
                    w1 = base
            else:
                w1 = w
        w1, w2 = self._buscar_bigrama(w1, w2)
        semilla = toks + [w2]
        if len(semilla) < 2:
            semilla = [w1, w2]
        for _ in range(8):
            t = self._caminar(semilla)
            if t and t not in self._ultimas and len(t.split()) >= 3:
                self._ultimas.append(t)
                if len(self._ultimas) > 5:
                    self._ultimas.pop(0)
                return self._formatear(t)
        self._ultimas.append(t)
        if len(self._ultimas) > 5:
            self._ultimas.pop(0)
        return self._formatear(t)

    def _formatear(self, t):
        if not t:
            return ''
        t = t.strip()
        t = t[0].upper() + t[1:]
        if t[-1] not in '.!?¿¡':
            t += '.'
        return t
