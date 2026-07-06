"""Acceso a datos léxicos — conocimiento, RAE, diccionario.
Carga bajo demanda. Sin indexación innecesaria."""
import os, json

RUTA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos')


class Conocimiento:
    def __init__(self):
        self._conocimiento = None
        self._rae = None
        self._diccionario = None

    def _cargar(self, archivo):
        ruta = os.path.join(RUTA, archivo)
        if os.path.exists(ruta):
            with open(ruta, encoding='utf-8') as f:
                return json.load(f)
        return {}

    @property
    def entidades(self):
        if self._conocimiento is None:
            self._conocimiento = self._cargar('conocimiento.json')
        return self._conocimiento.get('entidades', {})

    @property
    def rae(self):
        if self._rae is None:
            self._rae = self._cargar('rae_diccionario.json')
        return self._rae

    @property
    def diccionario(self):
        if self._diccionario is None:
            self._diccionario = self._cargar('diccionario.json')
        return self._diccionario

    def definir(self, palabra):
        p = palabra.lower().strip()
        if p in self.entidades:
            d = self.entidades[p]
            if isinstance(d, dict):
                return d.get('definicion') or d.get('tipo') or ''
            return str(d)
        if p in self.rae:
            entry = self.rae[p]
            if isinstance(entry, dict):
                defs = entry.get('definiciones')
                if defs and isinstance(defs, list):
                    return defs[0]
            return str(entry)
        if p in self.diccionario:
            entry = self.diccionario[p]
            if isinstance(entry, dict):
                return entry.get('definiciones', [''])[0] if isinstance(entry.get('definiciones'), list) else ''
            return str(entry)
        return None
