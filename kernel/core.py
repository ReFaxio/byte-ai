"""Byte — núcleo asociativo puro.
Memoria conversacional + n-grama como respaldo. 0 if/else."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernel.memoria import Memoria
from kernel.generador import Generador
from kernel.conversar import Conversar
from kernel.herramientas import calc


class Byte:
    def __init__(self):
        self.memoria = Memoria()
        self.generador = Generador()
        self.conversar = Conversar()
        self.ultimos = []

    def procesar(self, entrada):
        entrada = entrada.strip()
        if not entrada:
            return ""
        self.ultimos.append(entrada)
        if len(self.ultimos) > 10:
            self.ultimos.pop(0)
        respuesta = self.conversar.responder(entrada)
        if not respuesta or len(respuesta.split()) < 3:
            respuesta = self.generador.responder(entrada)
        if not respuesta:
            respuesta = ""
        self.memoria.guardar(entrada, respuesta)
        return self._formatear(respuesta)

    def _formatear(self, t):
        if not t:
            return ''
        t = t.strip()
        t = t[0].upper() + t[1:]
        if t[-1] not in '.!?¿¡':
            t += '.'
        return t

    def __call__(self, entrada):
        return self.procesar(entrada)
