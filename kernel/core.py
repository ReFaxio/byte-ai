"""Byte — núcleo generativo con memoria.
N-grama + working memory de últimas interacciones."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernel.generador import Generador
from kernel.memoria import Memoria

class Byte:
    def __init__(self):
        self.memoria = Memoria()
        self.generador = Generador()

    def procesar(self, entrada):
        entrada = entrada.strip()
        if not entrada:
            return ""
        # Contexto: últimas 3 interacciones como semilla adicional
        contexto = ''
        for e, r in self.memoria.ultimos(3):
            for w in e.split():
                contexto += w + ' '
            for w in r.split():
                contexto += w + ' '
        respuesta = self.generador.responder(entrada, contexto.strip())
        self.memoria.guardar(entrada, respuesta or "")
        return respuesta or ""

    def __call__(self, entrada):
        return self.procesar(entrada)
