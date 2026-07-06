"""Byte — núcleo generativo puro.
N-grama sobre datos reales. 0 if/else, 0 textos fijos."""
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
        respuesta = self.generador.responder(entrada)
        if not respuesta:
            respuesta = ""
        self.memoria.guardar(entrada, respuesta)
        return respuesta

    def __call__(self, entrada):
        return self.procesar(entrada)
