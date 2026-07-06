"""Byte — núcleo n-grama con respaldo conversacional."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernel.generador import Generador
from kernel.conversar import Conversar
from kernel.memoria import Memoria

class Byte:
    def __init__(self):
        self.memoria = Memoria()
        self.generador = Generador()
        self.conversar = Conversar()

    def procesar(self, entrada):
        entrada = entrada.strip()
        if not entrada:
            return ""
        respuesta = self.generador.responder(entrada)
        if not respuesta:
            respuesta = self.conversar.responder(entrada)
        self.memoria.guardar(entrada, respuesta or "")
        return respuesta or ""

    def __call__(self, entrada):
        return self.procesar(entrada)
