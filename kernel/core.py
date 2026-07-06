"""Byte — núcleo asociativo puro.
Entrada → generación secuencial n-grama. 0 if/else."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernel.memoria import Memoria
from kernel.generador import Generador
from kernel.herramientas import calc


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
