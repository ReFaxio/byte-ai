"""Byte — núcleo generativo con memoria.
Transformer + n-grama (fallback)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernel.generador import Generador
from kernel.memoria import Memoria
from kernel.transformer import Transformer, Vocabulario

RUTA_MODELO = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos', 'transformer_model.npz')

class Byte:
    def __init__(self):
        self.memoria = Memoria()
        self._g_ngrama = None
        self._transformer = None
        self._vocab = None
        self._cargar_transformer()

    def _cargar_transformer(self):
        self._vocab = Vocabulario.cargar()
        if self._vocab is None:
            return
        self._transformer = Transformer(vocab_size=self._vocab.size)
        if not self._transformer.cargar():
            self._transformer = None
            self._vocab = None

    def _generar_transformer(self, entrada):
        if self._transformer is None:
            return None
        try:
            palabras = Vocabulario._tokenizar(entrada)
            if not palabras:
                return None
            ids = self._vocab.encode(palabras)
            if not ids or all(i == 0 for i in ids):
                return None
            nuevos = self._transformer.generate(ids, max_new=24, temperature=0.8, top_k=20)
            todas = ids + nuevos
            texto = self._vocab.a_texto(todas)
            if len(texto.split()) < 3:
                return None
            texto = texto.strip()
            texto = texto[0].upper() + texto[1:] if texto else ''
            if texto and texto[-1] not in '.!?¿¡':
                texto += '.'
            return texto
        except Exception:
            return None

    def _generar_ngrama(self, entrada, contexto):
        if self._g_ngrama is None:
            self._g_ngrama = Generador()
        return self._g_ngrama.responder(entrada, contexto)

    def procesar(self, entrada):
        entrada = entrada.strip()
        if not entrada:
            return ""
        # Contexto de últimas interacciones
        contexto = ''
        for e, r in self.memoria.ultimos(3):
            for w in e.split():
                contexto += w + ' '
            for w in r.split():
                contexto += w + ' '
        # Transformer primero, n-grama como fallback
        respuesta = self._generar_transformer(entrada)
        if not respuesta:
            respuesta = self._generar_ngrama(entrada, contexto.strip())
        self.memoria.guardar(entrada, respuesta or "")
        return respuesta or ""

    def __call__(self, entrada):
        return self.procesar(entrada)
