"""Byte — núcleo generativo con Mamba + memoria."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernel.mamba import Mamba, Vocabulario, RUTA_DATOS
from kernel.memoria import Memoria

RUTA_MODELO = os.path.join(RUTA_DATOS, 'mamba_model.npz')

class Byte:
    def __init__(self):
        self.memoria = Memoria()
        self._mamba = None
        self._vocab = None
        self._cargar_mamba()

    def _cargar_mamba(self):
        self._vocab = Vocabulario.cargar()
        if self._vocab is None:
            return
        self._mamba = Mamba(vocab_size=self._vocab.size)
        self._mamba.vocab = self._vocab
        if not self._mamba.cargar():
            self._mamba = None
            self._vocab = None

    def _generar_mamba(self, entrada):
        if self._mamba is None:
            return None
        try:
            palabras = Vocabulario._tokenizar(entrada)
            if not palabras:
                return None
            ids = self._vocab.encode(palabras)
            if not ids:
                return None
            nuevos = self._mamba.generar(ids, max_new=36, temperature=0.7, top_k=0, top_p=0.9)
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

    def procesar(self, entrada):
        entrada = entrada.strip()
        if not entrada:
            return ""
        contexto = ''
        for e, r in self.memoria.ultimos(3):
            for w in e.split():
                contexto += w + ' '
            for w in r.split():
                contexto += w + ' '
        respuesta = self._generar_mamba(entrada)
        self.memoria.guardar(entrada, respuesta or "")
        return respuesta or ""

    def __call__(self, entrada):
        return self.procesar(entrada)
