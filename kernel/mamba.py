"""Mamba: modelo de estado SSM para Byte.
O(n) en tiempo, sin atencion cuadratica.
Usa embeddings y co-ocurrencia del Nucleo como inicializacion."""

import numpy as np
import os, json, re, time, pickle
from collections import Counter

RUTA_DATOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos')
RUTA_MODELO = os.path.join(RUTA_DATOS, 'mamba_model.npz')
RUTA_VOCAB = os.path.join(RUTA_DATOS, 'vocabulario.json')
RUTA_COOC = os.path.join(RUTA_DATOS, 'cooc_mat.npy')


class Vocabulario:
    def __init__(self, palabras=None):
        self.stoi = {}; self.itos = []
        if palabras:
            for p in palabras: self.stoi[p] = len(self.itos); self.itos.append(p)
    @property
    def size(self): return len(self.itos)
    @classmethod
    def desde_textos(cls, textos, max_size=16000, min_freq=2):
        c = Counter()
        for texto in textos:
            for pal in cls._tokenizar(texto): c[pal] += 1
        comunes = [p for p,_ in c.most_common(max_size-4) if _ >= min_freq]
        return cls(['<pad>','<unk>','<bos>','<eos>'] + comunes)
    @staticmethod
    def _tokenizar(texto):
        if not texto: return []
        texto = texto.lower()
        for a,b in [('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u'),('ü','u')]:
            texto = texto.replace(a,b)
        return re.sub(r'[^a-zñ ]',' ',texto).split()
    def encode(self, palabras):
        unk=self.stoi.get('<unk>',0); return [self.stoi.get(p,unk) for p in palabras]
    def decode(self, ids):
        return [self.itos[i] if i<len(self.itos) else '<unk>' for i in ids]
    def a_texto(self, ids): return ' '.join(self.decode(ids))
    def guardar(self, ruta=None):
        ruta=ruta or RUTA_VOCAB
        with open(ruta,'w',encoding='utf-8') as f: json.dump({'stoi':self.stoi,'itos':self.itos},f,ensure_ascii=False)
    @classmethod
    def cargar(cls, ruta=None):
        ruta=ruta or RUTA_VOCAB
        if not os.path.exists(ruta): return None
        with open(ruta,encoding='utf-8') as f: d=json.load(f)
        v=cls.__new__(cls); v.stoi=d['stoi']; v.itos=d['itos']; return v


class Mamba:
    def __init__(self, vocab_size=16000, d_model=128, d_state=64, d_ff=256):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_ff = d_ff

        s = 0.02
        rng = np.random.RandomState(42)
        self.emb = rng.randn(vocab_size, d_model).astype(np.float32) * s
        self.A = rng.randn(d_state, d_state).astype(np.float32) * 0.01
        self.B = rng.randn(d_model, d_state).astype(np.float32) * s
        self.C = rng.randn(d_state, d_model).astype(np.float32) * s
        self.W_ffn1 = rng.randn(d_model, d_ff).astype(np.float32) * s
        self.b_ffn1 = np.zeros(d_ff, dtype=np.float32)
        self.W_ffn2 = rng.randn(d_ff, vocab_size).astype(np.float32) * s
        self.b_ffn2 = np.zeros(vocab_size, dtype=np.float32)
        self.vocab = None

    def cargar_embeddings(self, ruta_nucleo):
        d = np.load(ruta_nucleo)
        if 'emb' in d and d['emb'].shape == self.emb.shape:
            self.emb = d['emb']
            print("  Embeddings cargados del Nucleo", flush=True)

    def forward(self, tokens_idx):
        B, T = tokens_idx.shape
        x = self.emb[tokens_idx]
        h = np.zeros((B, self.d_state), dtype=np.float32)
        for t in range(T):
            inp = x[:, t, :]
            h = h @ self.A.T + inp @ self.B
            out = h @ self.C
            if t == T - 1:
                logits = out
        logits = logits @ self.W_ffn1 + self.b_ffn1
        logits = np.maximum(0, logits)
        logits = logits @ self.W_ffn2 + self.b_ffn2
        return logits

    def generar(self, input_tokens, max_new=20, temperature=0.8, top_k=20):
        gen = list(input_tokens)
        h = np.zeros((1, self.d_state), dtype=np.float32)
        for _ in range(max_new):
            ctx = gen[-64:]
            arr = np.array([ctx], dtype=np.int64)
            x = self.emb[arr]
            for t in range(x.shape[1]):
                inp = x[:, t, :]
                h = h @ self.A.T + inp @ self.B
                out = h @ self.C
            logits = out @ self.W_ffn1 + self.b_ffn1
            logits = np.maximum(0, logits)
            logits = logits @ self.W_ffn2 + self.b_ffn2
            logits = logits[0] / temperature
            if top_k > 0:
                idxs = np.argpartition(-logits, top_k)[:top_k]
                vals = logits[idxs]
                exp_v = np.exp(vals - vals.max())
                probs = exp_v / exp_v.sum()
                choice = int(np.random.choice(idxs, p=probs))
            else:
                exp_v = np.exp(logits - logits.max())
                probs = exp_v / exp_v.sum()
                choice = int(np.random.choice(self.vocab_size, p=probs))
            gen.append(choice)
            if choice == 3: break
        return gen[len(input_tokens):]

    def entrenar(self, textos, epochs=2, lr=0.001):
        print("  Mamba entrenando...", flush=True)
        pasos = 0
        for epoch in range(epochs):
            losses = []
            for texto in textos:
                pals = Vocabulario._tokenizar(texto)
                if len(pals) < 8: continue
                ids = np.array(self.vocab.encode(pals), dtype=np.int64)
                B, T = 1, len(ids)
                x = self.emb[ids].reshape(1, T, -1)
                h = np.zeros((1, self.d_state), dtype=np.float32)
                for t in range(T - 1):
                    inp = x[:, t, :]
                    h_new = h @ self.A.T + inp @ self.B
                    out_s = h_new @ self.C
                    target = ids[t + 1]
                    h_f = np.maximum(0, out_s @ self.W_ffn1 + self.b_ffn1)
                    logits = h_f @ self.W_ffn2 + self.b_ffn2
                    logits_f = logits[0]
                    logits_f = logits_f - logits_f.max()
                    exp_l = np.exp(logits_f)
                    probs = exp_l / exp_l.sum()
                    loss = -np.log(probs[target] + 1e-10)
                    dlog = probs.copy()
                    dlog[target] -= 1.0
                    d_W_ffn2 = h_f.reshape(-1, 1) @ dlog.reshape(1, -1)
                    d_b_ffn2 = dlog
                    d_h_f = dlog @ self.W_ffn2.T
                    d_h_f = d_h_f * (h_f > 0)
                    d_W_ffn1 = out_s.T @ d_h_f
                    d_b_ffn1 = d_h_f[0]
                    d_out_s = d_h_f @ self.W_ffn1.T
                    self.W_ffn2 -= lr * d_W_ffn2
                    self.b_ffn2 -= lr * d_b_ffn2
                    self.W_ffn1 -= lr * d_W_ffn1
                    self.b_ffn1 -= lr * d_b_ffn1
                    d_C = h_new.T @ d_out_s
                    d_h = d_out_s @ self.C.T
                    d_B = inp.T @ d_h
                    self.C -= lr * d_C
                    self.B -= lr * d_B
                    h = h_new
                    losses.append(loss)
                    pasos += 1
                    if pasos % 500 == 0:
                        print(f"    paso {pasos} loss={np.mean(losses[-500:]):.4f}", flush=True)
            if losses:
                print(f"  epoch {epoch+1}/{epochs} loss={np.mean(losses):.4f}", flush=True)

    def guardar(self, ruta=None):
        ruta = ruta or RUTA_MODELO
        np.savez_compressed(ruta, emb=self.emb, A=self.A, B=self.B, C=self.C,
                           W_ffn1=self.W_ffn1, b_ffn1=self.b_ffn1,
                           W_ffn2=self.W_ffn2, b_ffn2=self.b_ffn2)

    def cargar(self, ruta=None):
        ruta = ruta or RUTA_MODELO
        if not os.path.exists(ruta): return False
        d = np.load(ruta); self.emb=d['emb']; self.A=d['A']; self.B=d['B']; self.C=d['C']
        self.W_ffn1=d['W_ffn1']; self.b_ffn1=d['b_ffn1']; self.W_ffn2=d['W_ffn2']; self.b_ffn2=d['b_ffn2']
        return True


def _extraer_rae(datos):
    textos = []
    if isinstance(datos, dict):
        for v in datos.values():
            if isinstance(v, dict):
                for dv in v.values():
                    if isinstance(dv, str) and len(dv) > 20: textos.append(dv)
            elif isinstance(v, str) and len(v) > 20: textos.append(v)
    return textos


def entrenar(rapido=False, epochs=2):
    print("=== Mamba (SSM) ===")
    t0 = time.time()

    rutas = []
    rae = os.path.join(RUTA_DATOS, 'rae_diccionario.json')
    if os.path.exists(rae): rutas.append(rae)
    import glob
    for r in sorted(glob.glob(os.path.join(RUTA_DATOS, 'wiki_parte_*.txt'))):
        if rapido and os.path.getsize(r) > 100e6:
            print(f"  Omitido: {os.path.basename(r)}"); continue
        rutas.append(r)
    print(f"  Archivos: {len(rutas)}")

    def iterar():
        for r in rutas:
            if r.endswith('.json'):
                with open(r) as f:
                    for t in _extraer_rae(json.load(f)): yield t
            else:
                with open(r) as f:
                    while True:
                        chunk = f.read(10*1048576)
                        if not chunk: break
                        yield chunk

    print("  Vocabulario...")
    vocab = Vocabulario.cargar()
    if vocab is None:
        vocab = Vocabulario.desde_textos(iterar(), max_size=16000, min_freq=1)
        vocab.guardar()
    print(f"    {vocab.size} palabras")

    m = Mamba(vocab_size=vocab.size)
    m.vocab = vocab
    if m.cargar():
        print("  Modelo Mamba cargado")
    else:
        ruta_nucleo = os.path.join(RUTA_DATOS, 'nucleo_model.npz')
        if os.path.exists(ruta_nucleo):
            m.cargar_embeddings(ruta_nucleo)

    m.entrenar(iterar(), epochs=epochs)
    m.guardar()

    print(f"\n  Hecho en {time.time()-t0:.1f}s")

    prompt = ['la', 'inteligencia', 'artificial']
    ids = [vocab.stoi.get(p, vocab.stoi.get('<unk>',0)) for p in prompt]
    nuevos = m.generar(ids, max_new=20)
    print(f"  Prompt: {' '.join(prompt)}")
    print(f"  Generado: {vocab.a_texto(ids + nuevos)}", flush=True)
    return m


if __name__ == '__main__':
    import sys
    rapido = '--rapido' in sys.argv
    entrenar(rapido=rapido)
