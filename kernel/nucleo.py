"""Nucleo: modelo base de Byte.
Fase 1: co-ocurrencia (segundos) + embeddings + FFN ligero.
Prepara estado interno para Fase 2 (Mamba)."""

import numpy as np
import os, json, re, time, pickle
from collections import Counter

RUTA_DATOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos')
RUTA_MODELO = os.path.join(RUTA_DATOS, 'nucleo_model.npz')
RUTA_VOCAB = os.path.join(RUTA_DATOS, 'vocabulario.json')
RUTA_COOC = os.path.join(RUTA_DATOS, 'cooc_mat.npy')


class Vocabulario:
    def __init__(self, palabras=None):
        self.stoi = {}
        self.itos = []
        if palabras:
            for p in palabras:
                self.stoi[p] = len(self.itos)
                self.itos.append(p)

    @property
    def size(self): return len(self.itos)

    @classmethod
    def desde_textos(cls, textos, max_size=16000, min_freq=2):
        c = Counter()
        for texto in textos:
            for pal in cls._tokenizar(texto):
                c[pal] += 1
        comunes = [p for p, _ in c.most_common(max_size - 4) if _ >= min_freq]
        return cls(['<pad>', '<unk>', '<bos>', '<eos>'] + comunes)

    @staticmethod
    def _tokenizar(texto):
        if not texto: return []
        texto = texto.lower()
        for a, b in [('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u'),('ü','u')]:
            texto = texto.replace(a, b)
        return re.sub(r'[^a-zñ ]', ' ', texto).split()

    def encode(self, palabras):
        unk = self.stoi.get('<unk>', 0)
        return [self.stoi.get(p, unk) for p in palabras]

    def decode(self, ids):
        return [self.itos[i] if i < len(self.itos) else '<unk>' for i in ids]

    def a_texto(self, ids): return ' '.join(self.decode(ids))

    def guardar(self, ruta=None):
        ruta = ruta or RUTA_VOCAB
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump({'stoi': self.stoi, 'itos': self.itos}, f, ensure_ascii=False)

    @classmethod
    def cargar(cls, ruta=None):
        ruta = ruta or RUTA_VOCAB
        if not os.path.exists(ruta): return None
        with open(ruta, encoding='utf-8') as f: d = json.load(f)
        v = cls.__new__(cls)
        v.stoi = d['stoi']; v.itos = d['itos']; return v


class Nucleo:
    def __init__(self, vocab_size=16000, d_model=128, d_ff=256):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.cooc = None
        self.vocab = None

        s = 0.02
        rng = np.random.RandomState(42)
        self.emb = rng.randn(vocab_size, d_model).astype(np.float32) * s
        self.W_ffn1 = rng.randn(d_model, d_ff).astype(np.float32) * s
        self.b_ffn1 = np.zeros(d_ff, dtype=np.float32)
        self.W_ffn2 = rng.randn(d_ff, vocab_size).astype(np.float32) * s
        self.b_ffn2 = np.zeros(vocab_size, dtype=np.float32)
        self.ln_g = np.ones(d_model, dtype=np.float32)
        self.ln_b = np.zeros(d_model, dtype=np.float32)

    def entrenar_cooc(self, textos, ventana=4):
        print("  Contando co-ocurrencias...", flush=True)
        n = self.vocab_size
        cooc = np.zeros((n, n), dtype=np.float32)
        total = 0
        for texto in textos:
            pals = Vocabulario._tokenizar(texto)
            if not pals: continue
            ids = self.vocab.encode(pals)
            for i in range(len(ids)):
                ini = max(0, i - ventana)
                fin = min(len(ids), i + ventana + 1)
                for j in range(ini, fin):
                    if i != j:
                        cooc[ids[i], ids[j]] += 1.0
                        total += 1
        cooc = cooc / (cooc.sum(axis=-1, keepdims=True) + 1e-8)
        cooc = np.nan_to_num(cooc, nan=0.0, posinf=0.0, neginf=0.0)
        np.save(RUTA_COOC, cooc)
        self.cooc = cooc.astype(np.float32)
        print(f"    {total} pares, matriz {n}x{n}", flush=True)

    def cargar_cooc(self):
        if os.path.exists(RUTA_COOC):
            self.cooc = np.load(RUTA_COOC).astype(np.float32)
            return True
        return False

    def forward(self, tokens_idx):
        B, T = tokens_idx.shape
        x = self.emb[tokens_idx]
        x_prom = x.mean(axis=1)
        h = np.maximum(0, x_prom @ self.W_ffn1 + self.b_ffn1)
        logits = h @ self.W_ffn2 + self.b_ffn2
        if self.cooc is not None:
            ult_id = tokens_idx[:, -1]
            cooc_logits = self.cooc[ult_id] * 5.0
            logits = logits + cooc_logits
        return logits

    def generar(self, input_tokens, max_new=20, temperature=0.8, top_k=20):
        gen = list(input_tokens)
        for _ in range(max_new):
            if len(gen) > 64:
                ctx = gen[-64:]
            else:
                ctx = gen
            arr = np.array([ctx], dtype=np.int64)
            logits = self.forward(arr)
            logits_last = logits[0] / temperature
            if top_k > 0:
                idxs = np.argpartition(-logits_last, top_k)[:top_k]
                vals = logits_last[idxs]
                exp_vals = np.exp(vals - vals.max())
                probs = exp_vals / exp_vals.sum()
                choice = int(np.random.choice(idxs, p=probs))
            else:
                exp_vals = np.exp(logits_last - logits_last.max())
                probs = exp_vals / exp_vals.sum()
                choice = int(np.random.choice(self.vocab_size, p=probs))
            gen.append(choice)
            if choice == 3:
                break
        return gen[len(input_tokens):]

    def entrenar(self, textos, epochs=1, lr=0.01):
        print("  Entrenando FFN por lotes...", flush=True)
        pasos = 0
        for epoch in range(epochs):
            losses = []
            for texto in textos:
                pals = Vocabulario._tokenizar(texto)
                if len(pals) < 8: continue
                ids = np.array(self.vocab.encode(pals), dtype=np.int64)
                for start in range(4, len(ids) - 1, 8):
                    ctx = ids[start-4:start+1]
                    target = ids[start+1]
                    arr = np.array([ctx], dtype=np.int64)
                    x = self.emb[arr].mean(axis=1)
                    h = np.maximum(0, x @ self.W_ffn1 + self.b_ffn1)
                    logits = h @ self.W_ffn2 + self.b_ffn2
                    if self.cooc is not None:
                        logits = logits + self.cooc[arr[0, -1]] * 3.0
                    logits_f = logits[0]
                    logits_f = logits_f - logits_f.max()
                    exp_l = np.exp(logits_f)
                    probs = exp_l / exp_l.sum()
                    loss = -np.log(probs[target] + 1e-10)
                    dlog = probs.copy()
                    dlog[target] -= 1.0
                    self.W_ffn2 -= lr * h.reshape(-1, 1) @ dlog.reshape(1, -1)
                    self.b_ffn2 -= lr * dlog
                    dh = (dlog @ self.W_ffn2.T) * (h > 0)
                    self.W_ffn1 -= lr * (x.reshape(-1, 1) @ dh.reshape(1, -1))
                    self.b_ffn1 -= lr * dh[0]
                    losses.append(loss)
                    pasos += 1
                    if pasos % 500 == 0:
                        print(f"    paso {pasos} loss={np.mean(losses[-500:]):.4f}", flush=True)
            if losses:
                print(f"  epoch {epoch+1}/{epochs} loss={np.mean(losses):.4f}", flush=True)

    def guardar(self, ruta=None):
        ruta = ruta or RUTA_MODELO
        np.savez_compressed(ruta,
            emb=self.emb, W_ffn1=self.W_ffn1, b_ffn1=self.b_ffn1,
            W_ffn2=self.W_ffn2, b_ffn2=self.b_ffn2,
            ln_g=self.ln_g, ln_b=self.ln_b)

    def cargar(self, ruta=None):
        ruta = ruta or RUTA_MODELO
        if not os.path.exists(ruta): return False
        d = np.load(ruta)
        self.emb = d['emb']
        self.W_ffn1 = d['W_ffn1']
        self.b_ffn1 = d['b_ffn1']
        self.W_ffn2 = d['W_ffn2']
        self.b_ffn2 = d['b_ffn2']
        self.ln_g = d['ln_g']
        self.ln_b = d['ln_b']
        self.vocab_size = self.emb.shape[0]
        self.d_model = self.emb.shape[1]
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


def entrenar(rapido=False, epochs=1):
    print("=== Nucleo ===")
    t0 = time.time()

    rutas = []
    rae = os.path.join(RUTA_DATOS, 'rae_diccionario.json')
    if os.path.exists(rae): rutas.append(rae)
    import glob
    for r in sorted(glob.glob(os.path.join(RUTA_DATOS, 'wiki_parte_*.txt'))):
        if rapido and os.path.getsize(r) > 100e6:
            print(f"  Omitido: {os.path.basename(r)}")
            continue
        rutas.append(r)
    print(f"  Archivos: {len(rutas)}")

    def iterar():
        for r in rutas:
            if r.endswith('.json'):
                with open(r, encoding='utf-8') as f:
                    for t in _extraer_rae(json.load(f)):
                        yield t
            else:
                with open(r, encoding='utf-8') as f:
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

    nuc = Nucleo(vocab_size=vocab.size)
    nuc.vocab = vocab
    if nuc.cargar():
        print("  Modelo cargado")
    if nuc.cargar_cooc():
        print("  Co-ocurrencia cargada")
    else:
        nuc.entrenar_cooc(iterar())
    nuc.entrenar(iterar(), epochs=epochs)
    nuc.guardar()

    print(f"\n  Hecho en {time.time()-t0:.1f}s")

    prompt = ['la', 'inteligencia', 'artificial']
    ids = [vocab.stoi.get(p, vocab.stoi.get('<unk>', 0)) for p in prompt]
    nuevos = nuc.generar(ids, max_new=20)
    print(f"  Prompt: {' '.join(prompt)}")
    print(f"  Generado: {vocab.a_texto(ids + nuevos)}", flush=True)
    return nuc


if __name__ == '__main__':
    import sys
    rapido = '--rapido' in sys.argv
    entrenar(rapido=rapido)
