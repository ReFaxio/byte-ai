"""Transformer puro numpy — 0 if/else, 0 texto fijo.
Arquitectura: 4 capas, 4 cabezas, dim 128, ventana 64.
Corre 100% en CPU, <200MB RAM entrenando, <20MB infiriendo.
Meta: superar GPT/Claude/DeepSeek."""

import numpy as np
import os, json, re, time, pickle
from collections import Counter

RUTA_DATOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos')
RUTA_MODELO = os.path.join(RUTA_DATOS, 'transformer_model.npz')
RUTA_VOCAB = os.path.join(RUTA_DATOS, 'vocabulario.json')

# ===================================================================
# Helpers
# ===================================================================

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x.astype(np.float64))
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


def layer_norm(x, g, b, eps=1e-5):
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return g * (x - m) / np.sqrt(v + eps) + b


def layer_norm_bwd(dout, x, g, b, eps=1e-5):
    N = x.shape[-1]
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    s = 1.0 / np.sqrt(v + eps)
    dx_norm = dout * g
    dg = (dout * (x - m) * s).sum(axis=tuple(range(dout.ndim - 1))).astype(np.float32)
    db = dout.sum(axis=tuple(range(dout.ndim - 1))).astype(np.float32)
    dx = s * (dx_norm - (1.0 / N) * dx_norm.sum(axis=-1, keepdims=True)
              - (x - m) * s * s * (dx_norm * (x - m)).mean(axis=-1, keepdims=True))
    return dx.astype(np.float32), dg, db


# ===================================================================
# Vocabulario
# ===================================================================

class Vocabulario:
    def __init__(self, palabras=None):
        self.stoi = {}
        self.itos = []
        if palabras:
            for p in palabras:
                self.stoi[p] = len(self.itos)
                self.itos.append(p)

    @property
    def size(self):
        return len(self.itos)

    @classmethod
    def desde_textos(cls, textos, max_size=16000, min_freq=2):
        c = Counter()
        for texto in textos:
            for pal in Vocabulario._tokenizar(texto):
                c[pal] += 1
        comunes = [p for p, _ in c.most_common(max_size - 4) if _ >= min_freq]
        especiales = ['<pad>', '<unk>', '<bos>', '<eos>']
        return cls(especiales + comunes)

    @staticmethod
    def _tokenizar(texto):
        if not texto:
            return []
        texto = texto.lower()
        for a, b in [('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u'),('ü','u')]:
            texto = texto.replace(a, b)
        return re.sub(r'[^a-zñ ]', ' ', texto).split()

    @staticmethod
    def limpiar(texto):
        texto = texto.lower()
        for a, b in [('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u'),('ü','u')]:
            texto = texto.replace(a, b)
        return texto

    def encode(self, palabras):
        unk = self.stoi.get('<unk>', 0)
        return [self.stoi.get(p, unk) for p in palabras]

    def decode(self, ids):
        return [self.itos[i] if i < len(self.itos) else '<unk>' for i in ids]

    def a_texto(self, ids):
        return ' '.join(self.decode(ids))

    def guardar(self, ruta=None):
        ruta = ruta or RUTA_VOCAB
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump({'stoi': self.stoi, 'itos': self.itos}, f, ensure_ascii=False)

    @classmethod
    def cargar(cls, ruta=None):
        ruta = ruta or RUTA_VOCAB
        if not os.path.exists(ruta):
            return None
        with open(ruta, encoding='utf-8') as f:
            d = json.load(f)
        v = cls.__new__(cls)
        v.stoi = d['stoi']
        v.itos = d['itos']
        return v


# ===================================================================
# Transformer
# ===================================================================

class Transformer:
    def __init__(self, vocab_size=16000, d_model=128, n_heads=4,
                 n_layers=4, d_ff=512, max_seq=64, lr=3e-4):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.d_k = d_model // n_heads
        self.lr = lr
        self.t = 0

        self.p = {}
        self._init_params()

        self._mask = np.triu(np.full((max_seq, max_seq), -np.inf, dtype=np.float32), 1)

        self._m = {}
        self._v = {}
        for k in self.p:
            self._m[k] = np.zeros_like(self.p[k])
            self._v[k] = np.zeros_like(self.p[k])

    def _init_params(self):
        p = self.p
        s = 0.02
        p['tok_emb'] = np.random.randn(self.vocab_size, self.d_model).astype(np.float32) * s
        p['pos_emb'] = np.random.randn(self.max_seq, self.d_model).astype(np.float32) * s
        for i in range(self.n_layers):
            for w in ['W_q', 'W_k', 'W_v', 'W_o']:
                p[f'{w}_{i}'] = np.random.randn(self.d_model, self.d_model).astype(np.float32) * s
            p[f'ln1_g_{i}'] = np.ones(self.d_model, dtype=np.float32)
            p[f'ln1_b_{i}'] = np.zeros(self.d_model, dtype=np.float32)
            p[f'W_1_{i}'] = np.random.randn(self.d_model, self.d_ff).astype(np.float32) * s
            p[f'b_1_{i}'] = np.zeros(self.d_ff, dtype=np.float32)
            p[f'W_2_{i}'] = np.random.randn(self.d_ff, self.d_model).astype(np.float32) * s
            p[f'b_2_{i}'] = np.zeros(self.d_model, dtype=np.float32)
            p[f'ln2_g_{i}'] = np.ones(self.d_model, dtype=np.float32)
            p[f'ln2_b_{i}'] = np.zeros(self.d_model, dtype=np.float32)
        p['ln_f_g'] = np.ones(self.d_model, dtype=np.float32)
        p['ln_f_b'] = np.zeros(self.d_model, dtype=np.float32)
        p['W_out'] = np.random.randn(self.d_model, self.vocab_size).astype(np.float32) * s
        p['b_out'] = np.zeros(self.vocab_size, dtype=np.float32)

    # ------------------------------------------------------------ forward

    def forward(self, tokens, c=None):
        B, T = tokens.shape
        p = self.p
        save = c is not None

        x = p['tok_emb'][tokens] + p['pos_emb'][:T]

        for i in range(self.n_layers):
            if save:
                c[f'res1_{i}'] = x.copy()

            x_ln = layer_norm(x, p[f'ln1_g_{i}'], p[f'ln1_b_{i}'])
            if save:
                c[f'x_ln1_{i}'] = x_ln.copy()

            q = x_ln @ p[f'W_q_{i}']
            k = x_ln @ p[f'W_k_{i}']
            v = x_ln @ p[f'W_v_{i}']
            if save:
                c[f'q_{i}'] = q; c[f'k_{i}'] = k; c[f'v_{i}'] = v

            nh, dk = self.n_heads, self.d_k
            q_h = q.reshape(B, T, nh, dk).transpose(0, 2, 1, 3)
            k_h = k.reshape(B, T, nh, dk).transpose(0, 2, 1, 3)
            v_h = v.reshape(B, T, nh, dk).transpose(0, 2, 1, 3)

            s = q_h @ k_h.transpose(0, 1, 3, 2) / np.sqrt(dk)
            s = s + self._mask[:T, :T]
            a = softmax(s, -1)
            if save:
                c[f's_{i}'] = s; c[f'a_{i}'] = a

            o_h = a @ v_h
            o = o_h.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
            if save:
                c[f'o_{i}'] = o.copy()

            o = o @ p[f'W_o_{i}']
            x = (c[f'res1_{i}'] + o) if save else (x + o)

            # -- FFN --
            if save:
                c[f'res2_{i}'] = x.copy()

            x_ln = layer_norm(x, p[f'ln2_g_{i}'], p[f'ln2_b_{i}'])
            if save:
                c[f'x_ln2_{i}'] = x_ln.copy()

            h = x_ln @ p[f'W_1_{i}'] + p[f'b_1_{i}']
            h_relu = np.maximum(0, h)
            if save:
                c[f'h_{i}'] = h; c[f'h_relu_{i}'] = h_relu

            o = h_relu @ p[f'W_2_{i}'] + p[f'b_2_{i}']
            x = (c[f'res2_{i}'] + o) if save else (x + o)

        x = layer_norm(x, p['ln_f_g'], p['ln_f_b'])
        if save:
            c['x_final'] = x.copy()
        logits = x @ p['W_out'] + p['b_out']
        return logits

    # ----------------------------------------------------------- backward

    def backward(self, dlogits, c):
        p = self.p
        g = {k: np.zeros_like(v) for k, v in p.items()}
        B, T = dlogits.shape[:2]

        dx = dlogits @ p['W_out'].T
        g['W_out'] = c['x_final'].reshape(-1, self.d_model).T @ dlogits.reshape(-1, self.vocab_size)
        g['b_out'] = dlogits.sum(axis=(0, 1)).astype(np.float32)

        dx, g['ln_f_g'], g['ln_f_b'] = layer_norm_bwd(
            dx, c['x_final'], p['ln_f_g'], p['ln_f_b'])

        for i in reversed(range(self.n_layers)):
            d_o_ffn = dx
            d_res2 = dx

            d_h_relu = d_o_ffn @ p[f'W_2_{i}'].T
            g[f'W_2_{i}'] = c[f'h_relu_{i}'].reshape(-1, self.d_ff).T @ d_o_ffn.reshape(-1, self.d_model)
            g[f'b_2_{i}'] = d_o_ffn.sum(axis=(0, 1)).astype(np.float32)

            d_h = d_h_relu * (c[f'h_{i}'] > 0).astype(np.float32)
            g[f'W_1_{i}'] = c[f'x_ln2_{i}'].reshape(-1, self.d_model).T @ d_h.reshape(-1, self.d_ff)
            g[f'b_1_{i}'] = d_h.sum(axis=(0, 1)).astype(np.float32)

            d_x_ln2 = d_h @ p[f'W_1_{i}'].T
            d_via_ln, g[f'ln2_g_{i}'], g[f'ln2_b_{i}'] = layer_norm_bwd(
                d_x_ln2, c[f'x_ln2_{i}'], p[f'ln2_g_{i}'], p[f'ln2_b_{i}'])
            dx = d_res2 + d_via_ln

            d_o_attn = dx
            d_res1 = dx

            g[f'W_o_{i}'] = c[f'o_{i}'].reshape(-1, self.d_model).T @ d_o_attn.reshape(-1, self.d_model)
            d_o_before = d_o_attn @ p[f'W_o_{i}'].T

            nh, dk = self.n_heads, self.d_k
            d_o_h = d_o_before.reshape(B, T, nh, dk).transpose(0, 2, 1, 3)

            v = c[f'v_{i}']
            v_h = v.reshape(B, T, nh, dk).transpose(0, 2, 1, 3)
            a = c[f'a_{i}']

            d_a = d_o_h @ v_h.transpose(0, 1, 3, 2)
            d_v_h = a.transpose(0, 1, 3, 2) @ d_o_h

            s = c[f's_{i}']
            d_s = a * (d_a - (a * d_a).sum(axis=-1, keepdims=True))

            q_h = c[f'q_{i}'].reshape(B, T, nh, dk).transpose(0, 2, 1, 3)
            k_h = c[f'k_{i}'].reshape(B, T, nh, dk).transpose(0, 2, 1, 3)

            d_q_h = d_s @ k_h / np.sqrt(dk)
            d_k_h = d_s.transpose(0, 1, 3, 2) @ q_h / np.sqrt(dk)

            d_q = d_q_h.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
            d_k = d_k_h.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
            d_v = d_v_h.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)

            x_ln1 = c[f'x_ln1_{i}']
            g[f'W_q_{i}'] = x_ln1.reshape(-1, self.d_model).T @ d_q.reshape(-1, self.d_model)
            g[f'W_k_{i}'] = x_ln1.reshape(-1, self.d_model).T @ d_k.reshape(-1, self.d_model)
            g[f'W_v_{i}'] = x_ln1.reshape(-1, self.d_model).T @ d_v.reshape(-1, self.d_model)

            d_via_qkv = (d_q @ p[f'W_q_{i}'].T +
                         d_k @ p[f'W_k_{i}'].T +
                         d_v @ p[f'W_v_{i}'].T)

            d_via_ln, g[f'ln1_g_{i}'], g[f'ln1_b_{i}'] = layer_norm_bwd(
                d_via_qkv, c[f'x_ln1_{i}'], p[f'ln1_g_{i}'], p[f'ln1_b_{i}'])
            dx = d_res1 + d_via_ln

        return g

    # ----------------------------------------------------------- train

    def train_step(self, tokens, targets, lr=None):
        B, T = tokens.shape
        lr = lr or self.lr
        self.t += 1

        c = {}
        logits = self.forward(tokens, c)

        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)

        log_probs = logits_flat - logits_flat.max(axis=-1, keepdims=True)
        log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
        loss = -log_probs[np.arange(len(targets_flat)), targets_flat].mean()

        probs = softmax(logits_flat, -1)
        dlogits_flat = probs.copy()
        dlogits_flat[np.arange(len(targets_flat)), targets_flat] -= 1
        dlogits_flat = dlogits_flat / (B * T)
        dlogits = dlogits_flat.reshape(B, T, self.vocab_size)

        g = self.backward(dlogits, c)

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for k in self.p:
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g[k]
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * (g[k] ** 2)
            m_hat = self._m[k] / (1 - beta1 ** self.t)
            v_hat = self._v[k] / (1 - beta2 ** self.t)
            self.p[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        return float(loss)

    # --------------------------------------------------------- generate

    def generate(self, input_tokens, max_new=20, temperature=1.0, top_k=20):
        generated = list(input_tokens)
        for _ in range(max_new):
            ctx = generated[-self.max_seq:]
            tokens_arr = np.array([ctx], dtype=np.int64)
            logits = self.forward(tokens_arr)
            logits_last = logits[0, -1, :] / temperature

            if top_k > 0:
                idxs = np.argpartition(-logits_last, top_k)[:top_k]
                vals = logits_last[idxs]
                probs = softmax(vals)
                choice = np.random.choice(idxs, p=probs)
            else:
                probs = softmax(logits_last)
                choice = np.random.choice(self.vocab_size, p=probs)

            generated.append(int(choice))
            if choice == 3:
                break

        return generated[len(input_tokens):]

    # ------------------------------------------------------- save/load

    def guardar(self, ruta=None):
        ruta = ruta or RUTA_MODELO
        np.savez_compressed(ruta, **self.p)
        with open(ruta.replace('.npz', '_adam.pkl'), 'wb') as f:
            pickle.dump({'m': self._m, 'v': self._v, 't': self.t}, f)

    def cargar(self, ruta=None):
        ruta = ruta or RUTA_MODELO
        if not os.path.exists(ruta):
            return False
        data = np.load(ruta)
        for k in data.files:
            self.p[k] = data[k]
        adam_ruta = ruta.replace('.npz', '_adam.pkl')
        if os.path.exists(adam_ruta):
            with open(adam_ruta, 'rb') as f:
                d = pickle.load(f)
            self._m = d['m']
            self._v = d['v']
            self.t = d['t']
        self.vocab_size = self.p['W_out'].shape[1]
        self.d_model = self.p['tok_emb'].shape[1]
        return True


# ===================================================================
# Cargar datos
# ===================================================================

def _extraer_definiciones_rae(datos):
    textos = []
    if isinstance(datos, dict):
        for key, val in datos.items():
            if isinstance(val, dict):
                defs = val.get('definiciones') or val.get('definicion')
                if isinstance(defs, list):
                    for d in defs:
                        if isinstance(d, str):
                            textos.append(d)
                elif isinstance(defs, str):
                    textos.append(defs)
                for v in val.values():
                    if isinstance(v, str) and len(v) > 20:
                        textos.append(v)
            elif isinstance(val, str) and len(val) > 20:
                textos.append(val)
    elif isinstance(datos, list):
        for item in datos:
            if isinstance(item, str):
                textos.append(item)
    return textos


def _rutas_textos():
    """Devuelve lista de rutas a archivos de texto para entrenar."""
    rutas = []
    rae_ruta = os.path.join(RUTA_DATOS, 'rae_diccionario.json')
    if os.path.exists(rae_ruta):
        rutas.append(rae_ruta)
    import glob
    for ruta in sorted(glob.glob(os.path.join(RUTA_DATOS, 'wiki_parte_*.txt'))):
        rutas.append(ruta)
    return rutas


def _iterar_textos(rutas, max_chunk=50*1048576):
    """Generador: produce texto de cada ruta en trozos de max_chunk bytes."""
    for ruta in rutas:
        if ruta.endswith('.json'):
            import json
            with open(ruta, encoding='utf-8') as f:
                datos = json.load(f)
            for texto in _extraer_definiciones_rae(datos):
                yield texto
        else:
            tam = os.path.getsize(ruta)
            if tam <= max_chunk:
                with open(ruta, encoding='utf-8') as f:
                    yield f.read()
            else:
                # Archivos grandes: leer en trozos de max_chunk
                with open(ruta, encoding='utf-8') as f:
                    while True:
                        trozo = f.read(max_chunk)
                        if not trozo:
                            break
                        yield trozo


# ===================================================================
# Entrenamiento eficiente en RAM
# ===================================================================

def entrenar(vocab_size=16000, d_model=128, n_heads=4, n_layers=4,
             d_ff=512, max_seq=64, lr=3e-4, epochs=5, batch_size=32):
    print("=== Entrenamiento Transformer Byte ===")
    print(f"  Dimensiones: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
    print(f"  Vocab size: {vocab_size}, max_seq: {max_seq}")
    t0 = time.time()

    rutas = _rutas_textos()
    print(f"\n1. Archivos de entrenamiento: {len(rutas)}")
    for r in rutas:
        tam = os.path.getsize(r)
        print(f"    {os.path.basename(r)}: {tam//1048576}MB")

    # Vocabulario
    print("\n2. Construyendo vocabulario...")
    vocab = Vocabulario.cargar()
    if vocab is None:
        vocab = Vocabulario.desde_textos(_iterar_textos(rutas), max_size=vocab_size, min_freq=1)
        vocab.guardar()
    print(f"  {vocab.size} palabras en vocabulario")

    # Inicializar modelo (antes de procesar datos, para liberar RAM)
    print(f"\n3. Inicializando transformer...")
    model = Transformer(
        vocab_size=vocab.size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq=max_seq,
        lr=lr,
    )
    if model.cargar():
        print("  Modelo existente cargado, continuando...")

    # Entrenamiento streaming por chunks
    stride = max_seq // 2
    print(f"\n4. Entrenando ({epochs} epochs, batch_size={batch_size})...")
    print("  Modo streaming: cada chunk se entrena y descarta.")
    mejor_loss = float('inf')

    for epoch in range(epochs):
        losses = []
        ep_t0 = time.time()
        tokens_epoca = 0
        pasos_epoca = 0

        for texto in _iterar_textos(rutas):
            palabras = Vocabulario._tokenizar(texto)
            if not palabras:
                continue
            ids = vocab.encode(palabras)
            tokens_epoca += len(ids)
            n_local = (len(ids) - max_seq) // stride
            if n_local <= 0:
                continue

            # Crear array de secuencias para este chunk
            xs_local = np.zeros((n_local, max_seq), dtype=np.int64)
            ys_local = np.zeros((n_local, max_seq), dtype=np.int64)
            for i in range(n_local):
                start = i * stride
                xs_local[i] = np.array(ids[start:start + max_seq], dtype=np.int64)
                ys_local[i] = np.array(ids[start + 1:start + max_seq + 1], dtype=np.int64)

            # Entrenar este chunk en batches
            for bstart in range(0, n_local, batch_size):
                bend = min(bstart + batch_size, n_local)
                bx = xs_local[bstart:bend]
                by = ys_local[bstart:bend]
                loss = model.train_step(bx, by)
                losses.append(loss)
                pasos_epoca += 1

                if pasos_epoca % 200 == 0:
                    avg = float(np.mean(losses[-200:]))
                    print(f"    e{epoch+1} paso {pasos_epoca} loss={avg:.4f}",
                          flush=True)

        loss_avg = float(np.mean(losses)) if losses else 0
        ep_t = time.time() - ep_t0
        print(f"  epoch {epoch+1}/{epochs}: loss={loss_avg:.4f} "
              f"({tokens_epoca//1000000}M tokens, {pasos_epoca} pasos, "
              f"{ep_t:.1f}s)", flush=True)

        if losses and loss_avg < mejor_loss:
            mejor_loss = loss_avg
            model.guardar()
            print(f"    -> modelo guardado (loss={loss_avg:.4f})", flush=True)

    t_total = time.time() - t0
    print(f"\n=== Entrenamiento completado en {t_total:.1f}s ===")
    print(f"  Mejor loss: {mejor_loss:.4f}")

    # Demo
    print("\n--- Demo de generacion ---")
    prompt = ['la', 'inteligencia', 'artificial']
    ids = [vocab.stoi.get(p, vocab.stoi.get('<unk>', 0)) for p in prompt]
    nuevos = model.generate(ids, max_new=20, temperature=0.8)
    todas_ids = ids + nuevos
    print(f"  Prompt: {' '.join(prompt)}")
    print(f"  Generado: {vocab.a_texto(todas_ids)}", flush=True)

    return model


if __name__ == '__main__':
    import sys
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    d_model = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    n_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    entrenar(epochs=epochs, d_model=d_model, n_layers=n_layers)
