"""Mamba JAX: modelo SSM O(n) con GPU.
Entrena en segundos en GPU, corre en CPU para inferencia.
Sin atencion cuadratica, estado recurrente lineal."""

import os, json, re, time, pickle
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

RUTA_DATOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos')
RUTA_MODELO = os.path.join(RUTA_DATOS, 'mamba_model.npz')
RUTA_VOCAB = os.path.join(RUTA_DATOS, 'vocabulario.json')


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
    _tabla = str.maketrans('áéíóúüÁÉÍÓÚÜ', 'aeiouuAEIOUU')
    _re_no_word = re.compile(r'[^a-zñ ]')
    @staticmethod
    def _tokenizar(texto):
        if not texto: return []
        return Vocabulario._re_no_word.sub(' ', texto.lower().translate(Vocabulario._tabla)).split()
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


def init_params(vocab_size, d_model, d_state, d_ff, key):
    s = 0.02
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    return {
        'emb': jax.random.normal(k1, (vocab_size, d_model)) * s,
        'A': jax.random.normal(k2, (d_state, d_state)) * 0.01,
        'B': jax.random.normal(k3, (d_model, d_state)) * s,
        'C': jax.random.normal(k4, (d_state, d_model)) * s,
        'W_ffn1': jax.random.normal(k5, (d_model, d_ff)) * s,
        'b_ffn1': jnp.zeros(d_ff),
        'W_ffn2': jax.random.normal(k6, (d_ff, vocab_size)) * s,
        'b_ffn2': jnp.zeros(vocab_size),
    }


def ssm_step(h, A, B, C, x_t):
    h_new = h @ A.T + x_t @ B
    out = h_new @ C
    return h_new, out


def forward(params, tokens):
    B, T = tokens.shape
    x = params['emb'][tokens]
    h0 = jnp.zeros((B, params['A'].shape[0]))

    def scan_fn(h, x_t):
        h_new, out = ssm_step(h, params['A'], params['B'], params['C'], x_t)
        return h_new, out

    _, outs = jax.lax.scan(scan_fn, h0, x.transpose(1, 0, 2))

    out_last = outs[-1]
    h_f = jax.nn.relu(out_last @ params['W_ffn1'] + params['b_ffn1'])
    logits = h_f @ params['W_ffn2'] + params['b_ffn2']
    return logits


def loss_fn(params, tokens, targets):
    logits = forward(params, tokens)
    logits_f = logits - logits.max(axis=-1, keepdims=True)
    log_probs = logits_f - jnp.log(jnp.exp(logits_f).sum(axis=-1, keepdims=True))
    oh = jax.nn.one_hot(targets, logits.shape[-1])
    return -jnp.mean((log_probs * oh).sum(axis=-1))


@jax.jit
def train_step(params, opt_state, t, tokens, targets):
    loss, grads = jax.value_and_grad(loss_fn)(params, tokens, targets)
    lr = 3e-4
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    new_params = {}
    new_state = {'t': t + 1}
    for k in params:
        m = opt_state[f'm_{k}'] * beta1 + grads[k] * (1 - beta1)
        v = opt_state[f'v_{k}'] * beta2 + grads[k] ** 2 * (1 - beta2)
        m_h = m / (1 - beta1 ** (t + 1))
        v_h = v / (1 - beta2 ** (t + 1))
        new_params[k] = params[k] - lr * m_h / (jnp.sqrt(v_h) + eps)
        new_state[f'm_{k}'] = m
        new_state[f'v_{k}'] = v
    return new_params, new_state, loss


def create_opt_state(params):
    state = {'t': 0}
    for k in params:
        state[f'm_{k}'] = jnp.zeros_like(params[k])
        state[f'v_{k}'] = jnp.zeros_like(params[k])
    return state


class Mamba:
    def __init__(self, vocab_size=16000, d_model=128, d_state=64, d_ff=256):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_ff = d_ff
        self.vocab = None

        key = jax.random.PRNGKey(42)
        self.p = init_params(vocab_size, d_model, d_state, d_ff, key)
        self.opt = create_opt_state(self.p)
        self.t = 0

    def forward_np(self, tokens_idx):
        x = np.array(self.p['emb'])[tokens_idx]
        d_state = self.p['A'].shape[0]
        h = np.zeros((1, d_state))
        A = np.array(self.p['A'])
        B = np.array(self.p['B'])
        C = np.array(self.p['C'])
        for t in range(x.shape[1]):
            inp = x[:, t, :]
            h = h @ A.T + inp @ B
            out = h @ C
        logits = out @ np.array(self.p['W_ffn1']) + np.array(self.p['b_ffn1'])
        logits = np.maximum(0, logits)
        logits = logits @ np.array(self.p['W_ffn2']) + np.array(self.p['b_ffn2'])
        return logits

    def generar(self, input_tokens, max_new=20, temperature=0.8, top_k=20):
        gen = list(input_tokens)
        d_state = self.p['A'].shape[0]
        h = np.zeros((1, d_state))
        A = np.array(self.p['A'])
        B = np.array(self.p['B'])
        C = np.array(self.p['C'])
        W1 = np.array(self.p['W_ffn1'])
        b1 = np.array(self.p['b_ffn1'])
        W2 = np.array(self.p['W_ffn2'])
        b2 = np.array(self.p['b_ffn2'])
        emb = np.array(self.p['emb'])

        for _ in range(max_new):
            ctx = gen[-64:]
            x = emb[np.array([ctx])]
            for t in range(x.shape[1]):
                inp = x[:, t, :]
                h = h @ A.T + inp @ B
                out = h @ C
            logits = out @ W1 + b1
            logits = np.maximum(0, logits)
            logits = logits @ W2 + b2
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

    def entrenar(self, ids, epochs=2, batch_size=256):
        print("  Mamba JAX entrenando...", flush=True)
        max_seq = 16
        step = max_seq // 2
        N = (len(ids) - max_seq) // step
        x = np.zeros((N, max_seq), dtype=np.int32)
        y = np.zeros(N, dtype=np.int32)
        for i in range(N):
            off = i * step
            x[i] = ids[off:off + max_seq]
            y[i] = ids[off + max_seq]
        print(f"  Dataset: {N} muestras, entrenando {epochs} epochs...", flush=True)
        pasos = 0
        for epoch in range(epochs):
            losses = []
            idxs = np.random.permutation(N)
            for start in range(0, N, batch_size):
                batch_idx = idxs[start:start + batch_size]
                bx = jnp.array(x[batch_idx], dtype=jnp.int32)
                by = jnp.array(y[batch_idx], dtype=jnp.int32)
                self.p, self.opt, loss = train_step(self.p, self.opt, self.t, bx, by)
                self.t += 1
                pasos += 1
                losses.append(float(loss))
                if pasos % 100 == 0:
                    avg = float(np.mean(losses[-100:]))
                    print(f"    paso {pasos} loss={avg:.4f}", flush=True)
            if losses:
                print(f"  epoch {epoch+1}/{epochs} loss={np.mean(losses):.4f}", flush=True)

    def guardar(self, ruta=None):
        ruta = ruta or RUTA_MODELO
        p_np = {k: np.array(v) for k, v in self.p.items()}
        np.savez_compressed(ruta, **p_np)
        opt_np = {'t': self.t}
        for k in self.p:
            opt_np[f'm_{k}'] = np.array(self.opt[f'm_{k}'])
            opt_np[f'v_{k}'] = np.array(self.opt[f'v_{k}'])
        with open(ruta.replace('.npz', '_opt.pkl'), 'wb') as f:
            pickle.dump(opt_np, f)

    def cargar(self, ruta=None):
        ruta = ruta or RUTA_MODELO
        if not os.path.exists(ruta): return False
        d = np.load(ruta)
        self.p = {k: jnp.array(d[k]) for k in d.files}
        self.vocab_size = self.p['emb'].shape[0]
        self.d_model = self.p['emb'].shape[1]
        opt_ruta = ruta.replace('.npz', '_opt.pkl')
        if os.path.exists(opt_ruta):
            with open(opt_ruta, 'rb') as f:
                od = pickle.load(f)
            self.t = od.get('t', 0)
            self.opt = {'t': self.t}
            for k in self.p:
                self.opt[f'm_{k}'] = jnp.array(od.get(f'm_{k}', jnp.zeros_like(self.p[k])))
                self.opt[f'v_{k}'] = jnp.array(od.get(f'v_{k}', jnp.zeros_like(self.p[k])))
        return True


def _extraer_rae(datos):
    textos = []
    if isinstance(datos, dict):
        for k, v in datos.items():
            if isinstance(v, str): textos.append(f"{k} {v}")
            elif isinstance(v, dict):
                for sub in v.values():
                    if isinstance(sub, str): textos.append(f"{k} {sub}")
    return textos


def entrenar(rapido=False, epochs=2):
    print("=== Mamba JAX (GPU) ===")
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
                with open(r, encoding='utf-8') as f:
                    for t in _extraer_rae(json.load(f)): yield t
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

    print("  Dispositivo:", jax.devices()[0].platform, flush=True)

    m = Mamba(vocab_size=vocab.size)
    m.vocab = vocab
    if m.cargar():
        print("  Modelo cargado")

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
