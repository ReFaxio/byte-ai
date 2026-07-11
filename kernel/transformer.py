"""Transformer con JAX — 0 if/else, 0 texto fijo.
Arquitectura: 4 capas, 4 cabezas, dim 128, ventana 64.
JAX autograd + JIT: 10-20x mas rapido que numpy puro."""

import jax
import jax.numpy as jnp
import numpy as np
import os, json, re, time, pickle
from collections import Counter

RUTA_DATOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos')
RUTA_MODELO = os.path.join(RUTA_DATOS, 'transformer_model.npz')
RUTA_VOCAB = os.path.join(RUTA_DATOS, 'vocabulario.json')

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
# Transformer — JAX puro
# ===================================================================

def crear_parametros(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq, key):
    """Crea todos los parametros del transformer.
    Devuelve un dict plano para facil guardado/carga."""
    s = 0.02
    params = {}
    key, sk = jax.random.split(key)
    params['tok_emb'] = jax.random.normal(sk, (vocab_size, d_model)) * s
    key, sk = jax.random.split(key)
    params['pos_emb'] = jax.random.normal(sk, (max_seq, d_model)) * s

    for i in range(n_layers):
        for w in ['W_q', 'W_k', 'W_v', 'W_o']:
            key, sk = jax.random.split(key)
            params[f'{w}_{i}'] = jax.random.normal(sk, (d_model, d_model)) * s
        params[f'ln1_g_{i}'] = jnp.ones(d_model)
        params[f'ln1_b_{i}'] = jnp.zeros(d_model)
        key, sk = jax.random.split(key)
        params[f'W_1_{i}'] = jax.random.normal(sk, (d_model, d_ff)) * s
        params[f'b_1_{i}'] = jnp.zeros(d_ff)
        key, sk = jax.random.split(key)
        params[f'W_2_{i}'] = jax.random.normal(sk, (d_ff, d_model)) * s
        params[f'b_2_{i}'] = jnp.zeros(d_model)
        params[f'ln2_g_{i}'] = jnp.ones(d_model)
        params[f'ln2_b_{i}'] = jnp.zeros(d_model)

    params['ln_f_g'] = jnp.ones(d_model)
    params['ln_f_b'] = jnp.zeros(d_model)
    key, sk = jax.random.split(key)
    params['W_out'] = jax.random.normal(sk, (d_model, vocab_size)) * s
    params['b_out'] = jnp.zeros(vocab_size)
    return params, key


def forward(params, tokens, mask):
    """Forward pass puro JAX. Sin efectos secundarios."""
    d_model = params['tok_emb'].shape[1]
    n_layers = sum(1 for k in params if k.startswith('W_q_'))
    n_heads = 4
    d_k = d_model // n_heads
    B, T = tokens.shape

    x = params['tok_emb'][tokens] + params['pos_emb'][:T]

    for i in range(n_layers):
        x_ln = (x - x.mean(axis=-1, keepdims=True)) / jnp.sqrt(x.var(axis=-1, keepdims=True) + 1e-5) * params[f'ln1_g_{i}'] + params[f'ln1_b_{i}']

        q = x_ln @ params[f'W_q_{i}']
        k = x_ln @ params[f'W_k_{i}']
        v = x_ln @ params[f'W_v_{i}']

        q_h = q.reshape(B, T, n_heads, d_k).transpose(0, 2, 1, 3)
        k_h = k.reshape(B, T, n_heads, d_k).transpose(0, 2, 1, 3)
        v_h = v.reshape(B, T, n_heads, d_k).transpose(0, 2, 1, 3)

        s = q_h @ k_h.transpose(0, 1, 3, 2) / jnp.sqrt(d_k)
        s = s + mask[:T, :T]
        a = jax.nn.softmax(s, axis=-1)

        o_h = a @ v_h
        o = o_h.transpose(0, 2, 1, 3).reshape(B, T, d_model)

        o = o @ params[f'W_o_{i}']
        x = x + o

        x_ln = (x - x.mean(axis=-1, keepdims=True)) / jnp.sqrt(x.var(axis=-1, keepdims=True) + 1e-5) * params[f'ln2_g_{i}'] + params[f'ln2_b_{i}']
        h = jax.nn.relu(x_ln @ params[f'W_1_{i}'] + params[f'b_1_{i}'])
        o = h @ params[f'W_2_{i}'] + params[f'b_2_{i}']
        x = x + o

    x = (x - x.mean(axis=-1, keepdims=True)) / jnp.sqrt(x.var(axis=-1, keepdims=True) + 1e-5) * params['ln_f_g'] + params['ln_f_b']
    logits = x @ params['W_out'] + params['b_out']
    return logits


def loss_fn(params, tokens, targets, mask):
    logits = forward(params, tokens, mask)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    logits_flat = logits_flat - logits_flat.max(axis=-1, keepdims=True)
    log_probs = logits_flat - jnp.log(jnp.exp(logits_flat).sum(axis=-1, keepdims=True))
    oh = jax.nn.one_hot(targets_flat, logits.shape[-1])
    return -jnp.mean((log_probs * oh).sum(axis=-1))


@jax.jit
def train_step(params, opt_state, tokens, targets, mask):
    """Un paso de entrenamiento: loss + grad + actualizar Adam."""
    loss, grads = jax.value_and_grad(loss_fn)(params, tokens, targets, mask)
    updates, opt_state = optim.update(grads, opt_state)
    params = optim.apply_updates(params, updates)
    return params, opt_state, loss


# ===================================================================
# Optimizer Adam (simple, sobre dicts)
# ===================================================================

class OptimAdam:
    def __init__(self, lr=3e-4):
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def init(self, params):
        state = {'t': 0}
        for k in params:
            state[f'm_{k}'] = jnp.zeros_like(params[k])
            state[f'v_{k}'] = jnp.zeros_like(params[k])
        return state

    def update(self, grads, state):
        state['t'] += 1
        t = state['t']
        new_params = {}
        new_state = {'t': t}
        for k in grads:
            m = self.beta1 * state[f'm_{k}'] + (1 - self.beta1) * grads[k]
            v = self.beta2 * state[f'v_{k}'] + (1 - self.beta2) * grads[k] ** 2
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            new_params[k] = state.get(k, grads[k]) - self.lr * m_hat / (jnp.sqrt(v_hat) + self.eps)
            new_state[f'm_{k}'] = m
            new_state[f'v_{k}'] = v
        return new_params, new_state

    def apply_updates(self, params, updates):
        return {k: updates[k] for k in params}


# Instancia global
optim = OptimAdam()


# ===================================================================
# Clase Transformer (wrapper para compatibilidad)
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

        key = jax.random.PRNGKey(42)
        self.p, _ = crear_parametros(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq, key)
        self._mask = jnp.triu(jnp.full((max_seq, max_seq), -jnp.inf), 1)
        self._opt_state = optim.init(self.p)
        self.t = 0

    def train_step(self, tokens, targets):
        self.t += 1
        tokens_j = jnp.array(tokens)
        targets_j = jnp.array(targets)
        self.p, self._opt_state, loss = train_step(
            self.p, self._opt_state, tokens_j, targets_j, self._mask)
        return float(loss)

    def generate(self, input_tokens, max_new=20, temperature=1.0, top_k=20):
        generated = list(input_tokens)
        for _ in range(max_new):
            ctx = generated[-self.max_seq:]
            tokens_arr = jnp.array([ctx], dtype=jnp.int32)
            logits = forward(self.p, tokens_arr, self._mask)
            logits_last = np.array(logits[0, -1, :] / temperature)

            if top_k > 0:
                idxs = np.argpartition(-logits_last, top_k)[:top_k]
                vals = logits_last[idxs]
                probs = np.exp(vals - vals.max()) / np.exp(vals - vals.max()).sum()
                choice = int(np.random.choice(idxs, p=probs))
            else:
                probs = np.exp(logits_last - logits_last.max())
                probs /= probs.sum()
                choice = int(np.random.choice(self.vocab_size, p=probs))

            generated.append(choice)
            if choice == 3:
                break

        return generated[len(input_tokens):]

    def guardar(self, ruta=None):
        ruta = ruta or RUTA_MODELO
        p_np = {k: np.array(v) for k, v in self.p.items()}
        np.savez_compressed(ruta, **p_np)
        opt_np = {}
        for k, v in self._opt_state.items():
            if isinstance(v, jax.Array):
                opt_np[k] = np.array(v)
            else:
                opt_np[k] = v
        with open(ruta.replace('.npz', '_adam.pkl'), 'wb') as f:
            pickle.dump(opt_np, f)

    def cargar(self, ruta=None):
        ruta = ruta or RUTA_MODELO
        if not os.path.exists(ruta):
            return False
        data = np.load(ruta)
        self.p = {k: jnp.array(data[k]) for k in data.files}
        adam_ruta = ruta.replace('.npz', '_adam.pkl')
        if os.path.exists(adam_ruta):
            with open(adam_ruta, 'rb') as f:
                d = pickle.load(f)
            self._opt_state = d
            self.t = d.get('t', 0)
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


def _rutas_textos(rapido=False):
    rutas = []
    rae_ruta = os.path.join(RUTA_DATOS, 'rae_diccionario.json')
    if os.path.exists(rae_ruta):
        rutas.append(rae_ruta)
    import glob
    for ruta in sorted(glob.glob(os.path.join(RUTA_DATOS, 'wiki_parte_*.txt'))):
        if rapido and os.path.getsize(ruta) > 100 * 1048576:
            print(f"  Omitido (rapido): {os.path.basename(ruta)} ({os.path.getsize(ruta)//1048576}MB)")
            continue
        rutas.append(ruta)
    return rutas


def _iterar_textos(rutas, max_chunk=50*1048576):
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
                with open(ruta, encoding='utf-8') as f:
                    while True:
                        trozo = f.read(max_chunk)
                        if not trozo:
                            break
                        yield trozo


# ===================================================================
# Entrenamiento
# ===================================================================

def entrenar(vocab_size=16000, d_model=128, n_heads=4, n_layers=4,
             d_ff=512, max_seq=64, lr=3e-4, epochs=5, batch_size=32, rapido=False):
    print("=== Entrenamiento Transformer Byte (JAX) ===")
    print(f"  Dimensiones: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
    print(f"  Vocab size: {vocab_size}, max_seq: {max_seq}, batch_size={batch_size}")
    if rapido:
        print("  Modo rapido: solo archivos <100MB")
    t0 = time.time()

    rutas = _rutas_textos(rapido)
    print(f"\n1. Archivos de entrenamiento: {len(rutas)}")
    for r in rutas:
        tam = os.path.getsize(r)
        print(f"    {os.path.basename(r)}: {tam//1048576}MB")

    print("\n2. Construyendo vocabulario...")
    vocab = Vocabulario.cargar()
    if vocab is None:
        vocab = Vocabulario.desde_textos(_iterar_textos(rutas), max_size=vocab_size, min_freq=1)
        vocab.guardar()
    print(f"  {vocab.size} palabras en vocabulario")

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

    stride = max_seq // 2
    print(f"\n4. Entrenando ({epochs} epochs, stride={stride}, batch_size={batch_size})...")
    print("  Modo streaming: cada chunk se entrena y descarta.")
    mejor_loss = float('inf')

    if rapido:
        print("  Iniciando JIT compilation...")

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

            xs_local = np.zeros((n_local, max_seq), dtype=np.int64)
            ys_local = np.zeros((n_local, max_seq), dtype=np.int64)
            for i in range(n_local):
                start = i * stride
                xs_local[i] = np.array(ids[start:start + max_seq], dtype=np.int64)
                ys_local[i] = np.array(ids[start + 1:start + max_seq + 1], dtype=np.int64)

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
    rapido = '--rapido' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    epochs = int(args[0]) if len(args) > 0 else 5
    d_model = int(args[1]) if len(args) > 1 else 128
    n_layers = int(args[2]) if len(args) > 2 else 4
    entrenar(epochs=epochs, d_model=d_model, n_layers=n_layers, rapido=rapido)
