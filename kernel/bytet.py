"""ByteT — Transformer hibrido atencion local-ventana + FFN.
0 if/else, 0 texto fijo.
Atencion solo a ventana local de 8 posiciones (no O(n^2)).
Entrena ~8x mas rapido que transformer completo."""

import jax
import jax.numpy as jnp
import numpy as np
import os, json, re, time, pickle
from collections import Counter

RUTA_DATOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos')
RUTA_MODELO = os.path.join(RUTA_DATOS, 'bytet_model.npz')
RUTA_VOCAB = os.path.join(RUTA_DATOS, 'vocabulario.json')

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
            for pal in Vocabulario._tokenizar(texto):
                c[pal] += 1
        comunes = [p for p, _ in c.most_common(max_size - 4) if _ >= min_freq]
        return cls(['<pad>', '<unk>', '<bos>', '<eos>'] + comunes)
    @staticmethod
    def _tokenizar(texto):
        if not texto: return []
        texto = texto.lower()
        for a,b in [('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u'),('ü','u')]:
            texto = texto.replace(a,b)
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


def _extraer_definiciones_rae(datos):
    textos = []
    if isinstance(datos, dict):
        for key, val in datos.items():
            if isinstance(val, dict):
                defs = val.get('definiciones') or val.get('definicion')
                if isinstance(defs, list):
                    for d in defs:
                        if isinstance(d, str): textos.append(d)
                elif isinstance(defs, str): textos.append(defs)
                for v in val.values():
                    if isinstance(v, str) and len(v) > 20: textos.append(v)
            elif isinstance(val, str) and len(val) > 20: textos.append(val)
    elif isinstance(datos, list):
        for item in datos:
            if isinstance(item, str): textos.append(item)
    return textos

def _rutas_textos(rapido=False):
    rutas = []
    rae_ruta = os.path.join(RUTA_DATOS, 'rae_diccionario.json')
    if os.path.exists(rae_ruta): rutas.append(rae_ruta)
    import glob
    for ruta in sorted(glob.glob(os.path.join(RUTA_DATOS, 'wiki_parte_*.txt'))):
        if rapido and os.path.getsize(ruta) > 100 * 1048576:
            print(f"  Omitido: {os.path.basename(ruta)}")
            continue
        rutas.append(ruta)
    return rutas

def _iterar_textos(rutas, max_chunk=50*1048576):
    for ruta in rutas:
        if ruta.endswith('.json'):
            import json
            with open(ruta, encoding='utf-8') as f: datos = json.load(f)
            for texto in _extraer_definiciones_rae(datos): yield texto
        else:
            tam = os.path.getsize(ruta)
            if tam <= max_chunk:
                with open(ruta, encoding='utf-8') as f: yield f.read()
            else:
                with open(ruta, encoding='utf-8') as f:
                    while True:
                        trozo = f.read(max_chunk)
                        if not trozo: break
                        yield trozo


def crear_mascara_local(max_seq, ventana):
    mask = jnp.full((max_seq, max_seq), -jnp.inf)
    for i in range(max_seq):
        izq = max(0, i - ventana)
        der = min(max_seq, i + ventana + 1)
        mask = mask.at[i, izq:der].set(0.0)
    return jnp.triu(mask, 1)


@jax.jit
def _train_step_jit(params, m, v, t, lr, tokens, targets):
    def loss_fn(p):
        B, T = tokens.shape
        d_model = params['tok_emb'].shape[1]
        dim_k = d_model // 4
        n_layers = max(int(k.split('_')[-1]) for k in params if k.startswith('W_q_')) + 1 if any(k.startswith('W_q_') for k in params) else 4
        x = p['tok_emb'][tokens] + p['pos_emb'][:T]
        n_layers = len([k for k in params if k.startswith('W_q_')])
        for i in range(n_layers):
            x_ln = (x - x.mean(axis=-1, keepdims=True)) / jnp.sqrt(x.var(axis=-1, keepdims=True) + 1e-5) * p[f'ln1_g_{i}'] + p[f'ln1_b_{i}']
            q = x_ln @ p[f'W_q_{i}']
            k = x_ln @ p[f'W_k_{i}']
            v = x_ln @ p[f'W_v_{i}']
            q_h = q.reshape(B,T,4,dim_k).transpose(0,2,1,3)
            k_h = k.reshape(B,T,4,dim_k).transpose(0,2,1,3)
            v_h = v.reshape(B,T,4,dim_k).transpose(0,2,1,3)
            s = q_h @ k_h.transpose(0,1,3,2) / jnp.sqrt(dim_k)
            mask = jnp.triu(jnp.full((T,T), -jnp.inf), 1)
            s = s + mask
            a = jax.nn.softmax(s, axis=-1)
            o_h = a @ v_h
            o = o_h.transpose(0,2,1,3).reshape(B,T,d_model)
            x = x + o @ p[f'W_o_{i}']
            x_ln = (x - x.mean(axis=-1, keepdims=True)) / jnp.sqrt(x.var(axis=-1, keepdims=True) + 1e-5) * p[f'ln2_g_{i}'] + p[f'ln2_b_{i}']
            x = x + jax.nn.relu(x_ln @ p[f'W_1_{i}'] + p[f'b_1_{i}']) @ p[f'W_2_{i}'] + p[f'b_2_{i}']
        x = (x - x.mean(axis=-1, keepdims=True)) / jnp.sqrt(x.var(axis=-1, keepdims=True) + 1e-5) * p['ln_f_g'] + p['ln_f_b']
        logits = x @ p['W_out'] + p['b_out']
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        logits_flat = logits_flat - logits_flat.max(axis=-1, keepdims=True)
        log_probs = logits_flat - jnp.log(jnp.exp(logits_flat).sum(axis=-1, keepdims=True))
        oh = jax.nn.one_hot(targets_flat, logits.shape[-1])
        return -jnp.mean((log_probs * oh).sum(axis=-1))
    loss, grads = jax.value_and_grad(loss_fn)(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    new_params = {}
    new_m = {}
    new_v = {}
    for k in params:
        new_m_k = beta1 * m[k] + (1 - beta1) * grads[k]
        new_v_k = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
        m_hat = new_m_k / (1 - beta1 ** t)
        v_hat = new_v_k / (1 - beta2 ** t)
        new_params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        new_m[k] = new_m_k
        new_v[k] = new_v_k
    return new_params, new_m, new_v, loss


class ByteT:
    def __init__(self, vocab_size=16000, d_model=128, n_layers=4,
                 d_ff=512, max_seq=64, ventana=8, lr=3e-4):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.ventana = ventana
        self.d_k = d_model // 4
        self.lr = lr
        self.t = 0

        key = jax.random.PRNGKey(42)
        self.p = {}
        self._init_params(key)
        self._m, self._v = {}, {}
        for k in self.p:
            self._m[k] = jnp.zeros_like(self.p[k])
            self._v[k] = jnp.zeros_like(self.p[k])
        self._mask = crear_mascara_local(max_seq, ventana)
        self.vocab = None

    def _init_params(self, key):
        s = 0.02
        p = self.p
        sk, sub = jax.random.split(key)
        p['tok_emb'] = jax.random.normal(sub, (self.vocab_size, self.d_model)) * s
        sk, sub = jax.random.split(sk)
        p['pos_emb'] = jax.random.normal(sub, (self.max_seq, self.d_model)) * s
        for i in range(self.n_layers):
            for w in ['W_q','W_k','W_v','W_o']:
                sk, sub = jax.random.split(sk)
                p[f'{w}_{i}'] = jax.random.normal(sub, (self.d_model, self.d_model)) * s
            p[f'ln1_g_{i}'] = jnp.ones(self.d_model)
            p[f'ln1_b_{i}'] = jnp.zeros(self.d_model)
            sk, sub = jax.random.split(sk)
            p[f'W_1_{i}'] = jax.random.normal(sub, (self.d_model, self.d_ff)) * s
            p[f'b_1_{i}'] = jnp.zeros(self.d_ff)
            sk, sub = jax.random.split(sk)
            p[f'W_2_{i}'] = jax.random.normal(sub, (self.d_ff, self.d_model)) * s
            p[f'b_2_{i}'] = jnp.zeros(self.d_model)
            p[f'ln2_g_{i}'] = jnp.ones(self.d_model)
            p[f'ln2_b_{i}'] = jnp.zeros(self.d_model)
        p['ln_f_g'] = jnp.ones(self.d_model)
        p['ln_f_b'] = jnp.zeros(self.d_model)
        sk, sub = jax.random.split(sk)
        p['W_out'] = jax.random.normal(sub, (self.d_model, self.vocab_size)) * s
        p['b_out'] = jnp.zeros(self.vocab_size)

    def forward(self, tokens, params=None):
        if params is None: params = self.p
        B, T = tokens.shape
        d_model = self.d_model
        dim_k = self.d_k
        x = params['tok_emb'][tokens] + params['pos_emb'][:T]
        for i in range(self.n_layers):
            x_ln = (x - x.mean(axis=-1, keepdims=True)) / jnp.sqrt(x.var(axis=-1, keepdims=True) + 1e-5) * params[f'ln1_g_{i}'] + params[f'ln1_b_{i}']
            q = x_ln @ params[f'W_q_{i}']
            k = x_ln @ params[f'W_k_{i}']
            v = x_ln @ params[f'W_v_{i}']
            q_h = q.reshape(B,T,4,dim_k).transpose(0,2,1,3)
            k_h = k.reshape(B,T,4,dim_k).transpose(0,2,1,3)
            v_h = v.reshape(B,T,4,dim_k).transpose(0,2,1,3)
            s = q_h @ k_h.transpose(0,1,3,2) / jnp.sqrt(dim_k)
            s = s + self._mask[:T,:T]
            a = jax.nn.softmax(s, axis=-1)
            o_h = a @ v_h
            o = o_h.transpose(0,2,1,3).reshape(B,T,d_model)
            x = x + o @ params[f'W_o_{i}']
            x_ln = (x - x.mean(axis=-1, keepdims=True)) / jnp.sqrt(x.var(axis=-1, keepdims=True) + 1e-5) * params[f'ln2_g_{i}'] + params[f'ln2_b_{i}']
            x = x + jax.nn.relu(x_ln @ params[f'W_1_{i}'] + params[f'b_1_{i}']) @ params[f'W_2_{i}'] + params[f'b_2_{i}']
        x = (x - x.mean(axis=-1, keepdims=True)) / jnp.sqrt(x.var(axis=-1, keepdims=True) + 1e-5) * params['ln_f_g'] + params['ln_f_b']
        return x @ params['W_out'] + params['b_out']

    def loss_fn(self, params, tokens, targets):
        logits = self.forward(tokens, params)
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)
        logits_flat = logits_flat - logits_flat.max(axis=-1, keepdims=True)
        log_probs = logits_flat - jnp.log(jnp.exp(logits_flat).sum(axis=-1, keepdims=True))
        oh = jax.nn.one_hot(targets_flat, self.vocab_size)
        return -jnp.mean((log_probs * oh).sum(axis=-1))

    def train_step(self, tokens, targets):
        self.t += 1
        tj = jnp.array(tokens)
        ty = jnp.array(targets)
        new_p, new_m, new_v, loss = _train_step_jit(
            self.p, self._m, self._v, self.t, self.lr, tj, ty)
        self.p = new_p
        self._m = new_m
        self._v = new_v
        return float(loss)

    def generate(self, input_tokens, max_new=20, temperature=1.0, top_k=20):
        generated = list(input_tokens)
        for _ in range(max_new):
            ctx = generated[-self.max_seq:]
            tokens_arr = jnp.array([ctx], dtype=jnp.int32)
            logits = self.forward(tokens_arr)
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
            if choice == 3: break
        return generated[len(input_tokens):]

    def guardar(self, ruta=None):
        ruta = ruta or RUTA_MODELO
        p_np = {k: np.array(v) for k,v in self.p.items()}
        np.savez_compressed(ruta, **p_np)
        opt = {'t': self.t}
        for k in self.p:
            opt[f'm_{k}'] = np.array(self._m[k])
            opt[f'v_{k}'] = np.array(self._v[k])
        with open(ruta.replace('.npz', '_adam.pkl'), 'wb') as f:
            pickle.dump(opt, f)

    def cargar(self, ruta=None):
        ruta = ruta or RUTA_MODELO
        if not os.path.exists(ruta): return False
        data = np.load(ruta)
        self.p = {k: jnp.array(data[k]) for k in data.files}
        adam_ruta = ruta.replace('.npz', '_adam.pkl')
        if os.path.exists(adam_ruta):
            with open(adam_ruta, 'rb') as f: d = pickle.load(f)
            self.t = d.get('t', 0)
            for k in self.p:
                self._m[k] = jnp.array(d.get(f'm_{k}', np.zeros(self.p[k].shape)))
                self._v[k] = jnp.array(d.get(f'v_{k}', np.zeros(self.p[k].shape)))
        self.vocab_size = self.p['W_out'].shape[1]
        self.d_model = self.p['tok_emb'].shape[1]
        return True


def entrenar(vocab_size=16000, d_model=128, n_layers=4,
             d_ff=512, max_seq=64, ventana=8, lr=3e-4, epochs=5,
             batch_size=32, rapido=False):
    print("=== ByteT (ventana local) ===")
    print(f"  d_model={d_model}, n_layers={n_layers}, ventana={ventana}")
    t0 = time.time()

    rutas = _rutas_textos(rapido)
    print(f"\nArchivos: {len(rutas)}")
    for r in rutas: print(f"  {os.path.basename(r)}: {os.path.getsize(r)//1048576}MB")

    print("\nVocabulario...")
    vocab = Vocabulario.cargar()
    if vocab is None:
        vocab = Vocabulario.desde_textos(_iterar_textos(rutas), max_size=vocab_size, min_freq=1)
        vocab.guardar()
    print(f"  {vocab.size} palabras")

    print("\nByteT...")
    model = ByteT(vocab_size=vocab.size, d_model=d_model, n_layers=n_layers,
                  d_ff=d_ff, max_seq=max_seq, ventana=ventana, lr=lr)
    model.vocab = vocab
    if model.cargar():
        print("  Modelo cargado")

    stride = max_seq // 2
    print(f"\nEntrenando ({epochs} epochs, batch={batch_size})...")
    mejor_loss = float('inf')

    for epoch in range(epochs):
        losses = []
        ep_t0 = time.time()
        tokens_epoca = 0
        pasos_epoca = 0

        for texto in _iterar_textos(rutas):
            palabras = Vocabulario._tokenizar(texto)
            if not palabras: continue
            ids = vocab.encode(palabras)
            tokens_epoca += len(ids)
            n_local = (len(ids) - max_seq) // stride
            if n_local <= 0: continue

            xs_local = np.zeros((n_local, max_seq), dtype=np.int64)
            ys_local = np.zeros((n_local, max_seq), dtype=np.int64)
            for i in range(n_local):
                start = i * stride
                xs_local[i] = np.array(ids[start:start+max_seq], dtype=np.int64)
                ys_local[i] = np.array(ids[start+1:start+max_seq+1], dtype=np.int64)

            for bstart in range(0, n_local, batch_size):
                bend = min(bstart + batch_size, n_local)
                bx = xs_local[bstart:bend]
                by = ys_local[bstart:bend]
                loss = model.train_step(bx, by)
                losses.append(loss)
                pasos_epoca += 1
                if pasos_epoca % 200 == 0:
                    avg = float(np.mean(losses[-200:]))
                    print(f"  e{epoch+1} paso {pasos_epoca} loss={avg:.4f}", flush=True)

        loss_avg = float(np.mean(losses)) if losses else 0
        ep_t = time.time() - ep_t0
        print(f"  epoch {epoch+1}/{epochs}: loss={loss_avg:.4f} "
              f"({tokens_epoca//1000000}M tokens, {pasos_epoca} pasos, {ep_t:.1f}s)", flush=True)
        if losses and loss_avg < mejor_loss:
            mejor_loss = loss_avg
            model.guardar()
            print(f"  -> guardado ({loss_avg:.4f})", flush=True)

    print(f"\nCompletado en {time.time()-t0:.1f}s, mejor loss: {mejor_loss:.4f}")

    prompt = ['la', 'inteligencia', 'artificial']
    ids = [vocab.stoi.get(p, vocab.stoi.get('<unk>',0)) for p in prompt]
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
    ventana = int(args[3]) if len(args) > 3 else 8
    entrenar(epochs=epochs, d_model=d_model, n_layers=n_layers,
             max_seq=64, ventana=ventana, rapido=rapido)
