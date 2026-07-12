"""Byte: entrenamiento streaming con todas las wikis."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob, json, time
from collections import Counter
import numpy as np

from kernel.mamba import Mamba, Vocabulario, RUTA_DATOS, RUTA_VOCAB

RUTA_RAE = os.path.join(RUTA_DATOS, 'rae_diccionario.json')
RUTA_MODELO = os.path.join(RUTA_DATOS, 'mamba_model.npz')
RUTA_CACHE = os.path.join(RUTA_DATOS, 'todos_ids.npy')

def tokenizar_y_cachear(vocab):
    if os.path.exists(RUTA_CACHE):
        print("  Cargando cache de tokens...", flush=True)
        return np.load(RUTA_CACHE)

    print("  Tokenizando corpus completo...", flush=True)
    todos = []
    if os.path.exists(RUTA_RAE):
        with open(RUTA_RAE, encoding='utf-8') as f:
            datos = json.load(f)
        texto_rae = []
        for k, v in datos.items():
            if isinstance(v, str): texto_rae.append(f"{k} {v}")
            elif isinstance(v, dict):
                for sub in v.values():
                    if isinstance(sub, str): texto_rae.append(f"{k} {sub}")
        texto = ' '.join(texto_rae)
        todos.extend(vocab.encode(Vocabulario._tokenizar(texto)))

    archivos = sorted(glob.glob(os.path.join(RUTA_DATOS, 'wiki_parte_*.txt')))
    for i, r in enumerate(archivos):
        if i % 25 == 0:
            print(f"  wiki {i}/{len(archivos)}...", flush=True)
        with open(r, encoding='utf-8') as f:
            texto = f.read()
        todos.extend(vocab.encode(Vocabulario._tokenizar(texto)))

    ids = np.array(todos, dtype=np.int32)
    del todos
    print(f"  Guardando {len(ids)} tokens en cache...", flush=True)
    np.save(RUTA_CACHE, ids)
    return ids

def construir_vocab():
    print("  Construyendo vocabulario...", flush=True)
    c = Counter()
    if os.path.exists(RUTA_RAE):
        with open(RUTA_RAE, encoding='utf-8') as f:
            datos = json.load(f)
        for k, v in datos.items():
            if isinstance(v, str):
                for p in Vocabulario._tokenizar(f"{k} {v}"): c[p] += 1
            elif isinstance(v, dict):
                for sub in v.values():
                    if isinstance(sub, str):
                        for p in Vocabulario._tokenizar(f"{k} {sub}"): c[p] += 1
    archivos = sorted(glob.glob(os.path.join(RUTA_DATOS, 'wiki_parte_*.txt')))
    for r in archivos[:5]:
        print(f"  Vocab: {os.path.basename(r)}", flush=True)
        with open(r, encoding='utf-8') as f:
            for p in Vocabulario._tokenizar(f.read()): c[p] += 1
    comunes = [p for p, _ in c.most_common(16000 - 4) if _ >= 1]
    vocab = Vocabulario(['<pad>', '<unk>', '<bos>', '<eos>'] + comunes)
    vocab.guardar()
    return vocab

def main():
    t0 = time.time()
    print("=== Byte: Entrenamiento streaming completo ===")

    print("\n--- Vocabulario ---")
    vocab = Vocabulario.cargar()
    if vocab is None:
        vocab = construir_vocab()
    print(f"  {vocab.size} palabras")

    print("\n--- Tokenizando ---")
    ids = tokenizar_y_cachear(vocab)
    print(f"  {len(ids)} tokens totales")

    print("\n--- Mamba ---")
    m = Mamba(vocab_size=vocab.size, d_model=256, d_state=128, d_ff=512)
    m.vocab = vocab
    if m.cargar():
        print("  Modelo cargado, continuando entrenamiento...")
    else:
        print("  Nuevo modelo")

    print("\n--- Entrenando ---")
    m.entrenar(ids, epochs=3, batch_size=256, max_seq=48)
    m.guardar()
    del ids

    t = time.time() - t0
    print(f"\n=== Completado en {t:.1f}s ===")

    print("\n--- Demo ---")
    prompt = ['la', 'inteligencia', 'artificial']
    ids_prompt = [vocab.stoi.get(p, vocab.stoi.get('<unk>', 0)) for p in prompt]
    nuevos = m.generar(ids_prompt, max_new=40, temperature=0.7, top_k=0, top_p=0.9)
    print(f"  Byte: {vocab.a_texto(ids_prompt + nuevos)}", flush=True)

if __name__ == '__main__':
    main()
