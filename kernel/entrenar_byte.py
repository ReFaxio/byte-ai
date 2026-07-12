"""Byte: entrenamiento completo con Mamba + Wikipedia + RAE.
Pre-tokeniza una vez, cachea en .npy, entrena desde cache."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob, json, time, re
from collections import Counter
import numpy as np

from kernel.mamba import Mamba, Vocabulario, RUTA_DATOS, RUTA_VOCAB

RUTA_RAE = os.path.join(RUTA_DATOS, 'rae_diccionario.json')
RUTA_MODELO = os.path.join(RUTA_DATOS, 'mamba_model.npz')
RUTA_TOKENS = os.path.join(RUTA_DATOS, 'tokens_cache.npy')
RUTA_META = os.path.join(RUTA_DATOS, 'tokens_meta.json')

def cachear_tokens(vocab):
    if os.path.exists(RUTA_TOKENS):
        print("  Cache de tokens existe, cargando...", flush=True)
        return np.load(RUTA_TOKENS)

    print("  Tokenizando todo el corpus...", flush=True)
    todos_ids = []
    total_chars = 0

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
        pals = Vocabulario._tokenizar(texto)
        todos_ids.extend(vocab.encode(pals))
        total_chars += len(texto)

    archivos = sorted(glob.glob(os.path.join(RUTA_DATOS, 'wiki_parte_*.txt')))
    for i, r in enumerate(archivos):
        if i % 50 == 0:
            print(f"  Tokenizando wiki {i}/{len(archivos)}... ({total_chars//1048576}MB chars)", flush=True)
        with open(r, encoding='utf-8') as f:
            texto = f.read()
        pals = Vocabulario._tokenizar(texto)
        todos_ids.extend(vocab.encode(pals))
        total_chars += len(texto)

    ids = np.array(todos_ids, dtype=np.int32)
    del todos_ids
    print(f"  Tokenizado: {len(ids)} tokens, guardando en {RUTA_TOKENS}...", flush=True)
    np.save(RUTA_TOKENS, ids)
    with open(RUTA_META, 'w') as f:
        json.dump({'total_tokens': int(len(ids)), 'total_chars': total_chars}, f)
    return ids

def main():
    t0 = time.time()
    print("=== Byte: Entrenamiento completo ===")

    if not os.path.exists(RUTA_RAE) and not glob.glob(os.path.join(RUTA_DATOS, 'wiki_parte_*.txt')):
        print("ERROR: No hay datos. Descarga Wikipedia primero con:")
        print("  python kernel/descargar_wikipedia.py")
        return

    print("\n--- Vocabulario ---")
    vocab = Vocabulario.cargar()
    if vocab is None:
        temp_pals = set()
        if os.path.exists(RUTA_RAE):
            with open(RUTA_RAE, encoding='utf-8') as f:
                for k, v in json.load(f).items():
                    if isinstance(v, str): temp_pals.update(Vocabulario._tokenizar(f"{k} {v}"))
                    elif isinstance(v, dict):
                        for sub in v.values():
                            if isinstance(sub, str): temp_pals.update(Vocabulario._tokenizar(f"{k} {sub}"))
        archivos = sorted(glob.glob(os.path.join(RUTA_DATOS, 'wiki_parte_*.txt')))
        if archivos:
            with open(archivos[0], encoding='utf-8') as f:
                texto = f.read()
            temp_pals.update(Vocabulario._tokenizar(texto))
        c = Counter(temp_pals)
        comunes = [p for p,_ in c.most_common(16000-4) if _ >= 1]
        vocab = Vocabulario(['<pad>','<unk>','<bos>','<eos>'] + comunes)
        vocab.guardar()
    print(f"  {vocab.size} palabras")

    print("\n--- Cache de tokens ---")
    ids = cachear_tokens(vocab)

    print("\n--- Mamba ---")
    m = Mamba(vocab_size=vocab.size)
    m.vocab = vocab
    if m.cargar():
        print("  Modelo existente cargado")
    else:
        print("  Nuevo modelo")

    print("\n--- Entrenando ---")
    m.entrenar_desde_ids(ids, epochs=2)
    m.guardar()
    del ids

    t = time.time() - t0
    print(f"\n=== Entrenamiento completado en {t:.1f}s ===")

    print("\n--- Demo ---")
    prompt = ['la', 'inteligencia', 'artificial']
    ids_prompt = [vocab.stoi.get(p, vocab.stoi.get('<unk>', 0)) for p in prompt]
    nuevos = m.generar(ids_prompt, max_new=30, temperature=0.8)
    print(f"  Prompt: {' '.join(prompt)}")
    print(f"  Byte:   {vocab.a_texto(ids_prompt + nuevos)}", flush=True)

if __name__ == '__main__':
    main()
