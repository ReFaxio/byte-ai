"""Byte: entrenamiento con RAE + 1 wiki sample."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob, json, time
import numpy as np

from kernel.mamba import Mamba, Vocabulario, RUTA_DATOS, RUTA_VOCAB

RUTA_RAE = os.path.join(RUTA_DATOS, 'rae_diccionario.json')
RUTA_MODELO = os.path.join(RUTA_DATOS, 'mamba_model.npz')

def cargar_textos():
    textos = []
    if os.path.exists(RUTA_RAE):
        with open(RUTA_RAE, encoding='utf-8') as f:
            datos = json.load(f)
        texto_rae = []
        for k, v in datos.items():
            if isinstance(v, str):
                texto_rae.append(f"{k} {v}")
            elif isinstance(v, dict):
                for sub in v.values():
                    if isinstance(sub, str):
                        texto_rae.append(f"{k} {sub}")
        textos.append(' '.join(texto_rae))
    archivos = sorted(glob.glob(os.path.join(RUTA_DATOS, 'wiki_parte_*.txt')))
    if archivos:
        r = archivos[0]
        print(f"  + 1 wiki: {os.path.basename(r)} ({os.path.getsize(r)//1048576}MB)", flush=True)
        with open(r, encoding='utf-8') as f:
            textos.append(f.read())
    return textos

def main():
    t0 = time.time()
    print("=== Byte: Entrenamiento RAE + wiki ===")

    print("\n--- Cargando textos ---")
    textos = cargar_textos()
    print(f"  {len(textos)} textos, ~{sum(len(t) for t in textos)//1048576}MB total")

    print("\n--- Vocabulario ---")
    vocab = Vocabulario.cargar()
    if vocab is None:
        from collections import Counter
        c = Counter()
        for t in textos:
            for p in Vocabulario._tokenizar(t):
                c[p] += 1
        comunes = [p for p, _ in c.most_common(16000 - 4) if _ >= 1]
        vocab = Vocabulario(['<pad>', '<unk>', '<bos>', '<eos>'] + comunes)
        vocab.guardar()
    print(f"  {vocab.size} palabras")

    print("\n--- Tokenizando ---")
    todos_ids = []
    for t in textos:
        pals = Vocabulario._tokenizar(t)
        todos_ids.extend(vocab.encode(pals))
    ids = np.array(todos_ids, dtype=np.int32)
    del todos_ids
    print(f"  {len(ids)} tokens")

    print("\n--- Mamba ---")
    m = Mamba(vocab_size=vocab.size, d_model=256, d_state=128, d_ff=512)
    m.vocab = vocab
    # Forzar entrenar desde cero con nuevo tamaño
    if os.path.exists(RUTA_MODELO):
        print("  Eliminando modelo anterior para nuevo tamaño...", flush=True)
        os.remove(RUTA_MODELO)
        opt_pkl = RUTA_MODELO.replace('.npz', '_opt.pkl')
        if os.path.exists(opt_pkl): os.remove(opt_pkl)
    print("  Nuevo modelo (d_model=256, d_state=128, d_ff=512)")

    print("\n--- Entrenando ---")
    m.entrenar(ids, epochs=50, batch_size=256, max_seq=48)
    m.guardar()

    t = time.time() - t0
    print(f"\n=== Entrenamiento completado en {t:.1f}s ===")

    print("\n--- Demo ---")
    prompt = ['la', 'inteligencia', 'artificial']
    ids_prompt = [vocab.stoi.get(p, vocab.stoi.get('<unk>', 0)) for p in prompt]
    nuevos = m.generar(ids_prompt, max_new=30, temperature=0.7, top_k=0, top_p=0.9)
    print(f"  Prompt: {' '.join(prompt)}")
    print(f"  Byte:   {vocab.a_texto(ids_prompt + nuevos)}", flush=True)

if __name__ == '__main__':
    main()
