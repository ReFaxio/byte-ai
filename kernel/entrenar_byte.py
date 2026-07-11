"""Byte: entrenamiento completo con Mamba + Wikipedia + RAE.
Corre este script cuando la descarga de Wikipedia termine.
Un solo comando: python kernel/entrenar_byte.py"""

import sys, os, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from kernel.mamba import Mamba, Vocabulario, RUTA_DATOS, RUTA_VOCAB

RUTA_RAE = os.path.join(RUTA_DATOS, 'rae_diccionario.json')
RUTA_MODELO = os.path.join(RUTA_DATOS, 'mamba_model.npz')

def iterar_textos():
    if os.path.exists(RUTA_RAE):
        with open(RUTA_RAE, encoding='utf-8') as f:
            datos = json.load(f)
        for k, v in datos.items():
            if isinstance(v, str): yield f"{k} {v}"
            elif isinstance(v, dict):
                for sub in v.values():
                    if isinstance(sub, str): yield f"{k} {sub}"
    for r in sorted(glob.glob(os.path.join(RUTA_DATOS, 'wiki_parte_*.txt'))):
        if os.path.getsize(r) > 100e6:
            print(f"  Omitido archivo grande: {os.path.basename(r)}")
            continue
        print(f"  Leyendo {os.path.basename(r)}...", flush=True)
        with open(r, encoding='utf-8') as f:
            while True:
                chunk = f.read(10*1048576)
                if not chunk: break
                yield chunk

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
        vocab = Vocabulario.desde_textos(iterar_textos(), max_size=16000, min_freq=1)
        vocab.guardar()
    print(f"  {vocab.size} palabras")

    print("\n--- Mamba ---")
    m = Mamba(vocab_size=vocab.size)
    m.vocab = vocab
    if m.cargar():
        print("  Modelo existente cargado")
    else:
        print("  Nuevo modelo")

    print("\n--- Entrenando ---")
    m.entrenar(iterar_textos(), epochs=2, lr=0.001)
    m.guardar()

    t = time.time() - t0
    print(f"\n=== Entrenamiento completado en {t:.1f}s ===")

    print("\n--- Demo ---")
    prompt = ['la', 'inteligencia', 'artificial']
    ids = [vocab.stoi.get(p, vocab.stoi.get('<unk>', 0)) for p in prompt]
    nuevos = m.generar(ids, max_new=30, temperature=0.8)
    print(f"  Prompt: {' '.join(prompt)}")
    print(f"  Byte:   {vocab.a_texto(ids + nuevos)}", flush=True)

if __name__ == '__main__':
    main()
