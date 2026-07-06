#!/usr/bin/env python3
"""Construye índice de conversación: mapea palabras a posiciones en el texto combinado.
Correr después de aprender.py, antes de usar Byte."""
import os, re, json, sys

RUTA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos')

ARCHIVOS = ['subtitulos_es.txt', 'quijote.txt', 'la_regenta.txt',
            'celestina.txt', 'don_juan_tenorio.txt',
            'dona_perfecta.txt', 'marianela.txt',
            'sombrero_tres_picos.txt', 'el_si_de_las_ninas.txt',
            'estudiante_salamanca.txt', 'lazarillo.txt']

def normalizar(t):
    t = t.lower()
    for a, b in [('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u'),('ü','u')]:
        t = t.replace(a, b)
    return re.sub(r'[^a-zñ ]', ' ', t).strip()

def main():
    print("Indizando textos de conversación...")
    ruta_comb = os.path.join(RUTA, 'memoria_texto.txt')
    ruta_idx = os.path.join(RUTA, 'memoria_idx.json')

    # Combinar todos los textos en uno solo
    idx = {}  # word -> [byte_positions]
    with open(ruta_comb, 'w', encoding='utf-8', errors='replace') as out:
        for archivo in ARCHIVOS:
            ruta = os.path.join(RUTA, archivo)
            if not os.path.exists(ruta):
                print(f"  {archivo}: no existe, saltando")
                continue
            with open(ruta, encoding='utf-8', errors='replace') as f:
                contenido = f.read()
            out.write(contenido)
            out.write('\n')
            print(f"  {archivo}: {len(contenido)} bytes")

    # Construir índice: cada palabra -> posiciones en el texto combinado
    print(f"\nTexto combinado: {os.path.getsize(ruta_comb)} bytes")
    print("Construyendo índice (esto puede tomar un minuto)...")

    with open(ruta_comb, 'r', encoding='utf-8', errors='replace') as f:
        contenido = f.read()

    normal = normalizar(contenido)
    palabras = normal.split()
    print(f"  {len(palabras)} palabras totales")

    # Índice: para cada palabra, guardar hasta 200 posiciones
    seen = set()
    for i, p in enumerate(palabras):
        if len(p) < 2:
            continue
        if p in seen:
            continue
        # Encontrar posiciones de esta palabra
        posiciones = []
        idx_p = 0
        pattern = ' ' + p + ' '
        count = 0
        while count < 200:
            idx_p = normal.find(pattern, idx_p)
            if idx_p == -1:
                break
            posiciones.append(idx_p)
            idx_p += 1
            count += 1
        if posiciones:
            idx[p] = posiciones
            if len(idx) % 10000 == 0:
                print(f"  {len(idx)} palabras indexadas...", end='\r')

    # También indexar bigramas comunes (w1 + ' ' + w2)
    print(f"\n  Indexando bigramas...")
    bigram_count = 0
    for i in range(len(palabras) - 1):
        if bigram_count >= 500000:
            break
        w1, w2 = palabras[i], palabras[i+1]
        if len(w1) < 2 or len(w2) < 2:
            continue
        bigrama = w1 + ' ' + w2
        if bigrama in idx:
            continue
        # Buscar primeras 50 ocurrencias
        posiciones = []
        idx_b = 0
        pattern = ' ' + bigrama + ' '
        count = 0
        while count < 50:
            idx_b = normal.find(pattern, idx_b)
            if idx_b == -1:
                break
            posiciones.append(idx_b)
            idx_b += 1
            count += 1
        if posiciones:
            idx[bigrama] = posiciones
            bigram_count += len(posiciones)
            if len(idx) % 5000 == 0:
                print(f"  {len(idx)} entradas en índice...", end='\r')

    with open(ruta_idx, 'w', encoding='utf-8') as f:
        json.dump(idx, f, ensure_ascii=False)

    print(f"\n\nÍndice guardado: {ruta_idx}")
    print(f"  {len(idx)} entradas (palabras + bigramas)")
    print(f"  Memoria: {os.path.getsize(ruta_idx)} bytes")

if __name__ == '__main__':
    main()
