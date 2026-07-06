#!/usr/bin/env python3
"""Entrenamiento n-grama — extrae secuencias de palabras de textos reales.
Genera datos/asociaciones.db (SQLite, indexado, listo para Pentium).
Sin GPU, sin RAM masiva, sin dependencias."""
import os, json, re, sqlite3, urllib.request, time, sys

RUTA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos')
RUTA_DB = os.path.join(RUTA, 'asociaciones.db')
STOP = frozenset({
    'que', 'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    'con', 'por', 'para', 'del', 'al', 'en', 'de', 'a', 'e', 'i',
    'o', 'u', 'y', 'no', 'si', 'se', 'le', 'lo', 'su', 'sus',
    'es', 'son', 'era', 'fue', 'ser', 'han', 'has', 'había',
    'este', 'esta', 'esto', 'ese', 'esa', 'eso', 'aquel',
    'muy', 'más', 'pero', 'sin', 'cada', 'como', 'todo',
    'entre', 'durante', 'desde', 'hasta', 'tras', 'ante',
    'sobre', 'bajo', 'contra', 'hacia', 'través',
})


def limpiar(texto):
    texto = texto.lower()
    texto = texto.replace('á', 'a').replace('é', 'e').replace('í', 'i')
    texto = texto.replace('ó', 'o').replace('ú', 'u').replace('ü', 'u')
    texto = re.sub(r'[^a-zñ ]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


def tokenizar(texto):
    return texto.split()


def texto_de_json(archivo, extractor):
    ruta = os.path.join(RUTA, archivo)
    if not os.path.exists(ruta):
        print(f"  {archivo}: no existe")
        return []
    with open(ruta, encoding='utf-8') as f:
        datos = json.load(f)
    return extractor(datos)


def extraer_definiciones(datos):
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


def extraer_conocimiento(datos):
    textos = []
    entidades = datos.get('entidades', {})
    for ent, info in entidades.items():
        if isinstance(info, dict):
            d = info.get('definicion') or info.get('tipo')
            if isinstance(d, str):
                textos.append(f"{ent} {d}")
    relaciones = datos.get('relaciones', [])
    if isinstance(relaciones, list):
        for r in relaciones:
            if isinstance(r, list) and len(r) >= 3:
                textos.append(r[2])
    return textos


def descargar_gutenberg():
    """Descarga algunos libros en español desde Project Gutenberg."""
    libros = [
        ('https://www.gutenberg.org/cache/epub/2000/pg2000.txt', 'quijote.txt'),
        ('https://www.gutenberg.org/cache/epub/16031/pg16031.txt', 'lazarillo.txt'),
        ('https://www.gutenberg.org/cache/epub/17177/pg17177.txt', 'celestina.txt'),
        ('https://www.gutenberg.org/cache/epub/7966/pg7966.txt', 'don_juan_tenorio.txt'),
        ('https://www.gutenberg.org/cache/epub/14668/pg14668.txt', 'el_si_de_las_ninas.txt'),
        ('https://www.gutenberg.org/cache/epub/47993/pg47993.txt', 'marianela.txt'),
        ('https://www.gutenberg.org/cache/epub/47994/pg47994.txt', 'dona_perfecta.txt'),
        ('https://www.gutenberg.org/cache/epub/10723/pg10723.txt', 'sombrero_tres_picos.txt'),
        ('https://www.gutenberg.org/cache/epub/1969/pg1969.txt', 'la_regenta.txt'),
        ('https://www.gutenberg.org/cache/epub/17090/pg17090.txt', 'estudiante_salamanca.txt'),
    ]
    textos = []
    for url, nombre in libros:
        ruta = os.path.join(RUTA, nombre)
        if os.path.exists(ruta):
            with open(ruta, encoding='utf-8') as f:
                textos.append(f.read())
            print(f"  {nombre}: ya descargado")
            continue
        print(f"  Descargando {nombre}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Byte/3.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read().decode('utf-8', errors='ignore')
            # Quitar header/footer de Gutenberg
            inicio = data.find('*** START OF')
            fin = data.find('*** END OF')
            if inicio != -1 and fin != -1:
                data = data[inicio:fin]
            with open(ruta, 'w', encoding='utf-8') as f:
                f.write(data)
            textos.append(data)
            print(f"  {nombre}: {len(data)} bytes")
        except Exception as e:
            print(f"  {nombre}: error {e}")
    return textos


def extraer_ngramas(textos, ventana=3, peso=1):
    total = 0
    for texto in textos:
        if not texto or not isinstance(texto, str):
            continue
        limpio = limpiar(texto)
        tokens = tokenizar(limpio)
        for i in range(len(tokens) - ventana + 1):
            yield tuple(tokens[i:i + ventana])
            total += 1
            if total % 500000 == 0:
                print(f"    {total} n-gramas extraídos...", end='\r')
    print(f"    {total} n-gramas extraídos (peso={peso})")


def construir_db(ngramas, db=None):
    """Construye SQLite con los n-gramas."""
    ruta = db or RUTA_DB
    if os.path.exists(ruta):
        os.remove(ruta)
    conn = sqlite3.connect(ruta)
    conn.execute("""CREATE TABLE ngramas (
        w1 TEXT, w2 TEXT, w3 TEXT, freq INTEGER DEFAULT 1
    )""")
    conn.execute("CREATE INDEX idx_w12 ON ngramas(w1, w2)")
    conn.execute("CREATE INDEX idx_w1 ON ngramas(w1)")

    batch = []
    batch_size = 100000
    total = 0
    conn.execute("BEGIN")
    for ngrama in ngramas:
        if len(ngrama) != 3:
            continue
        w1, w2, w3 = ngrama
        if not all(w.isalpha() or any(c in w for c in 'áéíóúñü') for w in ngrama):
            continue
        batch.append((w1, w2, w3))
        total += 1
        if len(batch) >= batch_size:
            conn.executemany(
                "INSERT INTO ngramas (w1, w2, w3) VALUES (?, ?, ?)", batch)
            batch = []
            print(f"    {total} insertados...", end='\r')
    if batch:
        conn.executemany(
            "INSERT INTO ngramas (w1, w2, w3) VALUES (?, ?, ?)", batch)
    conn.commit()

    # Merge duplicados: suma frecuencias
    print(f"\n  Fusionando duplicados...")
    conn.execute("""
        CREATE TABLE ngramas_final AS
        SELECT w1, w2, w3, COUNT(*) as freq FROM ngramas
        GROUP BY w1, w2, w3
    """)
    conn.execute("DROP TABLE ngramas")
    conn.execute("ALTER TABLE ngramas_final RENAME TO ngramas")
    conn.execute("CREATE INDEX idx_w12 ON ngramas(w1, w2)")
    conn.execute("CREATE INDEX idx_w1 ON ngramas(w1)")
    conn.commit()

    # Estadísticas
    cur = conn.execute("SELECT COUNT(*) FROM ngramas")
    total_unique = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(DISTINCT w1) FROM ngramas")
    unicas = cur.fetchone()[0]
    conn.close()
    return total_unique, unicas


def contar_ngramas(ngramas):
    """Cuenta frecuencias de trigramas en un iterable."""
    counts = {}
    for n in ngramas:
        counts[n] = counts.get(n, 0) + 1
    print(f"    {len(counts)} trigramas distintos")
    return counts


def main(descargar=True):
    print("Byte — aprender.py (entrenamiento n-grama)")
    t0 = time.time()

    print("\n1. Defs & libros por separado...")

    defs_rae = texto_de_json('rae_diccionario.json', extraer_definiciones)
    defs_dicc = texto_de_json('diccionario.json', extraer_definiciones)
    conc = texto_de_json('conocimiento.json', extraer_conocimiento)
    textos_def = defs_rae + defs_dicc + conc
    print(f"  Textos definición: {len(textos_def)}")

    libros = descargar_gutenberg() if descargar else []

    subs_ruta = os.path.join(RUTA, 'subtitulos_es.txt')
    if not os.path.exists(subs_ruta):
        partes = sorted([p for p in os.listdir(RUTA) if p.startswith('subt_')])
        if partes:
            print(f"  Recombinando {len(partes)} partes...")
            with open(subs_ruta, 'wb') as out:
                for p in partes:
                    with open(os.path.join(RUTA, p), 'rb') as f:
                        out.write(f.read())
            print(f"  Subtitulos recombindo: {os.path.getsize(subs_ruta)} bytes")
    if os.path.exists(subs_ruta):
        with open(subs_ruta, encoding='utf-8') as f:
            subtitulos = f.read()
        print(f"  Subtítulos: {len(subtitulos)} bytes añadido")
        libros.append(subtitulos)

    # Identidad de Byte: frases repetidas para que aprenda quién es
    id_ruta = os.path.join(RUTA, 'identidad_byte.txt')
    if os.path.exists(id_ruta):
        with open(id_ruta, encoding='utf-8') as f:
            identidad = f.read().strip().split('\n')
        identidad_repetida = ('\n'.join(identidad) + ' ') * 500
        print(f"  Identidad Byte: {len(identidad)} frases x500")
        libros.append(identidad_repetida)

    print(f"  Textos libros: {len(libros)}")

    print("\n2. Contando n-gramas de definiciones...")
    counts_def = contar_ngramas(extraer_ngramas(textos_def, ventana=3))

    total_def = sum(counts_def.values())
    print(f"\n3. Contando n-gramas de libros+subtitulos...")
    counts_lib_raw = contar_ngramas(extraer_ngramas(libros, ventana=3))
    # Filtrar trigramas con frecuencia ≥2 y aplicar peso 6x
    counts_lib = {n: c * 30 for n, c in counts_lib_raw.items() if c >= 2}
    print(f"  Descartados {len(counts_lib_raw) - len(counts_lib)} trigramas con freq=1")
    final = dict(counts_def)
    for n, c in counts_lib.items():
        final[n] = final.get(n, 0) + c
    print(f"  Peso libros+subtitulos: 30x (con filtro freq≥2)")
    print(f"  {len(final)} trigramas únicos totales")

    ngramas_pesados = []
    for (w1, w2, w3), freq in final.items():
        ngramas_pesados.append((w1, w2, w3, freq))
    print(f"  {len(ngramas_pesados)} filas para insertar")

    print("\n5. Construyendo base de datos SQLite...")
    ruta = RUTA_DB
    if os.path.exists(ruta):
        os.remove(ruta)
    conn = sqlite3.connect(ruta)
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA cache_size=100000")
    conn.execute("""CREATE TABLE ngramas (
        w1 TEXT, w2 TEXT, w3 TEXT, freq INTEGER DEFAULT 1
    )""")

    batch = []
    conn.execute("BEGIN")
    for i, (w1, w2, w3, freq) in enumerate(ngramas_pesados):
        batch.append((w1, w2, w3, freq))
        if len(batch) >= 200000:
            conn.executemany(
                "INSERT INTO ngramas (w1, w2, w3, freq) VALUES (?, ?, ?, ?)", batch)
            batch = []
            if i % 1000000 == 0:
                print(f"    {i//1000000}M/{len(ngramas_pesados)//1000000}M...", end='\r')
    if batch:
        conn.executemany(
            "INSERT INTO ngramas (w1, w2, w3, freq) VALUES (?, ?, ?, ?)", batch)
    conn.commit()

    print("\n  Creando índices...", end='')
    conn.execute("CREATE INDEX idx_w12 ON ngramas(w1, w2)")
    conn.execute("CREATE INDEX idx_w1 ON ngramas(w1)")
    print(" OK")

    cur = conn.execute("SELECT COUNT(*) FROM ngramas")
    total_unique = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(DISTINCT w1) FROM ngramas")
    unicas = cur.fetchone()[0]
    cur = conn.execute("SELECT SUM(freq) FROM ngramas")
    total_freq = cur.fetchone()[0]
    conn.close()

    #
    # 6. 4-gramas (streaming directo a SQLite, sin RAM)
    #
    print("\n6. Extrayendo 4-gramas (streaming)...")
    conn = sqlite3.connect(RUTA_DB)
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA cache_size=100000")

    # Tablas temporales para definiciones y libros por separado
    conn.execute("CREATE TABLE _4def (w1 TEXT, w2 TEXT, w3 TEXT, w4 TEXT)")
    conn.execute("CREATE TABLE _4lib (w1 TEXT, w2 TEXT, w3 TEXT, w4 TEXT)")
    batch_def, batch_lib = [], []
    BDEF, BLIB = 200000, 200000
    total4_def, total4_lib = 0, 0

    conn.execute("BEGIN")
    for n4 in extraer_ngramas(textos_def, ventana=4):
        if len(n4) != 4: continue
        batch_def.append(n4)
        total4_def += 1
        if len(batch_def) >= BDEF:
            conn.executemany("INSERT INTO _4def VALUES (?,?,?,?)", batch_def)
            batch_def = []
            if total4_def % 1000000 == 0:
                print(f"  defs: {total4_def//1000000}M...", end='\r')
    if batch_def:
        conn.executemany("INSERT INTO _4def VALUES (?,?,?,?)", batch_def)
    conn.commit()
    print(f"  defs: {total4_def} 4-gramas")

    conn.execute("BEGIN")
    for n4 in extraer_ngramas(libros, ventana=4):
        if len(n4) != 4: continue
        batch_lib.append(n4)
        total4_lib += 1
        if len(batch_lib) >= BLIB:
            conn.executemany("INSERT INTO _4lib VALUES (?,?,?,?)", batch_lib)
            batch_lib = []
            if total4_lib % 5000000 == 0:
                print(f"  libros: {total4_lib//1000000}M...", end='\r')
    if batch_lib:
        conn.executemany("INSERT INTO _4lib VALUES (?,?,?,?)", batch_lib)
    conn.commit()
    print(f"  libros: {total4_lib} 4-gramas")

    # Agregar definiciones (sin filtro, peso 1)
    print("  Agregando definiciones...")
    conn.execute("""CREATE TABLE ngramas4 AS
        SELECT w1, w2, w3, w4, COUNT(*) as freq FROM _4def
        GROUP BY w1, w2, w3, w4""")
    conn.execute("DROP TABLE _4def")

    # Agregar libros (filtro freq>=2, peso 30x)
    print("  Agregando libros (freq>=2, x30)...")
    conn.execute("""CREATE TABLE _lib_agg AS
        SELECT w1, w2, w3, w4, COUNT(*) as raw_freq FROM _4lib
        GROUP BY w1, w2, w3, w4 HAVING raw_freq >= 2""")
    conn.execute("DROP TABLE _4lib")

    # Merge: insertar libros con peso 30 en ngramas4
    cur = conn.execute("SELECT w1, w2, w3, w4, raw_freq FROM _lib_agg")
    batch = []
    conn.execute("BEGIN")
    idx = 0
    for row in cur:
        w1, w2, w3, w4, raw = row
        batch.append((w1, w2, w3, w4, raw * 30))
        idx += 1
        if len(batch) >= 200000:
            conn.executemany(
                "INSERT INTO ngramas4 (w1, w2, w3, w4, freq) VALUES (?,?,?,?,?)", batch)
            batch = []
            if idx % 1000000 == 0:
                print(f"  merge: {idx//1000000}M...", end='\r')
    if batch:
        conn.executemany(
            "INSERT INTO ngramas4 (w1, w2, w3, w4, freq) VALUES (?,?,?,?,?)", batch)
    conn.commit()
    conn.execute("DROP TABLE _lib_agg")

    print("\n  Creando índices ngramas4...")
    conn.execute("CREATE INDEX idx_w123 ON ngramas4(w1, w2, w3)")
    conn.execute("CREATE INDEX idx_w1_4 ON ngramas4(w1)")
    cur = conn.execute("SELECT COUNT(*) FROM ngramas4")
    total4 = cur.fetchone()[0]
    conn.close()

    t = time.time() - t0
    print(f"\n--- Entrenamiento completado ---")
    print(f"  {total_unique} trigramas")
    print(f"  {total4} 4-gramas")
    print(f"  {unicas} palabras distintas en w1")
    print(f"  Suma frecuencias: {total_freq}")
    print(f"  DB: {RUTA_DB}")
    print(f"  Tiempo: {t:.1f}s")


if __name__ == '__main__':
    descargar = '--no-download' not in sys.argv
    main(descargar=descargar)
