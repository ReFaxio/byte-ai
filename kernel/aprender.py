#!/usr/bin/env python3
"""Entrenamiento n-grama — extrae secuencias de palabras de textos reales.
Genera datos/asociaciones.db (SQLite, indexado, listo para Pentium).
Sin GPU, sin RAM masiva, sin dependencias."""
import os, json, re, sqlite3, urllib.request, time, sys

RUTA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos')
RUTA_DB = os.path.join(RUTA, 'asociaciones.db')
_ENG = frozenset('the and you that for with this have are was had not but all can has its who his her been will would could should about than then them were what when where which your some any also each very just after such because'.split())
_ESP = frozenset('que con las los por para del como muy bien pero sobre entre desde hasta porque cuando donde quien ella este todo eso esa esos esas estos estas puede tiene hace dice solo casa tiempo parte forma misma nunca siempre tambien entonces despues antes mucho poco algo nada cada aqui alli ahi ser haber estar tener hacer decir ir ver dar saber querer poder poner parecer creer llamar seguir encontrar dejar mirar pensar salir volver tomar pedir hablar pasar llegar llevar sentir tratar tocar cambiar vivir morir nacer crecer buscar esperar perder ganar meter sacar abrir cerrar traer mandar saber deber poder querer necesitar gustar doler importar'.split())
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
        ('https://www.gutenberg.org/cache/epub/320/pg320.txt', 'lazarillo.txt'),
        ('https://www.gutenberg.org/cache/epub/1619/pg1619.txt', 'celestina.txt'),
        ('https://www.gutenberg.org/cache/epub/5201/pg5201.txt', 'don_juan_tenorio.txt'),
        ('https://www.gutenberg.org/cache/epub/50027/pg50027.txt', 'el_si_de_las_ninas.txt'),
        ('https://www.gutenberg.org/cache/epub/17340/pg17340.txt', 'marianela.txt'),
        ('https://www.gutenberg.org/cache/epub/15725/pg15725.txt', 'dona_perfecta.txt'),
        ('https://www.gutenberg.org/cache/epub/29506/pg29506.txt', 'sombrero_tres_picos.txt'),
        ('https://www.gutenberg.org/cache/epub/17073/pg17073.txt', 'la_regenta.txt'),
        ('https://www.gutenberg.org/cache/epub/15781/pg15781.txt', 'estudiante_salamanca.txt'),
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
    textos_def = defs_rae
    print(f"  Textos definición: {len(textos_def)}")

    libros = descargar_gutenberg() if descargar else []

    print("  Subtítulos: eliminados")

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
    t = time.time() - t0
    print(f"\n--- Entrenamiento completado ---")
    print(f"  {total_unique} trigramas")
    print(f"  0 4-gramas")
    print(f"  {unicas} palabras distintas en w1")
    print(f"  Suma frecuencias: {total_freq}")
    print(f"  DB: {RUTA_DB}")
    print(f"  Tiempo: {t:.1f}s")


if __name__ == '__main__':
    descargar = '--no-download' not in sys.argv
    main(descargar=descargar)
