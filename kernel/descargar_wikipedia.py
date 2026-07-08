#!/usr/bin/env python3
"""Descarga y extrae texto de Wikipedia en español.
Dump oficial: ~600MB bz2, ~3GB texto.
Sin dependencias — solo stdlib + bz2 + xml.parsers.expat."""

import bz2, os, re, sys, time, urllib.request
import xml.parsers.expat

RUTA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos')
URL = 'https://dumps.wikimedia.org/eswiki/latest/eswiki-latest-pages-articles-multistream.xml.bz2'
PARTES = 16

# ---------------------------------------------------------------------------
# Limpiar marcado Wikipedia
# ---------------------------------------------------------------------------

def limpiar_wiki(texto):
    """Elimina marcado MediaWiki, solo deja texto plano."""
    if not texto:
        return ''
    # Quitar referencias
    texto = re.sub(r'<ref[^>]*>.*?</ref>', '', texto, flags=re.DOTALL)
    texto = re.sub(r'<ref[^>]*/>', '', texto)
    # Quitar comments
    texto = re.sub(r'<!--.*?-->', '', texto, flags=re.DOTALL)
    # Quitar {{...}} (plantillas, infoboxes, etc.)
    texto = re.sub(r'\{\{[^}]*?\}\}', '', texto)
    # Quitar [[]] dobles: [[Target|Text]] -> Text, [[Target]] -> Target
    texto = re.sub(r'\[\[([^|\]]*?)\]\]', r'\1', texto)
    texto = re.sub(r'\[\[[^|\]]*?\|([^\]]*?)\]\]', r'\1', texto)
    # Quitar '' y ''' (itálica y negrita)
    texto = texto.replace("'''", '').replace("''", '')
    # Quitar ===== headings =====
    texto = re.sub(r'^=+[^=]+=+\s*$', '', texto, flags=re.MULTILINE)
    # Quitar etiquetas HTML sobrantes
    texto = re.sub(r'<[^>]+>', '', texto)
    # Quitar enlaces externos [url text] -> text
    texto = re.sub(r'\[https?://[^\]]*? ([^\]]*?)\]', r'\1', texto)
    texto = re.sub(r'\[https?://[^\]]*?\]', '', texto)
    # Quitar categorías
    texto = re.sub(r'\[\[Categoría:[^\]]*?\]\]', '', texto)
    texto = re.sub(r'\[\[Category:[^\]]*?\]\]', '', texto)
    # Quitar archivos e imágenes
    texto = re.sub(r'\[\[(Archivo|File|Imagen|Image):[^\]]*?\]\]', '', texto)
    # Quitar bloques de código
    texto = re.sub(r'<code>.*?</code>', '', texto, flags=re.DOTALL)
    texto = re.sub(r'<pre>.*?</pre>', '', texto, flags=re.DOTALL)
    # Líneas que empiezan con {| |} (tablas wiki)
    texto = re.sub(r'^[\|\{\}!].*$', '', texto, flags=re.MULTILINE)
    # Múltiples espacios y líneas vacías
    texto = re.sub(r'\n\s*\n', '\n', texto)
    texto = re.sub(r'  +', ' ', texto)
    return texto.strip()


def es_valido(texto):
    """Filtra páginas vacías, cortas, o de contenido no útil."""
    if len(texto) < 100:
        return False
    # Saltar páginas de desambiguación
    if texto.startswith('desambiguación') or 'puede referirse a' in texto[:200]:
        return False
    # Saltar listas de episodios, personajes ficticios, etc.
    if re.match(r'^(Anexo|Lista|Episodio)s?(:| )', texto):
        return False
    return True


# ---------------------------------------------------------------------------
# Compartir archivos en partes
# ---------------------------------------------------------------------------

class EscritorPartes:
    """Divide el texto extraído en PARTES archivos."""

    def __init__(self):
        self.parte_actual = 0
        self.tokens_esta = 0
        self.max_tokens_por_parte = 2000000
        self.archivo = None
        self._abrir()

    def _abrir(self):
        ruta = os.path.join(RUTA, f'wiki_parte_{self.parte_actual}.txt')
        self.archivo = open(ruta, 'w', encoding='utf-8')
        self.tokens_esta = 0

    def escribir(self, texto):
        if not texto:
            return
        palabras = texto.split()
        self.tokens_esta += len(palabras)
        self.archivo.write(texto + '\n\n')
        if self.tokens_esta >= self.max_tokens_por_parte and self.parte_actual < PARTES - 1:
            self.archivo.close()
            print(f'  Parte {self.parte_actual} guardada ({self.tokens_esta} tokens)', flush=True)
            self.parte_actual += 1
            self._abrir()

    def cerrar(self):
        if self.archivo:
            self.archivo.close()
            print(f'  Parte {self.parte_actual} guardada ({self.tokens_esta} tokens)', flush=True)


# ---------------------------------------------------------------------------
# Parser XML streaming
# ---------------------------------------------------------------------------

class WikiParser:
    """Parser SAX del dump de Wikipedia."""

    def __init__(self, escritor):
        self.escritor = escritor
        self.en_text = False
        self.en_title = False
        self.en_ns = False
        self.texto_actual = []
        self.title_actual = []
        self.ns_actual = []
        self.profundidad_text = 0
        self.total_paginas = 0
        self.total_extraidas = 0

    def start_tag(self, name, attrs):
        name = name.split('}')[-1]  # quitar namespace xmlns
        if name == 'text':
            self.en_text = True
            self.texto_actual = []
            self.profundidad_text = 1
        elif name == 'title':
            self.en_title = True
            self.title_actual = []
        elif name == 'ns':
            self.en_ns = True
            self.ns_actual = []
        if self.en_text and name != 'text':
            self.profundidad_text += 1

    def end_tag(self, name):
        name = name.split('}')[-1]
        if name == 'text':
            self.en_text = False
            self._procesar_texto()
        elif name == 'title':
            self.en_title = False
        elif name == 'ns':
            self.en_ns = False
        elif name == 'page':
            self.total_paginas += 1
            if self.total_paginas % 10000 == 0:
                print(f'  {self.total_paginas} páginas, {self.total_extraidas} extraídas...', end='\r', flush=True)
        if self.en_text:
            self.profundidad_text -= 1

    def char_data(self, data):
        if self.en_text and self.profundidad_text == 0:
            # Solo texto directo dentro de <text>, no en sub-elementos
            # En realidad todo el contenido de <text> es texto plano en Wikipedia dumps
            pass
        if self.en_text:
            self.texto_actual.append(data)
        if self.en_title:
            self.title_actual.append(data)
        if self.en_ns:
            self.ns_actual.append(data)

    def _procesar_texto(self):
        texto_raw = ''.join(self.texto_actual)
        ns = ''.join(self.ns_actual).strip()
        # Solo namespace 0 (artículos)
        if ns != '0':
            return
        texto_limpio = limpiar_wiki(texto_raw)
        if es_valido(texto_limpio):
            self.escritor.escribir(texto_limpio)
            self.total_extraidas += 1


def descargar_y_extraer():
    """Descarga el dump y extrae texto plano en partes."""
    t0 = time.time()

    # 1. Descargar
    ruta_dump = os.path.join(RUTA, 'eswiki.xml.bz2')
    if not os.path.exists(ruta_dump):
        print(f'Descargando {URL}...')
        print('  (~600MB, puede tomar varios minutos)')
        req = urllib.request.Request(URL, headers={'User-Agent': 'Byte/4.0'})
        with urllib.request.urlopen(req, timeout=300) as r:
            total = int(r.headers.get('Content-Length', 0))
            descargado = 0
            with open(ruta_dump, 'wb') as f:
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    descargado += len(chunk)
                    if total:
                        pct = descargado * 100 // total
                        print(f'  {descargado//1048576}MB/{total//1048576}MB ({pct}%)', end='\r', flush=True)
        print(f'\n  Descarga completada: {descargado//1048576}MB', flush=True)
    else:
        tam = os.path.getsize(ruta_dump)
        print(f'  Usando dump existente: {tam//1048576}MB', flush=True)

    # 2. Extraer
    print(f'\nExtrayendo texto de Wikipedia (partes={PARTES})...')
    escritor = EscritorPartes()
    parser = WikiParser(escritor)

    p = xml.parsers.expat.ParserCreate()
    p.StartElementHandler = parser.start_tag
    p.EndElementHandler = parser.end_tag
    p.CharacterDataHandler = parser.char_data

    with bz2.open(ruta_dump, 'rb') as f:
        # Leer en chunks para no saturar RAM
        chunk_size = 1048576 * 4  # 4MB
        leido = 0
        tam = os.path.getsize(ruta_dump)
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            p.Parse(chunk, False)
            leido += len(chunk)
            pct = leido * 100 // tam
            print(f'\r  Procesando: {leido//1048576}MB/{tam//1048576}MB ({pct}%) - '
                  f'{parser.total_paginas} páginas, {parser.total_extraidas} extraídas',
                  end='', flush=True)
        p.Parse(b'', True)

    escritor.cerrar()
    t = time.time() - t0
    print(f'\n\nCompletado en {t:.0f}s')
    print(f'  {parser.total_paginas} páginas procesadas')
    print(f'  {parser.total_extraidas} artículos extraídos')
    print(f'  Archivos: {RUTA}/wiki_parte_0..{PARTES-1}.txt')

    # Limpiar dump
    os.remove(ruta_dump)
    print('  Dump eliminado')


if __name__ == '__main__':
    print('=== Descargar Wikipedia en español ===')
    descargar_y_extraer()
