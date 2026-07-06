"""Búsqueda Wikipedia. Sin cache en disco, sin API key, sin if/else."""
import json, urllib.request, urllib.parse

API = "https://es.wikipedia.org/w/api.php"
UA = "Byte/3.0 (pentium; andree@byte.ai)"
_cache = {}


def _api(params):
    params['format'] = 'json'
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode('utf-8'))
    except Exception:
        return None


def buscar(termino):
    if not termino:
        return None
    q = termino.lower().strip()
    if q in _cache:
        return _cache[q]
    data = _api({'action': 'query', 'list': 'search', 'srsearch': q, 'srlimit': 1, 'srprop': ''})
    if not data:
        return None
    pages = data.get('query', {}).get('search', [])
    if not pages:
        _cache[q] = None
        return None
    data = _api({'action': 'query', 'prop': 'extracts', 'exintro': True, 'explaintext': True,
                 'exchars': 400, 'titles': pages[0]['title']})
    if not data:
        return None
    for pid, info in data.get('query', {}).get('pages', {}).items():
        if pid != '-1' and 'extract' in info:
            res = info['extract'].strip()
            if res:
                _cache[q] = res
                return res
    return None
