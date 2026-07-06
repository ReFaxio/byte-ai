"""Herramientas del sistema — cálculos, shell.
Mínimo, sin if/else, solo lo que Byte necesita."""
import subprocess, re


def calc(expresion):
    try:
        exp = expresion.replace('x', '*').replace('÷', '/').replace(',', '.')
        exp = re.sub(r'[^0-9+\-*/().,% ]', '', exp)
        if not exp.strip():
            return None
        r = eval(exp, {'__builtins__': {}}, {})
        return str(r)
    except Exception:
        return None


def shell(comando):
    try:
        r = subprocess.run(comando, shell=True, capture_output=True, text=True, timeout=5)
        return r.stdout.strip()[:500] or r.stderr.strip()[:500] or None
    except Exception:
        return None


def detectar(entrada):
    e = entrada.lower().strip()
    if any(c in entrada for c in '+-*/x÷()') and sum(c.isdigit() for c in entrada) > 1:
        return 'calc'
    if e.startswith('>'):
        return 'shell'
    return None
