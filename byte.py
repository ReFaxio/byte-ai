#!/usr/bin/env python3
"""Byte — Entry point mínimo. 0 strings fijos."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from kernel.core import Byte


def cli():
    b = Byte()
    print("Byte — Ctrl+C o EOF para terminar\n", flush=True)
    while True:
        try:
            e = input("Tu: ").strip()
            if e:
                print("Byte:", b.procesar(e), flush=True)
        except (KeyboardInterrupt, EOFError):
            print("\n", flush=True)
            break

if __name__ == '__main__':
    cli()