"""Qual conjunto de dados o pipeline le, escolhido pela variavel CONJUNTO.

    (vazio) / deepse   benchmark de Choetkiertikul et al.   dados/            -> resultados/
    tawos              TAWOS, conjunto de replicacao         tawos/dados/      -> resultados/tawos/
    sintetica          backlog escrito e estimado por IAs    base_sintetica/dados/ -> resultados/sintetica/

Sem a variavel, o comportamento e exatamente o de antes: mesma pasta de dados,
mesma pasta de saida, mesma ordem das issues.
"""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
NOME = os.environ.get("CONJUNTO", "deepse")

_DADOS = {"deepse": RAIZ / "dados",
          "tawos": RAIZ / "tawos" / "dados",
          "sintetica": RAIZ / "base_sintetica" / "dados"}
if NOME not in _DADOS:
    raise SystemExit(f"CONJUNTO desconhecido: {NOME!r} (use {', '.join(_DADOS)})")

DADOS = _DADOS[NOME]
SAIDA = RAIZ / "resultados" if NOME == "deepse" else RAIZ / "resultados" / NOME
SAIDA.mkdir(parents=True, exist_ok=True)
