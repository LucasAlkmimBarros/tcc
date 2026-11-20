"""Confere que o software usa o mesmo modelo que foi testado.

Roda o software no modo demonstracao em todos os projetos do Deep-SE e do TAWOS e
compara o erro medio da mediana e do modelo com o que a sonda de texto publicou
(`resultados/sonda_texto.csv`, condicoes "mediana (nenhum texto)" e "texto
completo"). Os numeros precisam bater projeto a projeto.

Uso (de dentro de experimento/):  .venv/bin/python software/conferir.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import previsao as pv

RAIZ = Path(__file__).resolve().parent.parent
CONJUNTOS = [("Deep-SE", RAIZ / "dados", RAIZ / "resultados"),
             ("TAWOS", RAIZ / "tawos" / "dados", RAIZ / "resultados" / "tawos")]


def main():
    falhas = 0
    for nome, dados, res in CONJUNTOS:
        ref = pd.read_csv(res / "sonda_texto.csv").pivot(index="projeto", columns="condicao",
                                                          values="MAE")
        for arq in sorted(dados.glob("*.csv")):
            hist, novas = pv.demonstracao(pv.ler_csv(arq))
            s = pv.sugerir(hist, novas)
            y = novas.storypoint.values.astype(float)
            mediana = np.abs(y - s.mediana).mean()
            modelo = np.abs(y - s.modelo_valor).mean()
            esperado = ref.loc[arq.stem]
            ok = (np.isclose(mediana, esperado["mediana (nenhum texto)"])
                  and np.isclose(modelo, esperado["texto completo"]))
            falhas += not ok
            print(f"  {nome:<8} {arq.stem:<20} mediana {mediana:.3f}  modelo {modelo:.3f}"
                  f"  {'ok' if ok else 'DIFERENTE'}")
    print(f"\n{'tudo bate' if not falhas else f'{falhas} projetos diferentes'}")
    raise SystemExit(1 if falhas else 0)


if __name__ == "__main__":
    main()
