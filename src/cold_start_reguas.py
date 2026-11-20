"""Cold start, parte sem LLM: as referencias e as reguas do dia zero.

Antes de gastar uma chamada de API, fixa-se contra o que o LLM sera comparado.
Cada regua responde a um degrau da escada do PLANO.md:

  degrau 0  mediana dos outros projetos   o que se faria sem nada da equipe: o
                                          valor do meio de todas as historias de
                                          treino dos OUTROS 22 projetos
  degrau 2  referencia mais parecida      copiar o valor da referencia com o texto
                                          mais parecido (TF-IDF). E a mesma
                                          entrada que o LLM recebe, sem o LLM: se
                                          o LLM nao a vencer, ele nao acrescentou
                                          nada alem de semelhanca de palavras
  degrau 3  mediana da equipe             so existe depois de haver historico;
                                          e o teto de referencia, nao exigencia

E, para a curva de cruzamento, a mediana das n historias mais recentes antes do
teste, com n de 1 ate o historico inteiro: "com quantas historias estimadas a
mediana da equipe alcanca o metodo do dia zero".

Saidas: resultados/<conjunto>/cold_start_referencias.csv (quais historias
viraram referencia) e cold_start_previsoes_reguas.csv (uma linha por historia
de teste e regua).
"""
import numpy as np
import pandas as pd

from cold_start_comum import (conjuntos_de_referencia, projetos_cold_start,
                              vetorizador)
from sonda_comum import modelo_tfidf, limitador, salvar

TAMANHOS_CURVA = [1, 2, 3, 5, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200, 300,
                  400, 600, 800, None]  # None = todo o historico


def mais_parecida(vet, refs, treino, teste):
    """Valor da referencia de texto mais parecido com cada historia de teste."""
    valores = np.array(list(refs))
    R = vet.transform(treino.loc[list(refs.values())].texto_limpo)
    T = vet.transform(teste.texto_limpo)
    return valores[np.asarray((T @ R.T).todense()).argmax(axis=1)]


def main():
    dados = list(projetos_cold_start())
    treinos = {s: tr.storypoint for s, tr, _, _ in dados}

    linhas, refs_linhas = [], []
    registrar = lambda sigla, teste, cond, refs, p, n=np.nan: linhas.extend(
        {"projeto": sigla, "issuekey": k, "storypoint": y, "condicao": cond,
         "referencias": refs, "n_historico": n, "previsao": float(v)}
        for k, y, v in zip(teste.issuekey, teste.storypoint, p))

    for sigla, treino, hist, teste in dados:
        n = len(teste)
        outros = pd.concat([y for s, y in treinos.items() if s != sigla])
        registrar(sigla, teste, "mediana dos outros projetos", "-",
                  np.full(n, outros.median()))
        registrar(sigla, teste, "mediana da equipe", "-",
                  np.full(n, treino.storypoint.median()))

        vet = vetorizador().fit(treino.texto_limpo)  # so texto, sem rotulo
        for forma, refs in conjuntos_de_referencia(treino).items():
            registrar(sigla, teste, "referencia mais parecida (TF-IDF)", forma,
                      mais_parecida(vet, refs, treino, teste))
            for v, i in refs.items():
                refs_linhas.append({"projeto": sigla, "forma": forma, "valor": v,
                                    "issuekey": treino.loc[i, "issuekey"],
                                    "candidatas": int((treino.storypoint == v).sum())})

        # o modelo de texto treinado com todo o historico, para contexto
        m = modelo_tfidf().fit(hist.texto, hist.storypoint)
        registrar(sigla, teste, "TF-IDF + SVR (historico inteiro)", "-",
                  limitador(hist.storypoint)(m.predict(teste.texto)))

        for t in TAMANHOS_CURVA:
            if t is not None and t > len(hist):
                continue
            ultimas = hist.storypoint if t is None else hist.storypoint.iloc[-t:]
            registrar(sigla, teste, "mediana das ultimas n", "-",
                      np.full(n, ultimas.median()), n=len(ultimas))

        print(f"  {sigla:<12} treino={len(treino):>5}  teste amostrado={n:>3}  "
              f"referencias={len(refs)}", flush=True)

    r = pd.DataFrame(linhas)
    r["erro"] = (r.storypoint - r.previsao).abs()
    resumo = (r[r.condicao != "mediana das ultimas n"]
              .groupby(["condicao", "referencias", "projeto"]).erro.mean()
              .groupby(["condicao", "referencias"]).mean())
    print("\n=== MAE medio por projeto (reguas do dia zero) ===")
    print(resumo.to_string(float_format=lambda v: f"{v:.3f}"))

    salvar(pd.DataFrame(refs_linhas), "cold_start_referencias.csv")
    salvar(r.drop(columns="erro"), "cold_start_previsoes_reguas.csv")


if __name__ == "__main__":
    main()
