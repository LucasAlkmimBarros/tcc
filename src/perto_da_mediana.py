"""O modelo de texto fica perto da mediana? E de onde vem o ganho dele?

Analise descritiva, feita depois de ver os resultados (nao ha criterio fixado
antes: nada aqui escolhe modelo nem muda o software). Nasceu de usar o software:
no RoomBook, o modelo so sugeria 3 ou 5, ate para uma historia de 20 pontos.

No mesmo corte do experimento (80% historico, 20% teste), com o mesmo modelo
(TF-IDF + SVR, limitado ao intervalo do treino), mede-se por projeto:

- quantas sugestoes caem no mesmo cartao da mediana da equipe;
- das historias de teste com 13 pontos ou mais, quantas o modelo sugeriu com 13
  ou mais (o software sugere quebrar historias a partir de 13);
- o erro do modelo e o da mediana, separando as historias de teste pelo valor
  real: abaixo da mediana, igual a ela e acima dela.

Uso:  python src/perto_da_mediana.py ; CONJUNTO=tawos ... ; CONJUNTO=sintetica ...
"""
import numpy as np
import pandas as pd

from sonda_comum import (cartao, dividir_historico_teste, imprimir, limitador,
                         modelo_tfidf, projetos, salvar)
from conjunto import NOME

GRANDE = 13


def main():
    por_projeto, por_historia = [], []
    for nome, d in projetos():
        hist, teste = dividir_historico_teste(d)
        y = teste.storypoint.values.astype(float)
        mediana = float(hist.storypoint.median())
        m = modelo_tfidf().fit(hist.texto, hist.storypoint)
        p = limitador(hist.storypoint)(m.predict(teste.texto))
        grandes = y >= GRANDE
        por_projeto.append({
            "projeto": nome, "n_teste": len(y), "mediana": mediana,
            "pct_cartao_da_mediana": 100 * np.mean(cartao(p) == cartao([mediana])[0]),
            "maior_sugerido": cartao(p).max(), "maior_real": y.max(),
            "n_grandes": int(grandes.sum()),
            "grandes_sugeridas_grandes": int((cartao(p)[grandes] >= GRANDE).sum()),
        })
        posicao = np.where(y < mediana, "abaixo", np.where(y == mediana, "igual", "acima"))
        por_historia.append(pd.DataFrame({"projeto": nome, "posicao": posicao, "real": y,
                                          "erro_modelo": np.abs(p - y),
                                          "erro_mediana": np.abs(mediana - y)}))

    r = pd.DataFrame(por_projeto)
    h = pd.concat(por_historia, ignore_index=True)

    print(f"=== {NOME}: {len(r)} projetos, {len(h)} historias de teste ===")
    print(f"sugestoes no cartao da mediana: {r.pct_cartao_da_mediana.median():.0f}% "
          f"(mediana entre projetos; de {r.pct_cartao_da_mediana.min():.0f}% "
          f"a {r.pct_cartao_da_mediana.max():.0f}%)")
    print(f"historias de {GRANDE} ou mais no teste: {r.n_grandes.sum()}; "
          f"o modelo sugeriu {GRANDE} ou mais em {r.grandes_sugeridas_grandes.sum()}")

    faixas = (h.groupby("posicao").agg(historias=("real", "size"),
                                        erro_mediana=("erro_mediana", "mean"),
                                        erro_modelo=("erro_modelo", "mean"))
              .reindex(["abaixo", "igual", "acima"]))
    imprimir(faixas, "erro por historia, pelo valor real em relacao a mediana "
                     "(todas as historias de teste juntas)")

    salvar(r, "perto_da_mediana.csv")
    salvar(faixas.reset_index(), "perto_da_mediana_faixas.csv")


if __name__ == "__main__":
    main()
