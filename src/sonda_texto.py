"""Sonda 2: o modelo esta lendo o texto, ou so devolvendo a tendencia central?

Por que esta sonda existe. A curva de aprendizado (sonda 1) e plana: 25 historias
produzem quase o mesmo erro que milhares. Ha duas explicacoes possiveis e elas
levam a decisoes de produto opostas. Se o modelo e eficiente em dados, a empresa
pode adotar na primeira semana. Se o modelo nao esta aprendendo do texto, o
software nao tem valor nenhum sobre uma planilha que calcula a mediana. Esta
sonda distingue os dois casos.

Tres testes, todos no split cronologico do experimento principal:

A) Dispersao e correlacao. Um preditor que le o texto varia junto com a verdade.
   Mede-se a razao entre o desvio-padrao das previsoes e o da verdade (1,0 =
   varia tanto quanto) e a correlacao previsao-verdade.

B) Texto disponivel na previsao. No treino o modelo viu titulo+descricao, mas no
   refinamento a historia costuma estar magra. Reavalia-se o mesmo modelo dando
   so o titulo, para medir quanto do desempenho depende de a descricao existir no
   momento da previsao -- um risco de train/serve skew.

C) Embaralhamento do texto de teste. Destroi-se o pareamento texto<->rotulo,
   mantendo a distribuicao das previsoes. Se o erro quase nao piora, a
   informacao usada nao vinha do texto. E o teste decisivo: ele separa o sinal
   do texto da contribuicao do intercepto.

A leitura de (C) e uma decomposicao: prever o centro exato (mediana) e o piso;
texto embaralhado adiciona variacao puramente ruidosa; texto correto adiciona a
mesma variacao, mas atribuida corretamente. A diferenca entre as duas ultimas e
o sinal real do texto, e a diferenca entre a ultima e a mediana e o ganho
liquido que o software entrega.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sonda_comum import (dividir_historico_teste, imprimir, limitador,
                         metricas, modelo_tfidf, projetos, salvar)
from experimento import SEMENTE


def main():
    linhas, dispersao = [], []
    print("Sondando a contribuicao do texto por projeto:\n")
    for nome, d in projetos():
        hist, teste = dividir_historico_teste(d)
        limitar = limitador(hist.storypoint)
        y = teste.storypoint.values.astype(float)

        m = modelo_tfidf()
        m.fit(hist.texto, hist.storypoint)

        # C) mesmo modelo, texto de teste embaralhado entre as issues
        texto_embaralhado = teste.texto.sample(frac=1, random_state=SEMENTE).values
        # B) modelo treinado so com titulo, para o caso da historia magra
        m_titulo = modelo_tfidf()
        m_titulo.fit(hist.title.fillna(""), hist.storypoint)

        previsoes = {
            "mediana (nenhum texto)": np.full(len(y), hist.storypoint.median()),
            "texto embaralhado": limitar(m.predict(texto_embaralhado)),
            "texto completo": limitar(m.predict(teste.texto)),
            "so titulo na previsao": limitar(m.predict(teste.title.fillna(""))),
            "so titulo em treino e teste": limitar(m_titulo.predict(teste.title.fillna(""))),
        }
        for rotulo, p in previsoes.items():
            linhas.append({"projeto": nome, "condicao": rotulo, **metricas(y, p)})

        p_completo = previsoes["texto completo"]
        dispersao.append({
            "projeto": nome,
            "sd_verdade": y.std(),
            "sd_previsao": p_completo.std(),
            "razao_sd": p_completo.std() / y.std() if y.std() else np.nan,
            "r_previsao_verdade": (pearsonr(p_completo, y)[0]
                                   if p_completo.std() > 1e-9 else np.nan),
        })
        print(f"  {nome:<20} n_teste={len(teste):>4}", flush=True)

    r, disp = pd.DataFrame(linhas), pd.DataFrame(dispersao)

    imprimir(disp.set_index("projeto"), "A) o modelo varia suas previsoes?", 3)
    print(f"\n  razao de desvio media : {disp.razao_sd.mean():.3f}"
          f"   (1,0 = varia tanto quanto a verdade)")
    print(f"  correlacao media      : {disp.r_previsao_verdade.mean():.3f}")

    ordem = ["mediana (nenhum texto)", "texto embaralhado", "texto completo",
             "so titulo na previsao", "so titulo em treino e teste"]
    resumo = (r.groupby("condicao")[["MAE", "dentro_1", "cartao_exato"]]
              .mean().reindex(ordem))
    imprimir(resumo, f"B/C) decomposicao: quanto o texto contribui (media dos {r.projeto.nunique()})", 3)

    mediana = resumo.loc["mediana (nenhum texto)", "MAE"]
    embaralhado = resumo.loc["texto embaralhado", "MAE"]
    completo = resumo.loc["texto completo", "MAE"]
    print(f"\n  ruido introduzido pela variacao : {embaralhado - mediana:+.3f} MAE")
    print(f"  sinal recuperado pelo texto     : {embaralhado - completo:+.3f} MAE")
    print(f"  GANHO LIQUIDO sobre a mediana   : {mediana - completo:+.3f} MAE")
    print(f"  custo de perder a descricao     : "
          f"{resumo.loc['so titulo na previsao', 'MAE'] - completo:+.3f} MAE")

    salvar(r, "sonda_texto.csv")
    salvar(disp, "sonda_texto_dispersao.csv")


if __name__ == "__main__":
    main()
