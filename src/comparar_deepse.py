"""Compara o modelo deste trabalho com os resultados publicados do Deep-SE.

Por que isto existe. O benchmark usado aqui vem de Choetkiertikul et al., e o
artigo deles reporta, na Tabela II, o desempenho do proprio modelo (LD-RNN) e de
quatro alternativas nos MESMOS 16 projetos. Comparar contra esses numeros situa o
resultado deste trabalho: diz se a diferenca esta na implementacao ou na classe
de metodo.

A comparacao que mais importa nao e com o LD-RNN, e com o BoW+RF -- saco de
palavras com random forest, a alternativa rasa do proprio artigo e a parente mais
proxima do TF-IDF usado aqui. Ficar acima dela indica que a implementacao esta
correta; a distancia que sobra ate o LD-RNN e entao atribuivel a representacao
aprendida, nao a execucao.

OS NUMEROS PUBLICADOS NAO SE REPRODUZEM. Tawosi, Moussa e Sarro (IEEE TSE 2022)
acharam dois problemas no original (Secao 6 da replicacao): o codigo do Deep-SE
trocava todo story point acima do percentil 90 por um teto ANTES de separar
treino e teste, o que encolhe o erro no teste; e as baselines publicadas so se
reproduzem somando 1,0 as previsoes da media e da mediana. Por isso a comparacao
honesta e contra a coluna NoCut da replicacao (sem teto em lugar nenhum), em
referencia/deepse_replicacao_tawosi.csv. CutTrain (teto so no treino) tambem e
legitima: os replicadores aceitam o teto como limpeza de outliers do treino.

Proveniencia: a Tabela II vem do preprint de 2016 (arXiv:1609.00489), nao da
versao IEEE TSE 2019; a coluna Orig da replicacao, que traz os numeros da versao
TSE, bate com ela projeto a projeto -- e isso e conferido abaixo.
"""
import pandas as pd
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PUBLICADO = RAIZ / "referencia" / "deepse_resultados_publicados.csv"
REPLICACAO = RAIZ / "referencia" / "deepse_replicacao_tawosi.csv"
MEUS = RAIZ / "resultados" / "resultados_completos.csv"

# conferencia de transcricao: medias que o artigo declara no texto da Secao VI-E
DECLARADO = {"LD-RNN": (2.09, 52.66), "LSTM+RF": (2.61, 44.05)}


def main():
    pub = pd.read_csv(PUBLICADO)
    meu = pd.read_csv(MEUS)

    print("=== conferencia da transcricao da Tabela II ===")
    medias = pub.groupby("tecnica")[["MAE", "SA"]].mean()
    for tec, (mae, sa) in DECLARADO.items():
        obtido = medias.loc[tec]
        ok = abs(obtido.MAE - mae) < 0.01 and abs(obtido.SA - sa) < 0.1
        print(f"  {tec:<10} transcrito {obtido.MAE:.3f}/{obtido.SA:.2f}  "
              f"declarado {mae}/{sa}  {'OK' if ok else 'DIVERGE'}")

    p = pub.pivot(index="projeto", columns="tecnica", values="MAE")
    pln = (meu[meu.modelo.str.startswith("PLN")][["projeto", "MAE"]]
           .rename(columns={"MAE": "TF-IDF (aqui)"}))
    med = (meu[meu.modelo == "Mediana historica"][["projeto", "MAE"]]
           .rename(columns={"MAE": "Mediana (aqui)"}))
    j = p.reset_index().merge(pln, on="projeto").merge(med, on="projeto")

    print("\n=== MAE por projeto ===")
    cols = ["projeto", "LD-RNN", "LSTM+RF", "BoW+RF", "TF-IDF (aqui)",
            "Mediana", "Mediana (aqui)"]
    print(j[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\n=== medias nos 16 projetos ===")
    for c in ["LD-RNN", "TF-IDF (aqui)", "LSTM+RF", "Mediana (aqui)",
              "Mediana", "BoW+RF", "Media"]:
        print(f"  {c:<16} {j[c].mean():.3f}")

    print("\n=== o que o artigo original afirma (numeros contestados) ===")
    ldrnn, bow = j["LD-RNN"], j["BoW+RF"]
    aqui, med_aqui, med_pub = j["TF-IDF (aqui)"], j["Mediana (aqui)"], j["Mediana"]
    print(f"  LD-RNN supera a mediana publicada em {int((ldrnn < med_pub).sum())}/16"
          f"  ({(1 - ldrnn.mean() / med_pub.mean()) * 100:+.1f}% de MAE)")
    print(f"  TF-IDF daqui supera o BoW+RF do artigo em {int((aqui < bow).sum())}/16"
          f"  ({(1 - aqui.mean() / bow.mean()) * 100:+.1f}% de MAE)")

    rep = pd.read_csv(REPLICACAO)
    print("\n=== conferencia cruzada: coluna Orig da replicacao x Tabela II ===")
    mapa = {"Deep-SE": "LD-RNN", "Media": "Media", "Mediana": "Mediana"}
    orig = rep.assign(tecnica=rep.tecnica.map(mapa)).merge(
        pub, on=["projeto", "tecnica"])
    diverge = orig[(orig.MAE_Orig - orig.MAE).abs() > 0.005]
    print(f"  {len(orig)} pares conferidos, {len(diverge)} divergentes"
          f"  {'OK' if diverge.empty else '-> ' + ', '.join(diverge.projeto)}")

    r = rep.pivot(index="projeto", columns="tecnica",
                  values=["MAE_NoCut", "MAE_CutTrain"])
    sem_teto, teto_treino = r["MAE_NoCut"], r["MAE_CutTrain"]
    k = j.set_index("projeto").join(sem_teto.add_suffix(" (sem teto)"))
    print("\n=== comparacao honesta: replicacao sem o teto no teste ===")
    print(f"  Deep-SE sem teto        {sem_teto['Deep-SE'].mean():.3f}")
    print(f"  Deep-SE teto so treino  {teto_treino['Deep-SE'].mean():.3f}")
    print(f"  Mediana (replicacao)    {sem_teto['Mediana'].mean():.3f}"
          f"   | mediana daqui {med_aqui.mean():.3f}")
    print(f"  TF-IDF daqui            {aqui.mean():.3f}")
    print(f"\n  Deep-SE sem teto supera a mediana em "
          f"{int((sem_teto['Deep-SE'] < sem_teto['Mediana']).sum())}/16")
    print(f"  TF-IDF daqui supera o Deep-SE sem teto em "
          f"{int((k['TF-IDF (aqui)'] < k['Deep-SE (sem teto)']).sum())}/16")

if __name__ == "__main__":
    main()
