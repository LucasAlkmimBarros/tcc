"""Sonda 3: as condicoes do lado da empresa -- escala estavel, rotulo coerente,
vocabulario estavel.

Por que esta sonda existe. Volume de historias, medido na sonda 1, nao e
requisito. Mas tres outras condicoes sao verificaveis ANTES de prometer acuracia
a um cliente, e cada uma impoe um teto que nenhum modelo supera.

A) Deriva da escala. Story point e relativo e a equipe recalibra com o tempo.
   Compara-se a mediana do primeiro terco com a do ultimo terco de cada projeto.
   Se a escala anda, um modelo treinado em rotulo velho esta calibrado errado --
   o que nao invalida o produto, mas define a cadencia de retreino que ele
   precisa ter para continuar valendo.

B) Ruido de rotulo. Textos identicos que receberam story points diferentes medem
   o quanto a equipe discorda de si mesma. Esse desacordo e um teto de acuracia:
   se a mesma historia recebeu 3 e 23, nenhum modelo acerta as duas. E o
   diagnostico pre-venda mais honesto que existe.

C) Deriva de vocabulario. O TF-IDF so reconhece palavras vistas no treino.
   Mede-se de tres formas, porque elas respondem perguntas diferentes:
     - tipos distintos de palavra do teste ausentes do treino (palavras raras
       pesam muito aqui: sao muitas, mas aparecem pouco);
     - ocorrencias de palavra nunca vista (quanto do texto corrido e novo);
     - termos que o modelo extrai e que ficam fora do vocabulario do TF-IDF
       ajustado (limitado a 5.000 termos, min_df=2): o que o modelo de fato
       descarta. E esta ultima que mede a cegueira do modelo.
   Escrever "X% das palavras sao novas" com o numero de tipos distintos e
   enganoso: confunde tipo com ocorrencia.
"""
import numpy as np
import pandas as pd

from sonda_comum import (dividir_historico_teste, imprimir, modelo_tfidf,
                         projetos, salvar, SAIDA)


def deriva_escala(d):
    n = len(d)
    inicio = d.iloc[:n // 3].storypoint.median()
    fim = d.iloc[-(n // 3):].storypoint.median()
    return inicio, fim, (fim - inicio) / inicio * 100 if inicio else np.nan


def ruido_rotulo(d):
    """Grupos de texto identico e quanto os story points divergem dentro deles."""
    g = (d.assign(chave=d.texto.str.lower().str.strip())
         .groupby("chave")["storypoint"].agg(["count", "nunique", "min", "max"]))
    repetidos = g[g["count"] > 1]
    divergentes = repetidos[repetidos["nunique"] > 1]
    amplitude = (divergentes["max"] - divergentes["min"])
    return len(repetidos), len(divergentes), (amplitude.mean() if len(divergentes) else 0.0), \
           (amplitude.max() if len(divergentes) else 0.0)


def deriva_vocabulario(hist, teste):
    """Tres medidas de vocabulario novo no teste (ver C no topo do modulo)."""
    palavras = lambda s: " ".join(s.fillna("").str.lower()).split()
    vocab_hist = set(palavras(hist.texto))
    ocorrencias = palavras(teste.texto)
    tipos = set(ocorrencias)
    tipos_novos = sum(1 for w in tipos if w not in vocab_hist) / len(tipos) * 100
    ocorr_novas = sum(1 for w in ocorrencias if w not in vocab_hist) / len(ocorrencias) * 100

    tfidf = modelo_tfidf().named_steps["tfidfvectorizer"].fit(hist.texto)
    analisar, vocab = tfidf.build_analyzer(), set(tfidf.vocabulary_)
    termos = [w for txt in teste.texto for w in analisar(txt)]
    fora = sum(1 for w in termos if w not in vocab) / len(termos) * 100
    return tipos_novos, ocorr_novas, fora


def main():
    linhas = []
    for nome, d in projetos():
        hist, teste = dividir_historico_teste(d)
        ini, fim, var = deriva_escala(d)
        rep, div, amp_media, amp_max = ruido_rotulo(d)
        linhas.append({
            "projeto": nome, "issues": len(d),
            "sp_mediana_inicio": ini, "sp_mediana_fim": fim, "deriva_pct": var,
            "textos_repetidos": rep, "com_sp_divergente": div,
            "amplitude_media": amp_media, "amplitude_max": amp_max,
            **dict(zip(["tipos_novos_pct", "ocorrencias_novas_pct",
                        "fora_vocab_tfidf_pct"], deriva_vocabulario(hist, teste))),
            "descricao_vazia_pct": (d.description.isna() |
                                    d.description.astype(str).str.strip()
                                    .isin(["", "NULL"])).mean() * 100,
        })

    r = pd.DataFrame(linhas)

    imprimir(r.set_index("projeto")[["sp_mediana_inicio", "sp_mediana_fim", "deriva_pct"]]
             .sort_values("deriva_pct"),
             "A) deriva da escala: mediana do 1o terco vs. ultimo terco", 1)
    graves = (r.deriva_pct.abs() >= 33).sum()
    print(f"\n  projetos com deriva >= 33%: {graves}/16"
          f"   |   deriva absoluta media: {r.deriva_pct.abs().mean():.0f}%")

    imprimir(r.set_index("projeto")[["textos_repetidos", "com_sp_divergente",
                                     "amplitude_media", "amplitude_max"]]
             .sort_values("com_sp_divergente", ascending=False),
             "B) ruido de rotulo: texto identico, story point diferente", 1)
    print(f"\n  total de grupos divergentes: {int(r.com_sp_divergente.sum())}"
          f"   |   maior amplitude observada: {r.amplitude_max.max():.0f} pontos")

    imprimir(r.set_index("projeto")[["tipos_novos_pct", "ocorrencias_novas_pct",
                                     "fora_vocab_tfidf_pct", "descricao_vazia_pct"]]
             .sort_values("fora_vocab_tfidf_pct", ascending=False),
             "C) deriva de vocabulario e completude do texto", 1)
    print(f"\n  tipos distintos novos       : {r.tipos_novos_pct.mean():.1f}%")
    print(f"  ocorrencias nunca vistas    : {r.ocorrencias_novas_pct.mean():.1f}%")
    print(f"  fora do vocabulario TF-IDF  : {r.fora_vocab_tfidf_pct.mean():.1f}%"
          "   <- o que o modelo descarta")
    print(f"  descricao vazia             : {r.descricao_vazia_pct.mean():.1f}%")

    # a deriva explica onde o modelo vence a mediana?
    caminho = SAIDA / "resultados_completos.csv"
    if caminho.exists():
        res = pd.read_csv(caminho)
        pln = res[res.modelo.str.startswith("PLN")][["projeto", "MAE"]]
        med = (res[res.modelo == "Mediana historica"][["projeto", "MAE"]]
               .rename(columns={"MAE": "MAE_mediana"}))
        j = r.merge(pln, on="projeto").merge(med, on="projeto")
        j["ganho_pct"] = (j.MAE_mediana - j.MAE) / j.MAE_mediana * 100
        print("\n=== as condicoes explicam onde o modelo vence a mediana? ===")
        for col in ["deriva_pct", "fora_vocab_tfidf_pct",
                    "descricao_vazia_pct", "issues"]:
            print(f"  {col:<28} Spearman rho = "
                  f"{j[col].corr(j.ganho_pct, method='spearman'):+.3f}")

    salvar(r, "sonda_consistencia.csv")


if __name__ == "__main__":
    main()
