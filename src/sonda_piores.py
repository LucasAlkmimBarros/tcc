"""Sonda 8: nos projetos em que o modelo mais perde, o que mudou do historico para o teste?

Por que esta sonda existe. A figura do ganho por projeto mostra onde o modelo
perde para a mediana, mas nao diz por que. A pergunta do orientador foi: pegar os
piores projetos de cada base e dizer qual e o problema de cada um.

"Pior" e o projeto em que o modelo mais perde para a mediana, em pontos (o ganho
mais negativo), com a mediana calculada no mesmo historico do modelo (80%).

Para cada projeto a sonda compara o historico com o teste:

    mediana dos pontos                       a escala andou?
    % das historias que valem a mediana      a mediana acerta em cheio em quantas?
    % das historias com 0 ou 1 ponto         os pontos encolheram?
    % das historias do tipo "Story"          o tipo de item mudou? (so o TAWOS tem)

A segunda parte olha so o Moodle, o projeto de maior erro nas duas bases: acha a
ultima historia com 20 pontos ou mais, descreve os pontos antes e depois dela e
refaz o teste so com as historias de depois (80/20 no tempo, dentro delas).

Analise descritiva, feita depois de ver os resultados: nao decide nada.

Uso:  python src/sonda_piores.py            (Deep-SE)
      CONJUNTO=tawos python src/sonda_piores.py
"""
import numpy as np
import pandas as pd

from sonda_comum import (dividir_historico_teste, imprimir, limitador,
                         modelo_tfidf, projetos, salvar)

PIORES = 3
PONTOS_ALTOS = 20          # no Moodle, o valor que some depois da troca de escala
MOODLE = {"moodle", "MDL"}


def erros(hist, teste):
    """Erro medio da mediana do historico e do modelo de texto no teste."""
    y = teste.storypoint.to_numpy(dtype=float)
    mediana = float(hist.storypoint.median())
    m = modelo_tfidf().fit(hist.texto, hist.storypoint)
    p = limitador(hist.storypoint)(m.predict(teste.texto))
    return float(np.mean(np.abs(y - mediana))), float(np.mean(np.abs(y - p)))


def pct(serie, condicao):
    return float(100 * condicao(serie).mean())


def comparar(nome, d):
    hist, teste = dividir_historico_teste(d)
    mediana = float(hist.storypoint.median())
    mae_mediana, mae_modelo = erros(hist, teste)
    linha = {
        "projeto": nome, "historias": len(d), "n_teste": len(teste),
        "MAE_mediana": mae_mediana, "MAE_modelo": mae_modelo,
        "ganho": mae_mediana - mae_modelo,
        "mediana_historico": mediana,
        "mediana_teste": float(teste.storypoint.median()),
        "pct_vale_a_mediana_historico": pct(hist.storypoint, lambda s: s == mediana),
        "pct_vale_a_mediana_teste": pct(teste.storypoint, lambda s: s == mediana),
        "pct_0_ou_1_ponto_historico": pct(hist.storypoint, lambda s: s <= 1),
        "pct_0_ou_1_ponto_teste": pct(teste.storypoint, lambda s: s <= 1),
    }
    if "tipo" in d.columns:
        linha["pct_tipo_story_historico"] = pct(hist.tipo, lambda s: s == "Story")
        linha["pct_tipo_story_teste"] = pct(teste.tipo, lambda s: s == "Story")
    return linha


def moodle(d):
    """A troca de escala do Moodle: antes e depois da ultima historia de 20 ou mais."""
    altas = np.where(d.storypoint.to_numpy() >= PONTOS_ALTOS)[0]
    ultima = int(altas.max())
    antes, depois = d.iloc[:ultima + 1], d.iloc[ultima + 1:].reset_index(drop=True)
    hist, teste = dividir_historico_teste(d)

    linhas = []
    for rotulo, parte in (("antes da troca", antes), ("depois da troca", depois)):
        linhas.append({
            "parte": rotulo, "historias": len(parte),
            "primeira": parte.issuekey.iloc[0], "ultima": parte.issuekey.iloc[-1],
            "mediana": float(parte.storypoint.median()),
            "maior_ponto": float(parte.storypoint.max()),
            f"pct_com_{PONTOS_ALTOS}_ou_mais": pct(parte.storypoint,
                                                 lambda s: s >= PONTOS_ALTOS),
            "pct_0_ou_1_ponto": pct(parte.storypoint, lambda s: s <= 1),
            "no_historico_do_artigo": int((hist.index <= ultima).sum()
                                          if rotulo == "antes da troca"
                                          else (hist.index > ultima).sum()),
            "no_teste_do_artigo": int((teste.index <= ultima).sum()
                                      if rotulo == "antes da troca"
                                      else (teste.index > ultima).sum()),
        })
    r = pd.DataFrame(linhas)
    if "created" in d.columns:
        r["data_da_ultima"] = [str(antes.created.iloc[-1])[:10],
                               str(depois.created.iloc[-1])[:10]]

    # o mesmo teste do artigo, so com as historias de depois da troca
    h2, t2 = dividir_historico_teste(depois)
    mae_mediana, mae_modelo = erros(h2, t2)
    dentro = pd.DataFrame([
        {"teste": "como no artigo (projeto inteiro)", "n_teste": len(teste),
         **dict(zip(("MAE_mediana", "MAE_modelo"), erros(hist, teste)))},
        {"teste": "so as historias de depois da troca", "n_teste": len(t2),
         "MAE_mediana": mae_mediana, "MAE_modelo": mae_modelo},
    ])
    return r, dentro


def main():
    linhas, do_moodle = [], None
    for nome, d in projetos():
        linhas.append(comparar(nome, d))
        if nome in MOODLE:
            do_moodle = moodle(d)

    r = pd.DataFrame(linhas).sort_values("ganho").reset_index(drop=True)
    r.insert(1, "entre_os_piores", r.index < PIORES)
    imprimir(r.head(PIORES).drop(columns=["entre_os_piores"]).T,
             f"Os {PIORES} projetos em que o modelo mais perde para a mediana")
    salvar(r, "sonda_piores.csv")

    if do_moodle is not None:
        partes, dentro = do_moodle
        imprimir(partes.T, "Moodle: antes e depois da ultima historia de "
                           f"{PONTOS_ALTOS} pontos ou mais")
        imprimir(dentro, "Moodle: o teste do artigo e o teste so na escala nova")
        salvar(partes, "sonda_piores_moodle.csv")
        salvar(dentro, "sonda_piores_moodle_teste.csv")


if __name__ == "__main__":
    main()
