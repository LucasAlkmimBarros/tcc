"""Previsoes historia a historia no teste: a mediana e o modelo de texto.

As analises que precisam do erro de cada historia (historias grandes, aviso,
sensibilidade, tabela central) leem este arquivo, em vez de treinar o modelo de
novo cada uma. Tres condicoes, nas mesmas historias de teste (os 20% mais
recentes de cada projeto):

  mediana historica                        a mediana dos 60% mais antigos, como
                                           no Deep-SE e na replicacao
  base (TF-IDF + SVR)                      o modelo do software: TF-IDF + SVR
                                           linear, treinado nos 80% mais antigos
  TF-IDF original (experimento principal)  o modelo do experimento.py (escolha
                                           entre tres regressores pela
                                           validacao), refeito para guardar as
                                           previsoes

O script confere que o MAE por projeto do modelo refeito bate com
resultados_completos.csv.

Saida: previsoes_teste.csv
Uso:  python src/previsoes.py ; CONJUNTO=tawos python src/previsoes.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.svm import LinearSVR

from conjunto import SAIDA
from experimento import construir_modelos, dividir, SEMENTE
from sonda_comum import limitador, modelo_tfidf, projetos, salvar


def modelo_do_software(historico, teste):
    """TF-IDF + SVR linear, treinado em `historico`, previsao limitada ao treino."""
    y = historico.storypoint.values.astype(float)
    limitar = limitador(y)
    tfidf = modelo_tfidf().named_steps["tfidfvectorizer"]
    X_tr, X_te = tfidf.fit_transform(historico.texto), tfidf.transform(teste.texto)
    svr = LinearSVR(C=1.0, max_iter=5000, random_state=SEMENTE).fit(X_tr, y)
    return limitar(svr.predict(X_te))


def tfidf_original(treino, val, teste):
    """O modelo do experimento principal, refeito para guardar as previsoes."""
    limitar = limitador(treino.storypoint)
    melhor, melhor_mae = None, np.inf
    for rotulo, m in construir_modelos(SEMENTE).items():
        m.fit(treino.texto, treino.storypoint)
        e = mean_absolute_error(val.storypoint, limitar(m.predict(val.texto)))
        if e < melhor_mae:
            melhor, melhor_mae = rotulo, e
    final = construir_modelos(SEMENTE)[melhor]
    final.fit(pd.concat([treino.texto, val.texto]),
              pd.concat([treino.storypoint, val.storypoint]))
    return limitar(final.predict(teste.texto)), melhor


def main():
    prev = []
    for nome, d in projetos():
        treino, val, teste = dividir(d)
        historico = pd.concat([treino, val])

        def registrar(cond, p):
            prev.extend({"projeto": nome, "issuekey": k, "storypoint": y, "condicao": cond,
                         "previsao": round(float(x), 4)}
                        for k, y, x in zip(teste.issuekey, teste.storypoint, p))

        registrar("mediana historica", np.full(len(teste), treino.storypoint.median()))
        registrar("base (TF-IDF + SVR)", modelo_do_software(historico, teste))
        p_orig, escolhido = tfidf_original(treino, val, teste)
        registrar("TF-IDF original (experimento principal)", p_orig)
        print(f"  {nome}  (experimento escolheu {escolhido})", flush=True)

    p = pd.DataFrame(prev)
    mae = (p.storypoint - p.previsao).abs().groupby([p.condicao, p.projeto]).mean().unstack(0)

    # conferencia: o TF-IDF original refeito bate com o experimento principal
    exp = pd.read_csv(SAIDA / "resultados_completos.csv")
    exp = exp[exp.modelo.str.startswith("PLN")].set_index("projeto").MAE
    dif = (mae["TF-IDF original (experimento principal)"] - exp).abs().max()
    print(f"\nconferencia com resultados_completos.csv: maior diferenca de MAE {dif:.2e}")
    assert dif < 1e-4, "o TF-IDF refeito nao bate com o experimento principal"  # 4 casas

    print("\nMAE medio dos projetos:")
    print(mae.mean().to_string(float_format=lambda x: f"{x:.3f}"))
    salvar(p, "previsoes_teste.csv")


if __name__ == "__main__":
    main()
