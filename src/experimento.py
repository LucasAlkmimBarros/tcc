"""Etapa 2/3: validacao empirica da estimativa de esforco por PLN.

Compara o modelo proposto (TF-IDF sobre titulo+descricao da user story +
regressor) com os baselines empiricos que ele pretende substituir: a media e a
mediana historica de story points da equipe, e o palpite aleatorio.

Protocolo: divisao cronologica 60/20/20 (treino/validacao/teste) por projeto,
conforme Choetkiertikul et al. (IEEE TSE 2018). Metricas: MAE, MdAE e SA.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from caracterizar import carregar

from conjunto import DADOS, SAIDA
SEMENTE = 42
N_RODADAS_ALEATORIO = 1000


def dividir(d):
    """Divisao cronologica 60/20/20 sem embaralhamento.

    O embaralhamento aleatorio vazaria informacao do futuro para o treino,
    inflando artificialmente o desempenho; a ordem de abertura das issues e
    preservada para simular o uso real dentro de um projeto em andamento.
    """
    n = len(d)
    i, j = int(n * 0.6), int(n * 0.8)
    return d.iloc[:i], d.iloc[i:j], d.iloc[j:]


def mdae(y, p):
    return float(np.median(np.abs(np.asarray(y) - np.asarray(p))))


def mae_aleatorio(y_treino, y_teste, rng):
    """MAE do palpite aleatorio (Shepperd & MacDonell, 2012): para cada issue de
    teste sorteia-se um valor real observado no treino. Base de calculo do SA."""
    y_treino, y_teste = np.asarray(y_treino), np.asarray(y_teste)
    erros = [
        np.abs(y_teste - rng.choice(y_treino, size=len(y_teste), replace=True)).mean()
        for _ in range(N_RODADAS_ALEATORIO)
    ]
    return float(np.mean(erros))


def construir_modelos(semente):
    """Cada modelo vetoriza titulo+descricao com TF-IDF e regride o story point."""
    tfidf = lambda: TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=5000,
        sublinear_tf=True, stop_words="english", strip_accents="unicode",
    )
    return {
        "TF-IDF + Ridge": make_pipeline(tfidf(), Ridge(alpha=1.0)),
        "TF-IDF + SVR": make_pipeline(tfidf(), LinearSVR(C=1.0, max_iter=5000, random_state=semente)),
        "TF-IDF + Random Forest": make_pipeline(
            tfidf(), RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                           random_state=semente, n_jobs=-1)),
    }


def avaliar_projeto(nome, caminho):
    rng = np.random.default_rng(SEMENTE)
    d = carregar(caminho)
    treino, val, teste = dividir(d)

    y_tr, y_val, y_te = treino.storypoint, val.storypoint, teste.storypoint
    piso, teto = y_tr.min(), y_tr.max()
    limitar = lambda p: np.clip(p, piso, teto)

    base_aleatorio = mae_aleatorio(y_tr, y_te, rng)
    sa = lambda m: (1 - m / base_aleatorio) * 100

    resultados = []
    registrar = lambda modelo, pred: resultados.append({
        "projeto": nome, "modelo": modelo,
        "MAE": mean_absolute_error(y_te, pred),
        "MdAE": mdae(y_te, pred), "SA": sa(mean_absolute_error(y_te, pred)),
    })

    # --- baselines: as heuristicas que o modelo pretende substituir ---
    # o palpite aleatorio nao produz um vetor unico de predicoes: seu MAE e a
    # media de N_RODADAS_ALEATORIO sorteios, e por definicao seu SA e zero.
    resultados.append({"projeto": nome, "modelo": "Palpite aleatorio",
                       "MAE": base_aleatorio, "MdAE": np.nan, "SA": 0.0})
    registrar("Media historica", np.full(len(y_te), y_tr.mean()))
    registrar("Mediana historica", np.full(len(y_te), y_tr.median()))

    # --- modelo proposto: selecao do regressor pelo MAE de validacao ---
    x_tr, x_val = treino.texto, val.texto
    melhor, melhor_mae = None, np.inf
    for rotulo, modelo in construir_modelos(SEMENTE).items():
        modelo.fit(x_tr, y_tr)
        m = mean_absolute_error(y_val, limitar(modelo.predict(x_val)))
        if m < melhor_mae:
            melhor, melhor_mae = rotulo, m

    # reajuste em treino+validacao antes de medir no teste
    final = construir_modelos(SEMENTE)[melhor]
    final.fit(pd.concat([x_tr, x_val]), pd.concat([y_tr, y_val]))
    registrar(f"PLN ({melhor})", limitar(final.predict(teste.texto)))

    print(f"  {nome:<20} n={len(d):>5}  treino/val/teste={len(treino)}/{len(val)}/{len(teste)}"
          f"  melhor={melhor}")
    return resultados


def main():
    todos = []
    print("Executando validacao empirica por projeto:\n")
    for arq in sorted(DADOS.glob("*.csv")):
        todos += avaliar_projeto(arq.stem, arq)

    r = pd.DataFrame(todos)
    r.to_csv(SAIDA / "resultados_completos.csv", index=False)

    # Se a selecao por validacao escolheu regressores diferentes entre projetos,
    # as medias por nome de modelo misturariam projetos diferentes. Para resumir,
    # trata-se o modelo proposto como um so: o que a validacao escolheu em cada.
    if r[r.modelo.str.startswith("PLN")].modelo.nunique() > 1:
        r["modelo"] = r.modelo.where(~r.modelo.str.startswith("PLN"),
                                     "PLN (melhor por validacao)")

    print("\n=== MAE por projeto e modelo ===")
    piv = r.pivot(index="projeto", columns="modelo", values="MAE")
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n=== SA (%) do modelo de PLN ===")
    pln = r[r.modelo.str.startswith("PLN")]
    print(pln[["projeto", "modelo", "MAE", "MdAE", "SA"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print(f"\n=== Media geral ({r.projeto.nunique()} projetos) ===")
    print(r.groupby("modelo")[["MAE", "MdAE", "SA"]].mean()
          .sort_values("MAE").to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
