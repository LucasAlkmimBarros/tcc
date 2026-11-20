"""Infraestrutura compartilhada das sondas diagnosticas.

O experimento principal pergunta "o modelo e melhor que o baseline?". As sondas
perguntam "por que, e sob que condicoes" -- que e o que determina se o software
e aplicavel numa empresa. Elas nao substituem o experimento: o complementam.

Duas decisoes garantem comparabilidade com a Tabela 2 do artigo:

1. O conjunto de teste e o MESMO (ultimos 20% cronologicos de cada projeto).
2. O modelo e exatamente o que o artigo reporta -- importado de `experimento`
   em vez de reescrito -- para que as sondas nao possam divergir dele por
   descuido de configuracao.

Alem do MAE, as sondas medem duas metricas voltadas a decisao de quem compra o
software, que o MAE esconde: com que frequencia o erro fica dentro de 1 ponto, e
com que frequencia a previsao cai no cartao certo de Planning Poker.
"""
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from caracterizar import carregar
from experimento import construir_modelos, SEMENTE

from conjunto import DADOS, NOME, RAIZ, SAIDA
CACHE = RAIZ / "cache" / NOME if NOME != "deepse" else RAIZ / "cache"

# cartoes de Planning Poker: a escala em que a equipe realmente decide
CARTOES = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89])

FRACAO_TESTE = 0.2


def projetos():
    """Itera (nome, dataframe ordenado cronologicamente) sobre os 16 projetos."""
    for arq in sorted(DADOS.glob("*.csv")):
        yield arq.stem, carregar(arq)


def dividir_historico_teste(d):
    """Separa historico e teste no MESMO ponto de corte do experimento principal.

    `experimento.dividir` usa 60/20/20; o teste dele comeca em 0,8n. Aqui o
    historico e todo o passado (0,8n) porque as sondas nao selecionam modelo por
    validacao -- quando precisam (FLAML), a validacao e recortada de dentro do
    historico, tambem em ordem temporal.
    """
    corte = int(len(d) * (1 - FRACAO_TESTE))
    return d.iloc[:corte], d.iloc[corte:]


def modelo_tfidf():
    """O modelo do artigo: TF-IDF sobre titulo+descricao + SVR linear.

    Fixa-se o SVR em vez de reselecionar entre Ridge/SVR/RF porque ele venceu a
    selecao por validacao nos 16 projetos -- a escolha do regressor nao e fonte
    de variacao, e fixa-la torna as sondas mais rapidas e mais legiveis.
    """
    return construir_modelos(SEMENTE)["TF-IDF + SVR"]


def cartao(v):
    """Arredonda para o cartao de Planning Poker mais proximo."""
    v = np.asarray(v, dtype=float)
    return CARTOES[np.abs(v[:, None] - CARTOES).argmin(axis=1)]


def metricas(y, p):
    """MAE mais as duas metricas que uma empresa consegue interpretar.

    dentro_1     : % de previsoes que erram por no maximo 1 story point.
    cartao_exato : % que cai no mesmo cartao de Planning Poker que a verdade.
                   Compara-se cartao(previsao) com cartao(verdade) porque 13,5%
                   dos rotulos do benchmark nao estao na escala de Fibonacci.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "dentro_1": float((np.abs(y - p) <= 1).mean() * 100),
        "cartao_exato": float((cartao(p) == cartao(y)).mean() * 100),
    }


def limitador(y_treino):
    """Restringe a previsao ao intervalo observado no treino, como no artigo."""
    piso, teto = float(np.min(y_treino)), float(np.max(y_treino))
    return lambda p: np.clip(p, piso, teto)


MODELO_EMB = "sentence-transformers/all-MiniLM-L6-v2"


def embeddings(nome, textos):
    """Codifica os textos com MiniLM, com cache em disco por projeto.

    Usado pela busca das historias mais parecidas (sonda_vizinhos.py), a mesma
    busca que entregaria historias a um LLM num RAG.

    Limitacao registrada: o MiniLM trunca em 256 tokens, e 10% das issues do
    benchmark passam disso -- parte da descricao longa nao e lida.
    """
    CACHE.mkdir(exist_ok=True)
    arq = CACHE / f"emb_{nome}.npy"
    if arq.exists():
        return np.load(arq)
    from sentence_transformers import SentenceTransformer
    if not hasattr(embeddings, "_m"):
        print(f"  (carregando {MODELO_EMB})", flush=True)
        embeddings._m = SentenceTransformer(MODELO_EMB)
    v = embeddings._m.encode(list(textos), batch_size=64, show_progress_bar=False,
                             normalize_embeddings=True)
    np.save(arq, v)
    return v


def imprimir(df, titulo, casas=2):
    print(f"\n=== {titulo} ===")
    print(df.to_string(float_format=lambda v: f"{v:.{casas}f}"))


def salvar(df, nome):
    SAIDA.mkdir(exist_ok=True)
    caminho = SAIDA / nome
    df.to_csv(caminho, index=False)
    print(f"\n-> {caminho}")
