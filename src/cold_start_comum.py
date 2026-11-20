"""Infraestrutura comum do cold start (fase 3 do PLANO.md).

O cenario: uma equipe no dia zero, sem historico de estimativas que sirva para
treinar um modelo ou calcular a sua mediana. O que ela tem e um punhado de
historias de referencia, uma para cada valor da escala, que ela mesma escolhe.
Este modulo fixa tres coisas que todos os scripts da fase usam igual:

1. A amostra de teste. Os ultimos 20% cronologicos de cada projeto (o mesmo
   teste do experimento principal), com no maximo 100 historias sorteadas com
   semente fixa, para caber na cota gratuita da API.
2. As referencias. Para cada valor de Fibonacci (1, 2, 3, 5, 8, 13) que aparece
   nos 60% mais antigos do projeto, uma historia daquele valor. Duas formas de
   escolher: sorteio (repetido 5 vezes, para medir quanto a escolha pesa) e a
   historia mais tipica do valor (a mais proxima do centro das historias
   daquele valor, em TF-IDF). Valores fora de Fibonacci (0, 0,5, 20, 40, 100)
   nao viram referencia: o plano fixou 5 ou 6 referencias e a escala de
   Fibonacci, e o custo disso (nao conseguir prever 20 ou 40) entra no erro.
   Escolher pelo rotulo do treino simula a equipe escolhendo suas referencias.
3. O texto que vai ao LLM. Titulo + descricao, sem URL e sem chave de issue
   (MESOS-336), cortado em 1.500 caracteres. Sem isso o LLM poderia reconhecer
   a issue publica em vez de avaliar o texto. O prompt nao acrescenta o nome do
   projeto; o nome do produto que ja aparece no texto fica, porque faz parte da
   descricao (trocar a sigla como palavra estragaria textos comuns: "server",
   "cli").
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sonda_comum import projetos
from experimento import dividir, SEMENTE

FIBONACCI = [1, 2, 3, 5, 8, 13]
MAX_TESTE = 100
MAX_CARACTERES = 1500
SORTEIOS = 5

_URL = re.compile(r"(https?://|www\.)\S+", re.I)
_CHAVE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")


def limpar(texto):
    """Texto do prompt: sem URL e sem chave de issue."""
    t = _URL.sub("[link]", texto or "")
    t = _CHAVE.sub("[issue]", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) <= MAX_CARACTERES else t[:MAX_CARACTERES].rstrip() + " [...]"


def amostra_teste(teste, semente=SEMENTE):
    """Ate MAX_TESTE historias do teste, sorteadas e devolvidas em ordem cronologica."""
    if len(teste) <= MAX_TESTE:
        return teste
    idx = np.random.default_rng(semente).choice(len(teste), MAX_TESTE, replace=False)
    return teste.iloc[np.sort(idx)]


def vetorizador():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000,
                           sublinear_tf=True, stop_words="english",
                           strip_accents="unicode")


def referencias(treino, forma, sorteio=0, vet=None):
    """Uma historia de treino por valor de Fibonacci presente: {valor: indice}.

    forma = "tipica"  -> a mais proxima do centro das historias do valor (TF-IDF
                         ajustado so no texto do treino, sem rotulo)
    forma = "sorteio" -> uma ao acaso; `sorteio` (0..4) e a semente
    """
    rng = np.random.default_rng(sorteio)
    if forma == "tipica" and vet is None:
        vet = vetorizador().fit(treino.texto_limpo)
    escolha = {}
    for v in FIBONACCI:
        grupo = treino[treino.storypoint == v]
        if grupo.empty:
            continue
        if forma == "sorteio":
            escolha[v] = grupo.index[rng.integers(len(grupo))]
        else:
            X = vet.transform(grupo.texto_limpo)
            centro = np.asarray(X.mean(axis=0))
            sim = (X @ centro.T).ravel()
            escolha[v] = grupo.index[int(np.argmax(sim))]
    return escolha


def conjuntos_de_referencia(treino):
    """As 6 formas de escolher referencias: a tipica e 5 sorteios."""
    vet = vetorizador().fit(treino.texto_limpo)
    r = {"tipica": referencias(treino, "tipica", vet=vet)}
    for s in range(SORTEIOS):
        r[f"sorteio{s}"] = referencias(treino, "sorteio", sorteio=s)
    return r


def projetos_cold_start():
    """Itera (sigla, treino, historico, teste_amostrado) com o texto limpo pronto.

    treino    : 60% mais antigos (de onde saem as referencias)
    historico : 80% mais antigos (o que a equipe teria antes do teste)
    teste     : amostra de ate 100 historias dos 20% mais recentes
    """
    for sigla, d in projetos():
        d = d.copy()
        d["texto_limpo"] = [limpar(t) for t in d.texto]
        treino, val, teste = dividir(d)
        historico = d.iloc[:len(treino) + len(val)]
        yield sigla, treino, historico, amostra_teste(teste)
