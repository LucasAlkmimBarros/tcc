"""Nucleo do software de estimativa: so o que os experimentos validaram.

Tres regras, cada uma vinda de um resultado medido (DECISOES.md, RELATORIO.md):

    menos de 5 historias estimadas -> dia zero: o LLM compara a historia nova com
                                      as historias de referencia da equipe
    5 ou mais                      -> a mediana da equipe (com 5 historias ela ja
                                      alcanca o LLM do dia zero)
    25 ou mais                     -> tambem o modelo de texto, ao lado da mediana
                                      (com 25 historias o erro ja e quase o de 1.165)

O modelo, o prompt do dia zero e a leitura da resposta do LLM sao importados dos
scripts do experimento, nao reescritos: o software usa exatamente o que foi
testado. `conferir.py` prova isso refazendo os numeros publicados.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cold_start_comum import limpar, referencias  # noqa: E402
from cold_start_llm import CARTOES, MODELO_GEMINI, Estimador  # noqa: E402
from historias_padrao import historias_padrao, referencias_dos_votos  # noqa: E402,F401
from sonda_comum import (dividir_historico_teste, limitador, metricas,  # noqa: E402
                         modelo_tfidf)

MIN_MEDIANA = 5
MIN_MODELO = 25
MODELO_LOCAL = "qwen2.5:14b"

# nomes de coluna aceitos: o formato dos experimentos e o CSV exportado do Jira
COLUNAS = {
    "issuekey": ["issuekey", "issue key", "key", "chave"],
    "title": ["title", "summary", "titulo", "título", "resumo"],
    "description": ["description", "descricao", "descrição"],
    "storypoint": ["storypoint", "story points", "story point estimate", "story_point",
                   "custom field (story points)", "custom field (story point estimate)"],
    "created": ["created", "criado", "criada", "data de criacao", "data de criação"],
}


def ler_csv(arquivo):
    """Le o CSV, renomeia as colunas para as dos experimentos e ordena no tempo."""
    bruto = pd.read_csv(arquivo)
    por_nome = {str(c).strip().lower(): c for c in bruto.columns}
    achadas = {}
    for nome, opcoes in COLUNAS.items():
        col = next((por_nome[o] for o in opcoes if o in por_nome), None)
        if col is not None:
            achadas[nome] = col
    faltam = [c for c in ("title", "storypoint") if c not in achadas]
    if faltam:
        raise ValueError(f"não achei a coluna de {', '.join(faltam)}. "
                         f"Colunas do arquivo: {', '.join(map(str, bruto.columns))}")

    d = pd.DataFrame({nome: bruto[col] for nome, col in achadas.items()})
    if "issuekey" not in d:
        d["issuekey"] = [f"H-{i + 1}" for i in range(len(d))]
    if "description" not in d:
        d["description"] = ""
    d["storypoint"] = pd.to_numeric(d.storypoint, errors="coerce")

    # um historico salvo pela pagina traz a propria ordem (coluna "ordem"): ele
    # mistura chaves do Jira e coladas (NOVA-3), e o numero da chave nao serve
    salvo = next((c for c in bruto.columns if str(c).strip().lower() == "ordem"), None)
    if salvo is not None:
        d = d.assign(_ordem=pd.to_numeric(bruto[salvo], errors="coerce")) \
             .sort_values("_ordem", kind="stable").drop(columns="_ordem")
        d["texto"] = (d["title"].fillna("").astype(str) + " "
                      + d["description"].fillna("").astype(str)).str.strip()
        return d.reset_index(drop=True)

    # a mesma ordem de caracterizar.carregar: data de criacao, e o numero da
    # chave para desempatar (ou sozinho, quando nao ha data)
    d["ordem"] = pd.to_numeric(d.issuekey.astype(str).str.extract(r"-(\d+)$")[0],
                               errors="coerce")
    with warnings.catch_warnings():   # datas do Jira ("14/Mar/25 10:22 AM") sem formato fixo
        warnings.simplefilter("ignore", UserWarning)
        data = pd.to_datetime(d["created"], errors="coerce") if "created" in d else None
    if data is not None and data.notna().all():
        d = d.assign(_data=data).sort_values(["_data", "ordem"], kind="stable").drop(columns="_data")
    elif d.ordem.notna().all():
        d = d.sort_values("ordem", kind="stable")
    d["texto"] = (d["title"].fillna("").astype(str) + " "
                  + d["description"].fillna("").astype(str)).str.strip()
    return d.reset_index(drop=True)


def _prefixos(d):
    return d.issuekey.astype(str).str.extract(r"^(.+)-\d+$")[0]


def separar(d):
    """(historico, novas): as historias ja estimadas e as que ainda nao tem ponto."""
    tem = d.storypoint.notna()
    return d[tem], d[~tem]


def demonstracao(d):
    """Para um arquivo todo estimado: as 20% mais recentes fazem o papel de novas,
    no mesmo corte dos experimentos."""
    return dividir_historico_teste(d[d.storypoint.notna()])


def modo(n_historico):
    if n_historico < MIN_MEDIANA:
        return "dia zero"
    return "mediana" if n_historico < MIN_MODELO else "mediana e modelo"


def cartao(v):
    """O cartao de Planning Poker mais proximo de cada valor."""
    v, c = np.asarray(v, dtype=float), np.asarray(CARTOES, dtype=float)
    return c[np.abs(v[:, None] - c).argmin(axis=1)]


def historia(titulo, descricao=""):
    """Uma historia digitada (o card colado pelo PO), no mesmo formato das do CSV."""
    d = pd.DataFrame({"issuekey": ["NOVA"], "title": [titulo or ""],
                      "description": [descricao or ""]})
    d["texto"] = (d.title + " " + d.description).str.strip()
    return d


def sugerir(historico, novas, com_modelo=None):
    """Sugestoes para as historias novas a partir do historico da equipe.

    com_modelo=False tira o modelo de texto mesmo com 25 historias ou mais (a
    simulacao da faixa "mediana" na pagina); None segue a regra de `modo`.
    """
    s = novas[["issuekey", "title"]].copy()
    if modo(len(historico)) == "dia zero":
        return s
    y = historico.storypoint.astype(float)
    s["mediana"] = float(y.median())
    if com_modelo is None:
        com_modelo = modo(len(historico)) == "mediana e modelo"
    if com_modelo and len(historico) >= MIN_MODELO:
        m = modelo_tfidf().fit(historico.texto, y)
        valor = limitador(y)(m.predict(novas.texto))
        s["modelo_valor"] = valor
        s["modelo"] = cartao(valor)
    return s


def comparar(sugestoes, reais):
    """No modo demonstracao: erro de cada sugestao contra o valor real."""
    linhas = {"Mediana da equipe": metricas(reais, sugestoes.mediana)}
    if "modelo_valor" in sugestoes:
        linhas["Modelo de texto"] = metricas(reais, sugestoes.modelo_valor)
    return pd.DataFrame(linhas).T


def referencias_do_historico(historico):
    """A historia mais tipica de cada cartao (1 a 13), como no experimento do dia zero."""
    h = historico.copy()
    h["texto_limpo"] = [limpar(t) for t in h.texto]
    return {v: h.loc[i, "texto"] for v, i in referencias(h, "tipica").items()}


def dia_zero(refs, textos, provedor="gemini"):
    """Para cada texto novo, o cartao da referencia que o LLM achou mais proxima.

    refs: {cartao: texto da historia de referencia}. O LLM ve as referencias so em
    ordem, da menor para a maior, e nunca um numero de story point.
    """
    valores = sorted(v for v, t in refs.items() if t and str(t).strip())
    if len(valores) < 2:
        raise ValueError("preencha pelo menos duas histórias de referência")
    est = Estimador(provedor, MODELO_GEMINI if provedor == "gemini" else MODELO_LOCAL)
    textos_refs = [limpar(refs[v]) for v in valores]
    saida = []
    for t in textos:
        v, _, ok, _ = est.prompt_unico(limpar(t), textos_refs, valores)
        saida.append(v if ok else None)
    return saida


# --- o historico que cresce dentro da pagina (versao 2) ---

COLUNAS_BASE = ["issuekey", "title", "description", "storypoint", "texto"]


def vazia():
    """O historico de uma equipe que comeca do zero."""
    return pd.DataFrame({c: pd.Series(dtype=float if c == "storypoint" else object)
                         for c in COLUNAS_BASE})


def equipes(d):
    """Os projetos do CSV (PAG em PAG-12). Cada um tem a sua escala de pontos."""
    return sorted(_prefixos(d).dropna().unique())


def da_equipe(d, prefixo):
    return d[_prefixos(d) == prefixo].reset_index(drop=True)


def proxima_chave(base):
    nums = base.issuekey.astype(str).str.extract(r"^NOVA-(\d+)$")[0].dropna().astype(int)
    return f"NOVA-{nums.max() + 1 if len(nums) else 1}"


def decidir(base, titulo, descricao, pontos, chave=None):
    """O historico com mais uma decisao da equipe.

    Se a chave ja esta no arquivo (uma historia do backlog, sem pontos), ela
    recebe os pontos. Senao, entra no fim, como a historia mais recente.
    """
    base = base.copy()
    if chave and (base.issuekey.astype(str) == chave).any():
        base.loc[base.issuekey.astype(str) == chave, "storypoint"] = float(pontos)
        return base
    nova = historia(titulo, descricao)
    nova["issuekey"] = chave or proxima_chave(base)
    nova["storypoint"] = float(pontos)
    if "created" in base:
        nova["created"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    return pd.concat([base, nova], ignore_index=True)


def para_csv(base):
    """O historico num CSV que a propria pagina le de volta."""
    cols = [c for c in ["issuekey", "created", "title", "description", "storypoint"]
            if c in base]
    return base[cols].assign(ordem=range(1, len(base) + 1)).to_csv(index=False).encode()


# palavras vazias do portugues (sem acento, como o vetorizador as ve), mais as do
# modelo de historia ("como ... quero ... para", "criterios de aceitacao"). So para
# as historias parecidas: o modelo validado continua igual.
VAZIAS_PT = """a ao aos aquela aquelas aquele aqueles aquilo as ate com como da das de dela
delas dele deles depois do dos e ela elas ele eles em entre era essa essas esse esses
esta estas este estes eu foi ha isso isto ja la lhe mais mas me mesmo meu meus minha
minhas muito na nao nas nem no nos nossa nosso num numa o os ou para pela pelas pelo
pelos por quando que quem se sem ser seu seus so sua suas tambem te tem ter um uma uns
umas voce quero queremos possa pode poder deve devem criterios aceitacao esperado""".split()


def _vetorizador_parecidas():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode",
                           stop_words=sorted(ENGLISH_STOP_WORDS | set(VAZIAS_PT)))


def parecidas(historico, texto, k=3):
    """As k historias ja estimadas de texto mais parecido, para comparar na conversa.

    Nao e uma sugestao: copiar o ponto da historia mais parecida errou mais que a
    mediana nos testes (3,38 contra 2,00 no TAWOS; DECISOES.md).
    """
    vazio = pd.DataFrame(columns=["issuekey", "title", "storypoint", "semelhanca"])
    if len(historico) == 0 or not texto.strip():
        return vazio
    try:
        X = _vetorizador_parecidas().fit_transform(list(historico.texto) + [texto])
    except ValueError:      # nenhuma palavra util
        return vazio
    sim = (X[:-1] @ X[-1].T).toarray().ravel()   # vetores normalizados: cosseno
    idx = np.argsort(-sim, kind="stable")[:k]
    out = historico.iloc[idx][["issuekey", "title", "storypoint"]].assign(semelhanca=sim[idx])
    return out[out.semelhanca > 0].reset_index(drop=True)
