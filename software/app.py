"""Pagina do software (versao 2): o PO estima, registra a decisao da equipe, e o
historico cresce dentro da propria pagina.

O que mudou sobre a v1 (software_v1/), vindo de usar a pagina como PO
(AVALIACAO_PO.md):
- "Estimar" e a primeira aba: colar o card, ver a sugestao, as historias
  parecidas do historico e salvar os pontos que a equipe decidiu;
- uma equipe nova comeca do zero e passa pelas tres faixas dentro da pagina;
- o backlog do arquivo ganha uma coluna de decisao, salva de uma vez;
- "Teste no historico" (o antigo modo demonstracao) para qualquer equipe;
- menos texto, e sem o menu de desenvolvedor do Streamlit;
- a sugestao aparece depois do voto da equipe, para nao ancorar a estimativa
  (PLANO.md, "Apontar as historias grandes").

Uso (de dentro de experimento/):  .venv/bin/streamlit run software/app.py
"""
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import previsao as pv  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
EXEMPLOS = {nome: arq for nome, arq in {
    "RoomBook (base construída)": RAIZ / "base_sintetica" / "dados" / "roombook.csv",
    "Mesos (Deep-SE, código aberto)": RAIZ / "dados" / "mesos.csv",
    "Moodle (TAWOS, código aberto)": RAIZ / "tawos" / "dados" / "MDL.csv",
}.items() if arq.exists()}
NOVA_EQUIPE = "Equipe nova (começar do zero)"
CARTOES = [1, 2, 3, 5, 8, 13, 20, 40]      # o baralho das decisoes
CARTOES_REF = [1, 2, 3, 5, 8, 13]          # as referencias do dia zero
MAX_DIA_ZERO = 10
MIN_TESTE = 30                              # 20% de 30: 6 historias de teste
GRANDE = 13
SEM_DECISAO = "–"                           # celula do backlog ainda nao decidida

FAIXAS = ["dia zero", "mediana", "mediana e modelo"]
MINIMO = {"dia zero": 0, "mediana": pv.MIN_MEDIANA, "mediana e modelo": pv.MIN_MODELO}
NOME = {"dia zero": "Dia zero", "mediana": "Mediana",
        "mediana e modelo": "Mediana e modelo de texto"}
ROTULO = {"dia zero": f"**Dia zero** · menos de {pv.MIN_MEDIANA} estimadas",
          "mediana": f"**Mediana** · {pv.MIN_MEDIANA} a {pv.MIN_MODELO - 1} estimadas",
          "mediana e modelo": f"**Mediana e modelo de texto** · {pv.MIN_MODELO} ou mais"}
LLM = {"gemini": "Gemini (envia o texto ao Google)",
       "ollama": "Qwen local (não envia nada para fora)"}
DICA_CARD = "Cole a descrição do card, com os critérios de aceitação, se houver."
POR_QUE_DUAS = ("Nenhuma das duas foi melhor em tudo nos testes com as bases públicas: a mediana "
                "erra por mais de 1 ponto menos vezes, e o modelo comete menos erros grandes.")
PERTO = ("A sugestão é o cartão mais próximo do valor calculado. Nos testes, 7 em cada 10 "
         "sugestões do modelo caíram no mesmo cartão da mediana, e ele quase nunca sugeriu "
         "13 ou mais: história grande depende da conversa.")
DIA_ZERO_NOS_TESTES = ("Nos testes, o dia zero errou o mesmo que chutar o valor típico de "
                       "outras equipes; com poucas histórias estimadas, a mediana da própria "
                       "equipe passa à frente.")
FORMAS_REF = {"padrao": "Votar em 10 histórias prontas", "nossas": "Escrever as nossas"}
VOTO_ANTES = ("**Primeiro o voto da equipe, depois a sugestão.** Um número visto antes do "
              "voto puxa a estimativa da equipe para perto dele (efeito de ancoragem). E o "
              "modelo e a mediana ficam quase sempre no valor típico: vistos antes, puxariam "
              "para baixo justamente as histórias grandes.")


def num(v):
    """Numero sem ".0" e com virgula decimal: 5 -> "5", 2.5 -> "2,5"."""
    return f"{float(v):g}".replace(".", ",")


def historias(n):
    return f"{n} história{'s' if n != 1 else ''} estimada{'s' if n != 1 else ''}"


def curto(t, n=110):
    return t[:n] + ("…" if len(t) > n else "")


def com_pontos(t, colunas):
    """Mostra os pontos com num(), mas guarda o numero: clicar na coluna ordena 100
    acima de 8, e nao como texto."""
    return t.style.format({c: num for c in colunas})


@st.cache_data(show_spinner=False)
def ler(conteudo):
    return pv.ler_csv(io.BytesIO(conteudo))


@st.cache_data(show_spinner=False)
def sugerir(hist, novas, com_modelo):
    return pv.sugerir(hist, novas, com_modelo=com_modelo)


st.set_page_config(page_title="Estimativa de story points", layout="wide")
st.html("<style>[data-testid='stMainBlockContainer']{max-width:1180px;padding-top:2.5rem}"
        "</style>")
bases = st.session_state.setdefault("bases", {})        # historico de cada fonte
decisoes = st.session_state.setdefault("decisoes", {})  # decisoes salvas na sessao
desfazer = st.session_state.setdefault("desfazer", {})  # historicos anteriores, para desfazer

# --- de onde vem o historico ---
with st.sidebar:
    st.header("Dados")
    arquivo = st.file_uploader("CSV exportado do Jira", type="csv")
    opcao = st.selectbox("ou", [NOVA_EQUIPE] + list(EXEMPLOS), index=None,
                         placeholder="Equipe nova ou exemplo pronto")

id_fonte = erro_leitura = carregado = None
if arquivo is not None and opcao:
    with st.sidebar:
        st.caption("Usando o arquivo enviado. Para usar a opção acima, remova o arquivo.")
if arquivo is not None:
    id_fonte = f"arquivo::{arquivo.name}::{arquivo.size}"
    try:
        carregado = ler(arquivo.getvalue())
    except ValueError as e:
        erro_leitura = str(e)
elif opcao == NOVA_EQUIPE:
    id_fonte, carregado = "nova", pv.vazia()
elif opcao:
    id_fonte, carregado = f"exemplo::{opcao}", ler(EXEMPLOS[opcao].read_bytes())

equipe = None
if carregado is not None and len(pv.equipes(carregado)) > 1:
    with st.sidebar:
        equipe = st.selectbox("Equipe (projeto do Jira)", pv.equipes(carregado),
                              help="Cada equipe tem a sua escala de pontos: o software "
                                   "olha uma de cada vez.")
    carregado = pv.da_equipe(carregado, equipe)

id_base = f"{id_fonte}::{equipe}" if id_fonte else None
PREFIXO = equipe or (pv.equipes(carregado)[0] if carregado is not None
                     and len(pv.equipes(carregado)) == 1 else "PROJ")
if id_base and id_base not in bases:
    bases[id_base] = carregado
d = bases.get(id_base)

estimadas = backlog = n = faixa_equipe = None
if d is not None:
    estimadas, backlog = pv.separar(d)
    n = len(estimadas)
    faixa_equipe = pv.modo(n)

st.title("Estimativa de story points")
st.caption("Sugere story points a partir do histórico da equipe. A decisão é sempre da equipe.")
if erro_leitura:
    st.error(f"Não consegui ler o arquivo: {erro_leitura}")

# --- a regua: marca a faixa da equipe e acompanha quando ela muda ---
chave_regua = f"regua::{id_base}"
registrada = st.session_state.setdefault("faixa_registrada", {})
if chave_regua not in st.session_state:
    st.session_state[chave_regua] = faixa_equipe
elif faixa_equipe and registrada.get(chave_regua) not in (None, faixa_equipe):
    if st.session_state[chave_regua] == registrada[chave_regua]:
        st.session_state[chave_regua] = faixa_equipe
    st.toast(f"Com {historias(n)}, a equipe está na faixa {NOME[faixa_equipe]}.",
             icon=":material/trending_up:")
registrada[chave_regua] = faixa_equipe


def voltar_para_equipe():
    # clicar de novo na faixa marcada desmarca; a pagina volta para a da equipe
    if st.session_state[chave_regua] is None:
        st.session_state[chave_regua] = faixa_equipe


faixa = st.segmented_control("O que o histórico da equipe permite", FAIXAS,
                             format_func=ROTULO.get, key=chave_regua,
                             on_change=voltar_para_equipe, width="stretch")
acima = d is not None and faixa is not None and MINIMO[faixa] > n
if d is not None and faixa is not None:
    if faixa == faixa_equipe:
        partes = [f"**{historias(n)}**"]
        if n >= pv.MIN_MEDIANA:
            partes.append(f"mediana **{num(estimadas.storypoint.median())}**")
        if len(backlog):
            partes.append(f"**{len(backlog)}** no backlog sem pontos")
        st.markdown(" · ".join(partes))
    elif acima:
        st.info(f"Esta faixa precisa de {MINIMO[faixa]} histórias estimadas. A equipe tem "
                f"{n}: faltam {MINIMO[faixa] - n}.")
    else:
        st.markdown(f"**Simulação** da faixa *{NOME[faixa]}*. A equipe está em "
                    f"*{NOME[faixa_equipe]}*: clique nela para voltar.")


# --- guardar as decisoes da equipe ---

def guardar_para_desfazer(quantas):
    pilha = desfazer.setdefault(id_base, [])
    pilha.append((bases[id_base], quantas))
    del pilha[:-20]


def desfazer_ultima():
    anterior, quantas = desfazer[id_base].pop()
    bases[id_base] = anterior
    decisoes[id_base] = decisoes.get(id_base, 0) - quantas
    st.toast("Última decisão desfeita.", icon=":material/undo:")


def salvar_uma(prefixo, titulo, descricao):
    pontos = st.session_state.get(f"{prefixo}_pontos")
    if pontos is None:
        return
    chave = (st.session_state.get(f"{prefixo}_chave") or "").strip() or None
    guardar_para_desfazer(1)
    bases[id_base] = pv.decidir(bases[id_base], titulo, descricao, pontos, chave)
    decisoes[id_base] = decisoes.get(id_base, 0) + 1
    st.session_state[f"{prefixo}_pontos"] = None
    st.session_state[f"{prefixo}_voto"] = None
    for campo in ("chave", "titulo", "descricao"):
        st.session_state[f"{prefixo}_{campo}"] = ""
    st.session_state.pop(f"{prefixo}_resultado", None)
    st.session_state.pop(f"{prefixo}_opiniao", None)
    st.toast(f"Salva no histórico com {num(pontos)} pontos.", icon=":material/check:")


def salvar_lote(itens):
    """itens: (chave ou None, titulo, pontos). Chave do backlog recebe os pontos."""
    guardar_para_desfazer(len(itens))
    base = bases[id_base]
    for chave, titulo, pontos in itens:
        base = pv.decidir(base, titulo, "", pontos, chave)
    bases[id_base] = base
    decisoes[id_base] = decisoes.get(id_base, 0) + len(itens)
    st.session_state.pop("dz_lote", None)
    st.toast(f"{len(itens)} {'decisão salva' if len(itens) == 1 else 'decisões salvas'} "
             "no histórico.", icon=":material/check:")


def bloco_decisao(prefixo, titulo, descricao):
    st.markdown("**Decisão da equipe**")
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    with c1:
        pontos = st.pills("Pontos decididos", CARTOES, key=f"{prefixo}_pontos",
                          format_func=num, label_visibility="collapsed")
    with c2:
        st.text_input("Chave no Jira (opcional)", key=f"{prefixo}_chave",
                      placeholder=f"{PREFIXO}-123")
    if pontos is not None and pontos >= GRANDE:
        st.info(f"**{num(pontos)} pontos costuma ser grande demais para uma sprint.** Vale "
                "conversar sobre quebrar a história em partes menores antes de salvar.",
                icon=":material/content_cut:")
    st.button("Salvar no histórico", type="primary", key=f"{prefixo}_salvar",
              disabled=pontos is None, on_click=salvar_uma,
              args=(prefixo, titulo, descricao), icon=":material/save:")


def coluna_decisao():
    return st.column_config.SelectboxColumn("Decisão da equipe",
                                            options=[SEM_DECISAO] + [num(c) for c in CARTOES],
                                            help="Os pontos que a equipe decidiu.")


def decididas(ed):
    """As linhas em que a equipe escolheu um cartao, com os pontos como numero."""
    f = ed[~ed["Decisão da equipe"].fillna(SEM_DECISAO).astype(str).isin([SEM_DECISAO, ""])].copy()
    f["Decisão da equipe"] = f["Decisão da equipe"].astype(str).str.replace(",", ".").astype(float)
    return f


# --- telas ---

def tela_inicial():
    st.info("Para começar, carregue o CSV exportado do Jira, ou escolha na barra ao lado "
            "uma **equipe nova** ou um exemplo.")
    with st.expander("Como o software decide o que sugerir"):
        st.markdown(
            f"- **Menos de {pv.MIN_MEDIANA} histórias estimadas: dia zero.** A equipe vota "
            "em 10 histórias prontas, ou escreve as suas, e elas viram as referências de "
            "cada cartão. Um LLM diz de qual referência a história nova está mais perto.\n"
            f"- **{pv.MIN_MEDIANA} ou mais: a mediana da equipe**, o valor do meio entre os "
            "pontos que ela já deu.\n"
            f"- **{pv.MIN_MODELO} ou mais: também o modelo de texto**, que lê o título e a "
            f"descrição. Ele aparece ao lado da mediana, sem vencedor. {POR_QUE_DUAS}\n"
            "- **Cada decisão salva entra no histórico**, e as sugestões passam a usá-la.")


def estimar_uma(com_modelo):
    with st.form("uma", border=False):
        titulo = st.text_input("Título do card", key="uma_titulo")
        descricao = st.text_area("Descrição", key="uma_descricao", height=140,
                                 placeholder=DICA_CARD)
        enviou = st.form_submit_button("Estimar", type="primary")
    if enviou:
        if not (titulo.strip() or descricao.strip()):
            st.warning("Escreva o título ou a descrição do card.")
            st.session_state.pop("uma_resultado", None)
        else:
            h = pv.historia(titulo, descricao)
            st.session_state["uma_voto"] = None
            st.session_state["uma_pontos"] = None
            st.session_state["uma_resultado"] = {
                "s": pv.sugerir(estimadas, h, com_modelo=com_modelo),
                "parecidas": pv.parecidas(estimadas, h.texto.iloc[0]),
                "titulo": titulo, "descricao": descricao,
                "revelado": False, "voto": None}
    r = st.session_state.get("uma_resultado")
    if not r:
        return

    # o voto vem antes da sugestao, para ela nao ancorar a equipe
    if not r["revelado"]:
        st.info(VOTO_ANTES, icon=":material/visibility_off:")
        voto = st.pills("Voto da equipe", CARTOES, key="uma_voto", format_func=num)
        c1, c2 = st.columns([1, 3], vertical_alignment="center")
        c1.button("Ver a sugestão", type="primary", disabled=voto is None,
                  on_click=revelar, key="uma_revelar", icon=":material/visibility:")
        c2.button("Ver sem votar", type="tertiary", on_click=revelar, key="uma_sem_voto")
        return

    s, tem_modelo = r["s"], "modelo" in r["s"]
    with st.container(border=True):
        cols = st.columns(2)
        if tem_modelo:
            cols[0].metric("Modelo de texto", num(s.modelo.iloc[0]), help=PERTO)
            cols[0].caption(f"Calculou {num(round(s.modelo_valor.iloc[0], 1))}. Fica quase "
                            "sempre perto da mediana.")
        cols[1 if tem_modelo else 0].metric("Mediana da equipe", num(s.mediana.iloc[0]),
                                            help="A mesma para todas as histórias.")
        st.caption(POR_QUE_DUAS if tem_modelo else
                   f"O modelo de texto aparece com {pv.MIN_MODELO} histórias estimadas.")
        if r["voto"] is not None:
            st.markdown(f"**Voto da equipe: {num(r['voto'])}.** Se a sugestão for diferente, "
                        "vale ouvir por que a equipe votou assim antes de decidir.")

    p = r["parecidas"]
    if len(p):
        st.markdown("**Parecidas no histórico**, para comparar na conversa")
        st.dataframe(com_pontos(pd.DataFrame({"Chave": p.issuekey, "Título": p.title,
                                              "Pontos": p.storypoint}), ["Pontos"]),
                     hide_index=True, width="stretch")
        st.caption("Não é sugestão: copiar o ponto da mais parecida errou mais que a "
                   "mediana nos testes.")
    bloco_decisao("uma", r["titulo"], r["descricao"])


def revelar():
    """Mostra a sugestao. O voto, se houver, ja entra como a decisao da equipe."""
    r = st.session_state["uma_resultado"]
    r["voto"] = st.session_state.get("uma_voto")
    r["revelado"] = True
    st.session_state["uma_pontos"] = r["voto"]


def aba_backlog(com_modelo):
    if backlog.empty:
        st.info("Nenhuma história sem pontos neste arquivo. As que vierem do Jira sem story "
                "point aparecem aqui, com sugestões, para a equipe decidir em lote.")
        return
    s = sugerir(estimadas, backlog, com_modelo)
    tem_modelo = "modelo" in s
    mediana = f"Mediana ({num(s.mediana.iloc[0])} para todas)"
    sug = pd.DataFrame({"Chave": s.issuekey.values, "Título": s.title.values,
                        mediana: s.mediana.values})
    if tem_modelo:
        sug["Modelo de texto"] = s.modelo.values
    # a tabela de decisao nao mostra as sugestoes, para elas nao ancorarem o voto
    t = sug[["Chave", "Título"]].assign(**{"Decisão da equipe": SEM_DECISAO})
    st.markdown(f"**{len(backlog)}** histórias sem pontos. Preencha a decisão das que a "
                "equipe já discutiu e salve.")
    ed = st.data_editor(t, key=f"backlog::{id_base}::{len(backlog)}", hide_index=True,
                        width="stretch", disabled=["Chave", "Título"],
                        column_config={"Decisão da equipe": coluna_decisao()})
    with st.expander("Ver as sugestões (depois que a equipe votar)"):
        st.caption(VOTO_ANTES)
        st.dataframe(com_pontos(sug.assign(**{"Decisão da equipe": ed["Decisão da equipe"].values}),
                                [c for c in sug.columns if c in (mediana, "Modelo de texto")]),
                     hide_index=True, width="stretch")
    ed = sug.assign(**{"Decisão da equipe": ed["Decisão da equipe"].values})
    feitas = decididas(ed)
    c1, c2 = st.columns(2)
    c1.button(f"Salvar {len(feitas)} {'decisão' if len(feitas) == 1 else 'decisões'} no "
              "histórico", type="primary", disabled=feitas.empty, on_click=salvar_lote,
              args=(list(zip(feitas.Chave, feitas["Título"], feitas["Decisão da equipe"])),),
              width="stretch", icon=":material/save:")
    c2.download_button("Baixar o backlog com as sugestões (CSV)",
                       ed.to_csv(index=False, float_format="%g").encode(),
                       "backlog_sugestoes.csv", "text/csv",
                       width="stretch", icon=":material/download:")
    if tem_modelo:
        st.caption(POR_QUE_DUAS)


def aba_teste(com_modelo):
    if n < MIN_TESTE:
        st.info(f"O teste precisa de pelo menos {MIN_TESTE} histórias estimadas; a equipe "
                f"tem {n}.")
        return
    hist, teste = pv.demonstracao(estimadas)
    s = sugerir(hist, teste, com_modelo)
    tem_modelo = "modelo" in s
    st.markdown(f"Como as sugestões teriam se saído nas **{len(teste)} histórias mais "
                f"recentes** da equipe, usando só as {len(hist)} anteriores a elas "
                f"(mediana {num(s.mediana.iloc[0])}).")
    if len(teste) < 20:
        st.caption(f"Com só {len(teste)} histórias no teste, estes números mudam muito a "
                   "cada história nova. Ficam mais firmes com umas 100 estimadas.")
    comp = pv.comparar(s, teste.storypoint.values)
    for col, (nome, m) in zip(st.columns(2), comp.iterrows()):
        with col.container(border=True):
            st.markdown(f"**{nome}**")
            a, b, c = st.columns(3)
            a.metric("Erro médio", f"{m.MAE:.2f}".replace(".", ","),
                     help="Em pontos, por história. Menor é melhor.")
            b.metric("Até 1 ponto", f"{m.dentro_1:.0f}%",
                     help="Quantas vezes errou por no máximo 1 ponto. Maior é melhor.")
            c.metric("Cartão exato", f"{m.cartao_exato:.0f}%",
                     help="Quantas vezes acertou o cartão. Maior é melhor.")
    if tem_modelo:
        st.caption(f"Estes números são desta equipe. {POR_QUE_DUAS}")
    with st.expander(f"Ver história por história ({len(teste)})"):
        t = pd.DataFrame({"Chave": s.issuekey.values, "Título": s.title.values,
                          "Mediana das anteriores": s.mediana.values})
        if tem_modelo:
            t["Modelo de texto"] = s.modelo.values
        t["Valor real"] = teste.storypoint.values
        estilo = com_pontos(t, [c for c in t.columns if c not in ("Chave", "Título")]) \
            .set_properties(subset=["Valor real"], **{"background-color": "#F4F5F7"})
        st.dataframe(estilo, hide_index=True, width="stretch")


def aba_historico():
    if estimadas.empty:
        st.info("Nenhuma história estimada ainda.")
        return
    contagem = estimadas.storypoint.value_counts().sort_index()
    st.markdown("**Pontos já dados:** " + " · ".join(
        f"{num(v)} pts: {c}" for v, c in contagem.items()))
    recentes = estimadas.iloc[::-1]
    st.dataframe(com_pontos(pd.DataFrame({"Chave": recentes.issuekey.values,
                                          "Título": recentes.title.values,
                                          "Pontos": recentes.storypoint.values}), ["Pontos"]),
                 hide_index=True, width="stretch")
    st.caption("As mais recentes primeiro. Errou ao salvar? Use **Desfazer** na barra "
               "lateral.")


def tela_historico(com_modelo):
    abas = st.tabs(["Estimar", "Backlog", "Histórico", "Teste no histórico"])
    with abas[0]:
        estimar_uma(com_modelo)
    with abas[1]:
        aba_backlog(com_modelo)
    with abas[2]:
        aba_historico()
    with abas[3]:
        aba_teste(com_modelo)


def preencher():
    hist, teste = pv.demonstracao(estimadas)
    refs = pv.referencias_do_historico(hist)
    for v in CARTOES_REF:
        st.session_state[f"ref_{v}"] = refs.get(v, "")
    linhas = teste.texto.str.replace(r"\s+", " ", regex=True).head(MAX_DIA_ZERO)
    st.session_state["novas_dz"] = "\n".join(linhas)
    st.session_state["forma_dz"] = "varias"
    st.session_state["forma_ref"] = "nossas"
    st.session_state["reais_dz"] = dict(zip(linhas, teste.storypoint.head(MAX_DIA_ZERO)))
    st.session_state.pop("dz_lote", None)


def referencias_dia_zero(forma):
    """{cartao: texto da referencia} a partir do que esta na sessao, sem desenhar nada."""
    if forma == "padrao":
        return pv.referencias_dos_votos(votos_padrao())
    return {v: st.session_state[f"ref_{v}"] for v in CARTOES_REF
            if (st.session_state.get(f"ref_{v}") or "").strip()}


def votos_padrao():
    votos = {h["id"]: st.session_state.get(f"pad_{h['id']}") for h in pv.historias_padrao()}
    return {k: v for k, v in votos.items() if v is not None}


def votar_padrao():
    st.caption("Votem como numa planning, pensando no sistema de vocês: quantos pontos a "
               "equipe daria a cada história? Não precisa votar em todas: bastam 2 cartões "
               "diferentes. Cada cartão votado vira a referência daquele tamanho. O dia "
               "zero vai até 13: se a equipe acha que uma história passa disso, deixe sem "
               "voto.")
    for h in pv.historias_padrao():
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        with c1.expander(h["pt"]["titulo"]):
            st.markdown(h["pt"]["descricao"].replace("\n", "  \n"))
        c2.pills(h["pt"]["titulo"], CARTOES_REF, key=f"pad_{h['id']}", format_func=num,
                 label_visibility="collapsed")
    votos, titulo = votos_padrao(), {h["id"]: h["pt"]["titulo"] for h in pv.historias_padrao()}
    usadas, repetidas = {}, set()
    for i, v in votos.items():           # a mesma regra de pv.referencias_dos_votos
        if v in CARTOES_REF:
            repetidas |= {v} if v in usadas else set()
            usadas.setdefault(v, i)
    if usadas:
        st.markdown("**Referências:** " + " · ".join(
            f"{num(v)}: {titulo[usadas[v]]}" for v in sorted(usadas)))
    if repetidas:
        st.caption(f"Mais de uma história no cartão {', '.join(map(num, sorted(repetidas)))}: "
                   "vale a primeira da lista.")
    return referencias_dia_zero("padrao")


def escrever_nossas():
    st.caption("Uma história típica de cada tamanho para a equipe. Nenhum campo é "
               "obrigatório: bastam 2 preenchidos, e o cartão vazio fica de fora.")
    colunas = st.columns(3)
    for k, v in enumerate(CARTOES_REF):
        colunas[k % 3].text_area(
            f"{v} ponto{'s' if v > 1 else ''} (opcional)", key=f"ref_{v}", height=100,
            placeholder=f"Uma história típica de {v} ponto{'s' if v > 1 else ''}")
    return referencias_dia_zero("nossas")


def tela_dia_zero():
    simulando = d is not None and n >= pv.MIN_MEDIANA
    if simulando:
        st.button("Simular com o histórico carregado", on_click=preencher,
                  icon=":material/science:",
                  help="Referências: a história mais típica de cada cartão nas 80% mais "
                       f"antigas. Novas: {MAX_DIA_ZERO} das 20% mais recentes, com o valor real.")
    elif d is not None:
        st.markdown(f"Ainda não há mediana: ela precisa de {pv.MIN_MEDIANA} histórias "
                    "estimadas. Cada decisão salva aqui aproxima a equipe dela.")

    # o Streamlit apaga o valor de um campo que sai da tela: regravar os votos e as
    # referencias escritas faz os dois sobreviverem a troca entre as duas formas
    for k in [f"pad_{h['id']}" for h in pv.historias_padrao()] + [f"ref_{v}" for v in CARTOES_REF]:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]
    forma_ref = st.session_state.setdefault("forma_ref", "padrao")
    refs = referencias_dia_zero(forma_ref)
    with st.expander(f"1. Histórias de referência ({len(refs)} de {len(CARTOES_REF)} "
                     "cartões com referência)", expanded=len(refs) < 2):
        st.radio("Como montar as referências", list(FORMAS_REF), format_func=FORMAS_REF.get,
                 key="forma_ref", horizontal=True, label_visibility="collapsed")
        if forma_ref == "padrao":
            refs = votar_padrao()
        else:
            refs = escrever_nossas()
        preenchidas = sorted(refs)
        if 2 <= len(preenchidas) < len(CARTOES_REF):
            st.caption(f"O LLM só escolhe entre os cartões com referência "
                       f"({', '.join(map(num, preenchidas))}). Ele vê as referências em "
                       "ordem, da menor para a maior, sem os números.")

    st.subheader("2. História nova")
    forma = st.radio("Como informar", ["uma", "varias"], horizontal=True, key="forma_dz",
                     label_visibility="collapsed",
                     format_func={"uma": "Uma história",
                                  "varias": f"Várias, uma por linha (até {MAX_DIA_ZERO})"}.get)
    if forma == "uma":
        titulo = st.text_input("Título do card", key="dz_titulo")
        descricao = st.text_area("Descrição", key="dz_descricao", height=120,
                                 placeholder=DICA_CARD)
        texto = f"{titulo} {descricao}".strip()
        linhas = [texto] if texto else []
    else:
        texto_novas = st.text_area("Uma por linha", key="novas_dz", height=150)
        linhas = [t.strip() for t in texto_novas.splitlines() if t.strip()][:MAX_DIA_ZERO]

    provedor = st.radio("Quem compara", list(LLM), index=None, format_func=LLM.get,
                        horizontal=True, key="provedor_dz")
    if provedor == "gemini":
        st.warning("**O texto sai deste computador** e vai para a API do Google. Não use "
                   "com histórias que não podem sair da empresa.")
    falta = ("Monte pelo menos 2 referências, de cartões diferentes." if len(preenchidas) < 2 else
             "Escolha quem compara." if provedor is None else
             "Escreva a história nova." if not linhas else None)
    clicou = st.button("Enviar ao Gemini e sugerir" if provedor == "gemini" else "Sugerir",
                       type="primary", disabled=falta is not None)
    if falta:
        st.caption(falta)
    if clicou:
        try:
            with st.spinner("Comparando com as referências..."):
                sugestoes = pv.dia_zero(refs, linhas, provedor)
        except (Exception, SystemExit) as e:  # chave ausente sai com SystemExit
            st.error(f"Não consegui consultar o LLM: {e}")
        else:
            if forma == "uma":
                st.session_state["dz_resultado"] = {"v": sugestoes[0], "provedor": provedor,
                                                    "titulo": titulo, "descricao": descricao,
                                                    "ref": refs.get(sugestoes[0])}
            else:
                st.session_state["dz_lote"] = {"linhas": linhas, "sugestoes": sugestoes,
                                               "provedor": provedor}

    quem = {"gemini": "Comparada pelo Gemini: o texto foi enviado ao Google.",
            "ollama": "Comparada pelo Qwen local: nada saiu deste computador."}
    if forma == "uma" and st.session_state.get("dz_resultado"):
        r = st.session_state["dz_resultado"]
        with st.container(border=True):
            st.metric("Sugestão do dia zero",
                      num(r["v"]) if r["v"] is not None else "sem resposta",
                      help="O cartão da referência que o LLM achou mais parecida.")
            if r.get("ref"):
                st.markdown(f"Mais parecida com a referência de **{num(r['v'])} pontos**: "
                            f"*{curto(r['ref'], 160)}*")
            st.caption(f"{quem[r['provedor']]} {DIA_ZERO_NOS_TESTES}")
        bloco_decisao("dz", r["titulo"], r["descricao"])
    elif forma == "varias" and st.session_state.get("dz_lote"):
        r = st.session_state["dz_lote"]
        t = pd.DataFrame({"História": [curto(x) for x in r["linhas"]],
                          "Sugestão": [num(v) if v is not None else "sem resposta"
                                       for v in r["sugestoes"]]})
        reais = st.session_state.get("reais_dz") or {}
        simulado = bool(reais) and all(x in reais for x in r["linhas"])
        st.caption(f"{quem[r['provedor']]} {DIA_ZERO_NOS_TESTES}")
        if simulado:   # historias do proprio historico: so comparar com o valor real
            t["Valor real"] = [num(reais[x]) for x in r["linhas"]]
            st.dataframe(t, hide_index=True, width="stretch")
            return
        t["Decisão da equipe"] = SEM_DECISAO
        ed = st.data_editor(t, key="dz_editor", hide_index=True, width="stretch",
                            disabled=["História", "Sugestão"],
                            column_config={"Decisão da equipe": coluna_decisao()})
        f = decididas(ed)
        feitas = [(None, r["linhas"][i], p) for i, p in zip(f.index, f["Decisão da equipe"])]
        st.button(f"Salvar {len(feitas)} {'decisão' if len(feitas) == 1 else 'decisões'} no "
                  "histórico", type="primary", disabled=not feitas, on_click=salvar_lote,
                  args=(feitas,), icon=":material/save:")


with st.sidebar:
    if d is not None and len(d):
        k = decisoes.get(id_base, 0)
        st.download_button("Baixar histórico", pv.para_csv(d), "historico_estimativas.csv",
                           "text/csv", icon=":material/download:", width="stretch",
                           type="primary" if k else "secondary")
        if k:
            st.caption(f"**{k} {'decisão' if k == 1 else 'decisões'} nesta sessão.** "
                       "Baixe o histórico para não perder.")
        if desfazer.get(id_base):
            st.button("Desfazer", on_click=desfazer_ultima, icon=":material/undo:",
                      width="stretch", help="Desfaz a última decisão salva.")
    st.divider()
    st.caption("**Privacidade.** O modelo de texto e o Qwen local rodam neste computador. "
               "Só o Gemini, no dia zero, envia texto ao Google.")

if d is None and faixa is None:
    tela_inicial()
elif faixa == "dia zero":
    tela_dia_zero()
elif d is None:
    st.info("Carregue um CSV exportado do Jira ou escolha uma equipe nova ou um exemplo.")
elif not acima:
    tela_historico(com_modelo=faixa == "mediana e modelo")
