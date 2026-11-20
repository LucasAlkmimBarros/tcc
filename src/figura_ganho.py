"""Figura do resultado principal: o ganho do modelo sobre a mediana, projeto a projeto.

Uma barra por projeto, nos 39 projetos (16 do Deep-SE e 23 do TAWOS), e uma para a
base construida (RoomBook), com o intervalo de 95% do ganho reamostrando as 50
historias de teste (sensibilidade_base_construida.csv, corte oficial): o erro medio
da mediana da equipe menos o erro medio do modelo de texto do software (TF-IDF +
SVR), no mesmo teste (20% mais recentes). A mediana e a da literatura: a dos 60%
mais antigos (treino), como no Deep-SE e na replicacao (resultados_completos.csv,
"Mediana historica"); o modelo treina com os 80% (treino e validacao), e o erro
dele vem de sonda_texto.csv. Na base construida, a mediana e 5 com 150 ou com 200
historias, entao o erro dela no teste e o mesmo. Barra para a direita: o modelo
erra menos. Para a esquerda: a mediana erra menos.

Uso:  python src/figura_ganho.py
      (grava resultados/figura_ganho_por_projeto.pdf, .png e .csv)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

from apontar_grandes import so_no_tawos
from conjunto import RAIZ

RES = RAIZ / "resultados"
# mesmas cores da figura anterior: azul para o modelo, laranja para a mediana
# (par validado com o validador de paleta; o lado da barra ja diz o sinal)
COR_MODELO, COR_MEDIANA = "#2a78d6", "#eb6834"
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e8e7e2"
virgula = lambda v, casas=2: f"{v:+.{casas}f}".replace(".", ",").replace("-", "−")

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.edgecolor": "#c9c8c3",
    "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "figure.dpi": 200,
})


def ganhos(arquivo, completos=None):
    s = pd.read_csv(arquivo).pivot(index="projeto", columns="condicao", values="MAE")
    mediana = s["mediana (nenhum texto)"]
    if completos is not None:  # bases publicas: a mediana dos 60% de treino
        r = pd.read_csv(completos)
        mediana = (r[r.modelo == "Mediana historica"].set_index("projeto").MAE
                   .reindex(s.index))
        assert mediana.notna().all(), "projeto sem mediana historica"
    return pd.DataFrame({"erro_mediana": mediana,
                         "erro_modelo": s["texto completo"],
                         "ganho": mediana - s["texto completo"]})


def painel(ax, g, titulo, repetidos=()):
    g = g.sort_values("ganho")
    nomes = [f"{p}*" if p in repetidos else p for p in g.index]
    cores = [COR_MODELO if v > 0 else COR_MEDIANA for v in g.ganho]
    ax.barh(range(len(g)), g.ganho, height=0.62, color=cores, zorder=3)
    ax.axvline(0, color=TINTA, lw=0.8, zorder=4)
    ax.set_yticks(range(len(g)), nomes, fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(g) - 0.3)
    ax.xaxis.grid(True, color=GRADE, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    # rotulo so nos dois extremos: o resto se le no eixo e no CSV
    for i in (0, len(g) - 1):
        v = g.ganho.iloc[i]
        ax.text(v + (0.03 if v > 0 else -0.03), i, virgula(v), va="center",
                ha="left" if v > 0 else "right", fontsize=7.5, color=TINTA)
    frente = int((g.ganho > 0).sum())
    ax.set_title(f"{titulo} ({len(g)} projetos)\nmodelo à frente em {frente}; "
                 f"média {virgula(g.ganho.mean())}",
                 fontsize=8.5, color=TINTA, loc="left", pad=6, linespacing=1.4)


def painel_construida(ax, g, ic):
    v = g.ganho.iloc[0]
    ax.barh([0], [v], height=0.62, color=COR_MODELO if v > 0 else COR_MEDIANA, zorder=3)
    ax.errorbar([v], [0], xerr=[[v - ic[0]], [ic[1] - v]], fmt="none", ecolor=TINTA,
                elinewidth=0.9, capsize=3, zorder=5)
    ax.axvline(0, color=TINTA, lw=0.8, zorder=4)
    ax.set_yticks([0], ["RoomBook"], fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, 0.7)
    ax.xaxis.grid(True, color=GRADE, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.text(ic[1] + 0.03, 0, virgula(v),
            va="center", ha="left", fontsize=7.5, color=TINTA)
    ax.set_title(f"(c) Base construída (1 projeto)\nintervalo: {virgula(ic[0])} a {virgula(ic[1])}", fontsize=8.5, color=TINTA, loc="left", pad=6, linespacing=1.4)


def main():
    ds = ganhos(RES / "sonda_texto.csv", RES / "resultados_completos.csv")
    tw = ganhos(RES / "tawos" / "sonda_texto.csv", RES / "tawos" / "resultados_completos.csv")
    rb = ganhos(RES / "sintetica" / "sonda_texto.csv")
    of = pd.read_csv(RES / "sensibilidade_base_construida.csv").query("corte_oficial").iloc[0]
    ic = (of.ic95_inf, of.ic95_sup)
    repetidos = set(tw.index) - so_no_tawos()

    fig = plt.figure(figsize=(6.4, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[17, 3.2], wspace=0.7, hspace=0.75)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.92, bottom=0.2)
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[:, 1], sharex=a)
    c = fig.add_subplot(gs[1, 0], sharex=a)
    painel(a, ds, "(a) Deep-SE")
    painel(b, tw, "(b) TAWOS", repetidos)
    painel_construida(c, rb, ic)
    limite = max(abs(ds.ganho).max(), abs(tw.ganho).max(), ic[1]) + 0.27
    for ax in (a, b, c):
        ax.set_xlim(-limite, limite)
        ax.xaxis.set_major_locator(MultipleLocator(0.4))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else virgula(v, 1)))
    fig.supxlabel("Erro médio da mediana da equipe − erro médio do modelo de texto "
                  "(story points por história)", fontsize=8.5, color=TINTA2, y=0.105)
    fig.legend(handles=[Patch(color=COR_MEDIANA, label="a mediana da equipe erra menos (barra à esquerda)"),
                        Patch(color=COR_MODELO, label="o modelo de texto erra menos (barra à direita)")],
               loc="lower center", ncol=2, frameon=False, fontsize=7.8, labelcolor=TINTA2,
               bbox_to_anchor=(0.5, 0.035), handlelength=1.2, columnspacing=1.6)
    fig.text(0.5, 0.012, "* projeto que também está no Deep-SE", ha="center",
             fontsize=7.5, color=TINTA2)

    for f in ["figura_ganho_por_projeto.pdf", "figura_ganho_por_projeto.png"]:
        fig.savefig(RES / f, bbox_inches="tight", facecolor="white")
    tabela = pd.concat([ds.assign(conjunto="Deep-SE"), tw.assign(conjunto="TAWOS"),
                        rb.assign(conjunto="base construida")]).reset_index()
    tabela["tambem_no_deepse"] = (tabela.conjunto == "TAWOS") & tabela.projeto.isin(repetidos)
    tabela[["conjunto", "projeto", "erro_mediana", "erro_modelo", "ganho", "tambem_no_deepse"]] \
        .to_csv(RES / "figura_ganho_por_projeto.csv", index=False, float_format="%.4f")
    print("gerado:", RES / "figura_ganho_por_projeto.pdf")
    print(f"Deep-SE: a frente em {(ds.ganho > 0).sum()} de {len(ds)}, media {ds.ganho.mean():+.3f}")
    print(f"TAWOS:   a frente em {(tw.ganho > 0).sum()} de {len(tw)}, media {tw.ganho.mean():+.3f}; "
          f"repetidos: {len(repetidos)}")


if __name__ == "__main__":
    main()
