"""As 10 historias padrao do dia zero e a regra que transforma votos em referencias.

A pagina (software/previsao.py) e o teste (dia_zero_padrao.py) usam este mesmo
modulo, para que a pagina faca exatamente o que foi testado (PLANO.md, "Dia zero
com historias padrao").

Regra: cada voto de 1 a 13 vira a referencia daquele cartao. Se duas historias
recebem o mesmo cartao, o teste mede duas formas: vale a primeira da lista, ou as
duas entram juntas na mesma referencia. Votos de 20 e 40 nao viram referencia,
como no dia zero testado, que vai ate 13 (a pagina nem oferece esses cartoes).

Uso (gera o prompt que o Lucas cola no app do Gemini):
    python src/historias_padrao.py > base_sintetica/voto_padrao.txt
"""
import json
from pathlib import Path

from cold_start_comum import FIBONACCI

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO = RAIZ / "software" / "historias_padrao.json"
TIPO = {"PAD-02": "bug", "PAD-10": "bug", "PAD-07": "task"}   # o resto e "story"


def historias_padrao():
    """[{id, pt: {titulo, descricao}, en: {title, description}}, ...], na ordem da lista."""
    return json.loads(ARQUIVO.read_text(encoding="utf-8"))["historias"]


def texto(h, lingua="pt"):
    """Titulo + descricao, como o texto de uma historia do CSV."""
    if lingua == "pt":
        return f"{h['pt']['titulo']} {h['pt']['descricao']}".strip()
    return f"{h['en']['title']} {h['en']['description']}".strip()


def referencias_dos_votos(votos, lingua="pt", todas=False):
    """{cartao: texto da referencia} a partir de {id da historia: pontos votados}.

    Historias sem voto ficam de fora. Cartoes fora de 1 a 13 nao viram referencia.
    Dois votos no mesmo cartao: todas=False usa a historia que vem primeiro na
    lista; todas=True junta as historias do cartao numa referencia so ("Story 1:
    ... Story 2: ..."), para o prompt de referencias agrupadas.
    """
    grupos = {}
    for h in historias_padrao():
        v = votos.get(h["id"])
        if v is None:
            continue
        v = float(v)
        v = int(v) if v.is_integer() else v
        if v in FIBONACCI:
            grupos.setdefault(v, []).append(texto(h, lingua))
    if not todas:
        return {v: g[0] for v, g in grupos.items()}
    return {v: g[0] if len(g) == 1 else
            "\n\n".join(f"Story {i + 1}: {t}" for i, t in enumerate(g))
            for v, g in grupos.items()}


def tem_cartao_repetido(votos):
    vs = [float(v) for v in votos.values() if v is not None and float(v) in FIBONACCI]
    return len(vs) != len(set(vs))


def ler_votos(arquivo):
    """Um arquivo com uma linha JSON por historia: {"id": "PAD-01", "story_point": 5}."""
    votos = {}
    for linha in Path(arquivo).read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if linha.startswith("{"):
            j = json.loads(linha)
            votos[str(j["id"]).strip().upper()] = j["story_point"]
    return votos


def prompt_do_voto():
    """O prompt de estimativa da base construida, com as 10 historias como unico bloco.

    Copiado do bloco 1 (base_sintetica/blocos/bloco_01.txt), com duas trocas: um
    bloco de 10 em vez de blocos de 25, e o id do exemplo (PAD-01 em vez de HIST-001).
    """
    original = (RAIZ / "base_sintetica" / "blocos" / "bloco_01.txt").read_text(encoding="utf-8")
    cabecalho = original[:original.index("BLOCK 1")]
    trocas = [("I will send the items in blocks of 25.", "I will send the items in one block of 10."),
              ('{"id": "HIST-001", "story_point": 3}', '{"id": "PAD-01", "story_point": 3}')]
    for de, para in trocas:
        assert de in cabecalho, de
        cabecalho = cabecalho.replace(de, para)
    linhas = [json.dumps({"id": h["id"], "type": TIPO.get(h["id"], "story"),
                          "title": h["en"]["title"], "description": h["en"]["description"]},
                         ensure_ascii=False, separators=(",", ":"))
              for h in historias_padrao()]
    return cabecalho + "BLOCK 1\n" + "\n".join(linhas) + "\n"


if __name__ == "__main__":
    print(prompt_do_voto(), end="")
