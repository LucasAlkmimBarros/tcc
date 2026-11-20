"""Cliente de LLM com cache em disco: toda chamada fica guardada e rodar de novo
nao paga de novo.

Por que existe. O experimento de cold start (fase 3 do PLANO.md) faz milhares de
chamadas a um LLM na faixa gratuita de uma API, que tem limite por minuto e por
dia. Tres exigencias saem disso:

1. Reprodutibilidade. A resposta de um LLM de nuvem pode mudar de um dia para o
   outro. Guardar cada resposta em disco, com o modelo exato que respondeu
   (`modelVersion`) e a data, transforma a execucao num registro fixo: quem
   rodar de novo le o cache e obtem os mesmos numeros.
2. Retomada. Quando a cota do dia acaba, o script para; no dia seguinte,
   recomeca de onde parou, porque tudo o que ja foi respondido esta no cache.
3. Rastreabilidade. A chave do cache e o hash de (provedor, modelo, versao do
   prompt, configuracao, texto do prompt). Mudar uma virgula no prompt gera
   outra chave, e a resposta antiga nunca e reaproveitada por engano.

Dois provedores:
    gemini   API do Google, faixa gratuita. A chave vem de GEMINI_API_KEY ou de
             ~/.config/tcc/gemini_key e nunca e gravada em arquivo nem em log.
    ollama   modelo local (piloto e depuracao), sem custo e sem limite.

Erros: 503 (modelo sobrecarregado) e 500 sao temporarios e tentados de novo
com espera crescente. 429 por minuto espera o tempo que a API pede. 429 por dia
levanta `CotaEsgotada`, para o chamador parar e retomar depois pelo cache.
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CACHE_LLM = RAIZ / "cache" / "llm"
PROMPTS = RAIZ / "prompts"

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")

# configuracao fixa de geracao: temperatura zero para a resposta ser a mais
# provavel, e "pensamento" baixo para caber na cota de tokens por minuto
CONFIG_GEMINI = {"temperature": 0, "thinkingConfig": {"thinkingLevel": "low"}}
CONFIG_OLLAMA = {"temperature": 0, "seed": 42}


class CotaEsgotada(Exception):
    """A cota diaria da faixa gratuita acabou; retomar depois pelo cache."""


def carregar_prompt(nome):
    """Le um modelo de prompt versionado em prompts/. A versao e o nome do arquivo."""
    return (PROMPTS / f"{nome}.txt").read_text()


def _chave_api():
    k = os.environ.get("GEMINI_API_KEY")
    if not k:
        arq = Path.home() / ".config" / "tcc" / "gemini_key"
        if arq.exists():
            k = arq.read_text().strip()
    if not k:
        raise SystemExit("defina GEMINI_API_KEY ou crie ~/.config/tcc/gemini_key")
    return k


def _arquivo_cache(provedor, modelo, versao, config, prompt):
    bruto = json.dumps({"provedor": provedor, "modelo": modelo, "versao": versao,
                        "config": config, "prompt": prompt},
                       sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(bruto.encode()).hexdigest()
    return CACHE_LLM / h[:2] / f"{h}.json"


def em_cache(prompt, modelo, versao, provedor="gemini"):
    config = CONFIG_GEMINI if provedor == "gemini" else CONFIG_OLLAMA
    return _arquivo_cache(provedor, modelo, versao, config, prompt).exists()


class _Ritmo:
    """Espacamento minimo entre chamadas, para nao estourar o limite por minuto."""
    intervalo = float(os.environ.get("LLM_INTERVALO", "4.5"))
    ultima = 0.0

    @classmethod
    def esperar(cls):
        falta = cls.intervalo - (time.monotonic() - cls.ultima)
        if falta > 0:
            time.sleep(falta)
        cls.ultima = time.monotonic()


def _postar(url, corpo, cabecalhos, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(corpo).encode(),
                                 headers={"Content-Type": "application/json", **cabecalhos})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _espera_pedida(erro):
    """Segundos que a API pede para esperar num 429 (campo retryDelay)."""
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', erro)
    return float(m.group(1)) if m else 60.0


def _chamar_gemini(prompt, modelo):
    corpo = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": CONFIG_GEMINI}
    cab = {"x-goog-api-key": _chave_api()}
    espera = 5.0
    for tentativa in range(12):
        _Ritmo.esperar()
        try:
            d = _postar(GEMINI_URL.format(modelo), corpo, cab)
        except urllib.error.HTTPError as e:
            erro = e.read().decode(errors="replace")
            if e.code == 429:
                # o erro lista as cotas estouradas (quotaId); a diaria nao volta logo
                CACHE_LLM.mkdir(parents=True, exist_ok=True)
                (CACHE_LLM / "ultimo_429.json").write_text(erro)
                cotas = re.findall(r'"quotaId":\s*"([^"]+)"', erro)
                if any("PerDay" in c for c in cotas):
                    raise CotaEsgotada(", ".join(cotas))
                s = _espera_pedida(erro)
                print(f"    (429 {', '.join(cotas) or 'sem quotaId'}: esperando {s:.0f}s)",
                      flush=True)
                time.sleep(s + 1)
                continue
            if e.code in (500, 502, 503, 504):
                print(f"    ({e.code}: esperando {espera:.0f}s)", flush=True)
                time.sleep(espera)
                espera = min(espera * 2, 60)
                continue
            raise RuntimeError(f"Gemini {e.code}: {erro[:800]}")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"    (rede: {e}; esperando {espera:.0f}s)", flush=True)
            time.sleep(espera)
            espera = min(espera * 2, 60)
            continue
        cand = (d.get("candidates") or [{}])[0]
        partes = cand.get("content", {}).get("parts", [])
        texto = "".join(p.get("text", "") for p in partes if not p.get("thought"))
        return {"texto": texto, "modelo_respondeu": d.get("modelVersion"),
                "uso": d.get("usageMetadata"), "fim": cand.get("finishReason")}
    raise RuntimeError("Gemini: tentativas esgotadas")


# contexto pedido ao Ollama: o padrao corta prompts longos (os em lote passam de
# 8 mil tokens) sem avisar. Nao entra na chave do cache: nao muda a resposta de um
# prompt que ja cabia no contexto padrao, como todos os unitarios.
CONTEXTO_OLLAMA = 16384


def _chamar_ollama(prompt, modelo):
    d = _postar(OLLAMA_URL, {"model": modelo, "prompt": prompt, "stream": False,
                             "options": {**CONFIG_OLLAMA, "num_ctx": CONTEXTO_OLLAMA}},
                {}, timeout=1800)
    return {"texto": d.get("response", ""), "modelo_respondeu": d.get("model"),
            "uso": {"prompt": d.get("prompt_eval_count"), "saida": d.get("eval_count"),
                    "num_ctx": CONTEXTO_OLLAMA},
            "fim": d.get("done_reason")}


def perguntar(prompt, modelo, versao, provedor="gemini"):
    """Devolve o registro da resposta (texto, modelo que respondeu, uso, data).

    Se a mesma pergunta ja foi feita, le do disco e nao chama a API.
    """
    config = CONFIG_GEMINI if provedor == "gemini" else CONFIG_OLLAMA
    arq = _arquivo_cache(provedor, modelo, versao, config, prompt)
    if arq.exists():
        return json.loads(arq.read_text())
    chamar = _chamar_gemini if provedor == "gemini" else _chamar_ollama
    r = chamar(prompt, modelo)
    registro = {"provedor": provedor, "modelo": modelo, "versao_prompt": versao,
                "config": config, "prompt": prompt, **r,
                "data": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    arq.parent.mkdir(parents=True, exist_ok=True)
    tmp = arq.with_suffix(".tmp")
    tmp.write_text(json.dumps(registro, ensure_ascii=False, indent=1))
    tmp.replace(arq)
    return registro


def extrair_json(texto):
    """Le o objeto JSON da resposta (o modelo as vezes poe ```json em volta).

    Tenta primeiro do primeiro "{" ao ultimo "}" (respostas aninhadas, em lote);
    se nao for JSON valido, o primeiro objeto simples, como antes.
    """
    t = texto or ""
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(t[i:j + 1])
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*?\}", t, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
