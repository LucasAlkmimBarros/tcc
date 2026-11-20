# Quando o texto ajuda a estimar *story points*?

Código, dados e resultados do Trabalho de Conclusão de Curso de Lucas Alkmim Barros
(Ciência da Computação, PUC Minas), orientado por Humberto Torres Marques Neto.

O trabalho compara um modelo de texto (TF-IDF + SVR linear) com a mediana do
histórico da equipe em três bases: as duas públicas da área (Deep-SE e TAWOS) e uma
base construída, em que as histórias descrevem o trabalho. Também testa um "dia
zero" com LLMs, para equipes sem histórico, e reúne os resultados num protótipo.

## Estrutura

```
src/             experimentos e análises (um script por tabela ou teste)
software/        protótipo: página que sugere story points
base_sintetica/  base construída (RoomBook): histórias, pontos, votos e prompts
prompts/         prompts do dia zero e do teste de contaminação
referencia/      tabelas publicadas usadas na comparação
resultados/      saídas dos scripts: Deep-SE na raiz, TAWOS em tawos/, base construída em sintetica/
```

Os resultados já vêm prontos, inclusive as respostas dos LLMs: dá para conferir os
números do artigo sem refazer nenhuma chamada.

## Dados

- **Deep-SE** (16 projetos, 23.313 histórias; Choetkiertikul et al., IEEE TSE 2019).
  Baixe os CSVs de <https://github.com/morakotch/datasets> (pasta
  `storypoint/IEEE TSE2018/dataset`) para `dados/`.
- **TAWOS** (23 projetos, 29.899 histórias; Tawosi et al., MSR 2022). É distribuído
  como dump MySQL (<https://doi.org/10.5522/04/21308124>). O trabalho usa as
  histórias da replicação de Tawosi, Moussa e Sarro, cujas chaves estão em
  <https://github.com/SOLAR-group/AgileEffortEstimation> (`datasets/Tawosi_Dataset/*_deep-se.csv`).

  ```bash
  mkdir -p tawos && cd tawos
  curl -L -o TAWOS.sql.zip https://ndownloader.figshare.com/files/37806375 && unzip TAWOS.sql.zip
  docker run -d --name tawos-mysql -e MYSQL_ROOT_PASSWORD=tawos -p 127.0.0.1:3307:3306 \
    -v "$PWD":/dump:ro -v tawos-data:/var/lib/mysql mysql:8.0 --max_allowed_packet=1G
  docker exec tawos-mysql mysql -uroot -ptawos -e "create database tawos_cleaned"
  docker exec -i tawos-mysql sh -c 'mysql -uroot -ptawos tawos_cleaned < /dump/TAWOS.sql'
  cd ..
  # copie os *_deep-se.csv da replicação para tawos/replicacao/ e exporte:
  .venv/bin/python src/tawos.py            # banco -> tawos/dados/
  .venv/bin/python src/conferir_tawos.py   # confere com a Tabela 2 da replicação
  ```

- **Base construída** (1 projeto, 250 histórias): já está em `base_sintetica/`. O
  GPT-6 escreveu as histórias (`prompt_escrever.txt`) num chat temporário, em blocos
  de 25 (`blocos/` guarda as respostas), e o Gemini 3.1 Pro deu os pontos
  (`prompt_estimar.txt`) numa conversa separada, lendo só o texto. É uma base gerada
  por IA, feita para testar condições, e não dado de uma equipe real.

## Ambiente

Python 3.9:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Todos os passos usam semente fixa (42): duas execuções dão os mesmos arquivos.

## Reproduzir os resultados

A variável `CONJUNTO` escolhe a base dos scripts de uma base só: vazia (Deep-SE),
`tawos` ou `sintetica`. Primeiro, o experimento principal, as verificações e as
previsões por história, em cada base:

```bash
.venv/bin/python src/experimento.py     # modelo contra média, mediana e palpite aleatório
.venv/bin/python src/verificar.py       # sete verificações que falham alto se algo quebrar
.venv/bin/python src/previsoes.py       # previsões do modelo e da mediana, história a história
CONJUNTO=tawos .venv/bin/python src/experimento.py   # e assim por diante
```

Depois, cada parte do artigo:

| No artigo | Scripts (em `src/`) |
|---|---|
| Seção 3.2: as bases batem com as tabelas publicadas | `caracterizar.py`, `conferir_tawos.py`, `comparar_deepse.py`, `comparar_tawos.py` |
| Tabela 3: o modelo e as soluções publicadas contra a mediana | `llm_publicado.py` (baixa as previsões de Shetty et al.), `tabela_central.py` |
| Figura 3: o ganho por projeto | `figura_ganho.py` |
| Tabela 4 e Seção 4.2: outras medidas | `sensibilidade.py`, `perto_da_mediana.py` |
| Seção 4.3: as histórias grandes e o aviso | `apontar_grandes.py`, `aviso_grandes.py` |
| Tabela 5 e Seção 4.4: por que o modelo passa da mediana numa base e não nas outras | `sonda_texto.py`, `sonda_volume.py`, `sonda_consistencia.py`, `sonda_piores.py`, `sonda_vizinhos.py` |
| Base construída: o modelo contra a mediana | `base_sintetica.py` (com `CONJUNTO=sintetica`) |
| Tabela 6 e Seção 4.5: o dia zero | `cold_start_reguas.py`, `cold_start_llm.py`, `cold_start_llm_lote.py`, `cold_start_analise.py`, `contaminacao.py` (com `CONJUNTO=tawos`) |
| Seção 4.5: as histórias padrão | `dia_zero_padrao.py` (com `CONJUNTO=sintetica`; a regra fica em `historias_padrao.py`) |
| Protótipo: a partir de quantas histórias o modelo aparece | `limiar_modelo.py` |

`tabela_central.py`, `figura_ganho.py`, `sensibilidade.py`, `apontar_grandes.py` e
`aviso_grandes.py` leem as três bases de uma vez e rodam sem `CONJUNTO`.

**Chamadas de LLM.** Os scripts do dia zero guardam toda resposta em cache e só
chamam o LLM para o que ainda não foi respondido. Para refazer as chamadas: o
Gemini precisa da chave da API em `GEMINI_API_KEY` (ou em `~/.config/tcc/gemini_key`),
e o Qwen local precisa do Ollama rodando, com o modelo `qwen2.5:14b`.

## Protótipo

```bash
.venv/bin/streamlit run software/app.py
```

A página lê o CSV exportado do Jira (ou começa uma equipe do zero) e sugere story
points conforme o histórico da equipe:

| Histórias já estimadas | Sugestão |
|---|---|
| menos de 5 | **dia zero:** a equipe vota em 10 histórias prontas ou escreve uma por cartão, e um LLM compara a história nova com essas referências |
| de 5 a 24 | a **mediana** da equipe |
| 25 ou mais | o **modelo de texto**, ao lado da mediana |

A página pede primeiro o voto da equipe e só então mostra a sugestão, para que o
número não puxe a estimativa. No CSV, as colunas obrigatórias são o título
(`Summary` ou `title`) e os story points (`Story Points`, `Story point estimate` ou
`storypoint`); a chave (`Issue key`), a descrição (`Description`) e a data de
criação (`Created`) são opcionais. O menu também traz a base construída como
exemplo.

O modelo de texto e o Qwen rodam no próprio computador. O dia zero com o Gemini
envia o texto das histórias para a API do Google.

O software importa o modelo e o prompt dos scripts do experimento, e
`software/conferir.py` confere que ele reproduz, projeto a projeto, os erros do
modelo e da mediana nas duas bases públicas:

```bash
.venv/bin/python software/conferir.py
```
