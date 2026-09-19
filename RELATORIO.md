# Relatório Técnico - Laboratório Estatístico Interativo

**Disciplina:** Matemática e Estatística para Computação
**Integrante(s):** Sara Martins Oliveira de Sousa - RA 72650204
**Repositório:** *https://github.com/saritaa1906/laboratorio-estatistico-academico*
**Vídeo:** **

---

## 1. Dataset e justificativa

Usei o **Bank Marketing** do UCI Machine Learning Repository
([link original](https://archive.ics.uci.edu/dataset/222/bank%2Bmarket)),
com **45.211 registros** de campanhas de telemarketing de um banco português
oferecendo depósitos a prazo. A base tem 7 variáveis numéricas (`age`,
`balance`, `day`, `duration`, `campaign`, `pdays`, `previous`) e 9
categóricas (`job`, `marital`, `education`, `default`, `housing`, `loan`,
`contact`, `month`, `poutcome`), além do alvo binário `y` — atende com folga
o critério mínimo (1.000 registros, 4 numéricas, 2 categóricas).

Escolhi esse dataset porque ele levanta uma pergunta de negócio real e
intuitiva - 'o que faz uma pessoa aceitar guardar dinheiro no banco quando
ligam oferecendo?' - algo com o qual lido de perto na minha rotina de trabalho
na área. Além disso, o conjunto de dados possui variáveis com formatos bem
diferentes (idade é quase simétrica, saldo e duração da ligação são fortemente
assimétricos), o que torna os Módulos 3 e 4 (TCL e ajuste de distribuições) visualmente ricos.

## 2. Decisões de tratamento dos dados

- **Nulos:** o dataset oficial não tem valores ausentes (`Has Missing
  Values? No`, conforme a página do UCI), então nenhuma linha foi removida
  por essa razão.
- **Categorias "unknown":** colunas como `job`, `education`, `contact` e
  `poutcome` trazem o valor `"unknown"` como uma categoria válida (não um
  nulo) - decidimos mantê-la como categoria própria em vez de descartar
  linhas, já que descartar reduziria a amostra sem necessidade e
  `"unknown"` já é tratado como resposta legítima pelo levantamento
  original.
- **Tipagem:** colunas numéricas (`age`, `balance`, `duration`, `campaign`,
  `pdays`, `previous`, `day`) são convertidas para `float` antes de entrar
  em qualquer função de `minhastats.py` (`.dropna().astype(float).tolist()`),
  isolando a etapa de "carregar dado" (Pandas) da etapa de "calcular"
  (núcleo próprio).
- **`pdays = -1`:** é um valor sentinela do dataset original, significando
  "nunca contatado antes" - não é um outlier nem um erro de digitação, e por
  isso não foi filtrado; ele aparece naturalmente na cauda esquerda quando
  essa variável é analisada na aba de distribuições.
- **Amostragem para os gráficos:** com 45 mil linhas, alguns gráficos de
  dispersão amostram até 10-50 mil pontos (`df.sample(...)`) só por
  desempenho de renderização - isso não afeta nenhuma estatística exibida,
  que é sempre calculada sobre a coluna completa, não sobre a amostra do
  gráfico.

## 3. Núcleo estatístico - fórmulas implementadas

Todas as funções abaixo estão em `minhastats.py`, em Python puro (sem
NumPy/SciPy/`statistics` nas contas).

**Média:**
$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

**Mediana:** valor central da amostra ordenada; média dos dois centrais se
$n$ for par.

**Moda:** valor(es) de maior frequência (pode haver mais de um — a função
devolve uma lista).

**Amplitude:** $\max(x) - \min(x)$

**Variância amostral** (correção de Bessel) **e populacional:**
$$s^2 = \frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n-1} \qquad\qquad \sigma^2 = \frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n}$$

**Desvio-padrão:** $s = \sqrt{s^2}$

**Percentil $p$** (interpolação linear, mesma convenção do `numpy.percentile`):
posição $= \dfrac{(n-1)\cdot p}{100}$ no vetor ordenado, interpolando entre
os dois vizinhos quando a posição não é inteira. Quartis são os percentis
25, 50 e 75.

**Regra do IQR (outliers):** um valor é outlier se estiver fora de
$[\,Q_1 - 1{,}5\cdot IQR,\; Q_3 + 1{,}5\cdot IQR\,]$, com $IQR = Q_3 - Q_1$.

**Coeficiente de variação:** $CV = \dfrac{s}{|\bar{x}|}\times 100$

**Covariância amostral:**
$$cov(x,y) = \frac{\sum_{i=1}^n (x_i-\bar{x})(y_i-\bar{y})}{n-1}$$

**Correlação de Pearson:** $r = \dfrac{cov(x,y)}{s_x \cdot s_y}$

**Regressão linear (mínimos quadrados):**
$$b_1 = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2} \qquad b_0 = \bar{y} - b_1\bar{x} \qquad R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$$

**Regra de Sturges** (nº de classes do histograma): $k = \lceil 1 + 3{,}322\cdot\log_{10}(n)\rceil$

**Densidade Normal** (curva teórica sobreposta no Módulo 4):
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}}\, e^{-\frac{(x-\mu)^2}{2\sigma^2}}, \quad \mu=\bar{x},\ \sigma=\text{desvio-padrão populacional}$$

**Densidade Exponencial:** $f(x) = \lambda e^{-\lambda x}$ para $x \ge 0$, com $\lambda = 1/\bar{x}_{positivos}$

## 4. Validação: núcleo próprio × NumPy/SciPy

Executada via `python -m pytest -v` (ver `tests/test_minhastats.py`).
A amostra de referência é a mesma dos testes: `gamma(2, 9, n=500)` para as
medidas de uma variável e um par `(x, y)` linear com ruído gaussiano para
covariância/correlação/regressão — os mesmos dados usados para checar contra
`numpy`/`scipy`, com seed fixa para reprodutibilidade.

| Função | Resultado (minhastats) | Referência (NumPy/SciPy) | Diferença | Tolerância (rtol) |
|---|---:|---:|---:|---:|
| `media` | 17.032957 | 17.032957 | 3,6 × 10⁻¹⁵ | 1e-9 |
| `mediana` | 14.146618 | 14.146618 | 0 | 1e-9 |
| `variancia(amostral=True)` | 138.086307 | 138.086307 | 0 | 1e-9 |
| `variancia(amostral=False)` | 137.810135 | 137.810135 | 0 | 1e-9 |
| `desvio_padrao` | 11.751013 | 11.751013 | 0 | 1e-9 |
| `percentil(25)` | 8.521233 | 8.521233 | 0 | 1e-6 |
| `percentil(50)` | 14.146618 | 14.146618 | 0 | 1e-6 |
| `percentil(75)` | 22.294118 | 22.294118 | 0 | 1e-6 |
| `amplitude` | 75.618765 | 75.618765 | 0 | 1e-9 |
| `coeficiente_variacao` | 68.989860% | 68.989860% | 1,4 × 10⁻¹⁴ | 1e-9 |
| `covariancia` | 28.974339 | 28.974339 | 0 | 1e-6 |
| `correlacao_pearson` | 0.953296 | 0.953296 | 1,1 × 10⁻¹⁶ | 1e-6 |
| `regressao_linear` (b0) | 7.067701 | 7.067701 (`np.polyfit`) | 8,9 × 10⁻¹⁶ | 1e-6 |
| `regressao_linear` (b1) | 3.500506 | 3.500506 (`np.polyfit`) | 4,4 × 10⁻¹⁶ | 1e-6 |
| `densidade_normal` (x=60,70,80) | ver testes | `scipy.stats.norm.pdf` | 0 | 1e-9 |
| `num_classes_sturges(500)` | 10 | $1+3{,}322\log_{10}(500)=9{,}966\Rightarrow\lceil\cdot\rceil=10$ | — | exato |

As diferenças da ordem de 10⁻¹⁴ a 10⁻¹⁶ são erro de arredondamento de ponto
flutuante puro (a ordem em que Python e NumPy somam os termos não é
idêntica) - irrelevantes frente às tolerâncias documentadas, e é exatamente
o comportamento que o guia da atividade descreve como esperado.

Casos-limite também testados e cobertos por `pytest.raises(ValueError)`:
amostra vazia em todas as medidas de tendência central/dispersão, variância
amostral com menos de 2 valores, coeficiente de variação com média zero,
correlação/regressão com variável X constante, e covariância com vetores de
tamanhos diferentes.

`docs/screenshots/07-testes.png`

## 5. Os módulos, um a um

### Módulo 0 — Dados
`docs/screenshots/01-dados.png`

A aplicação mostrou as 100 primeiras pessoas (linhas) e 17 perguntas/variáveis (colunas), sendo 7 variáveis numéricas e 10 variáveis de texto. A tabela apresenta informações como idade, profissão, estado civil, escolaridade, dívida, saldo em conta e empréstimos, permitindo visualizar diferentes características dos clientes.

### Módulo 2 — Estatística descritiva
`docs/screenshots/02-descritiva.png`
`docs/screenshots/02-descritiva(1).png`
`docs/screenshots/02-descritiva(2).png`
`docs/screenshots/02-descritiva(3).png`

A variável escolhida foi Idade da Pessoa. A aplicação mostrou média de 40,94 anos, mediana de 39 anos e desvio-padrão de 10,62 anos. O IQR detectou 487 valores muito fora do normal (outliers). A interpretação automática indica que a maior parte das pessoas está concentrada nos valores menores; visualmente, o histograma confirma uma concentração maior nas idades mais baixas e uma cauda para a direita, indicando assimetria positiva.


### Módulo 3 — Simulações (LGN e TCL)
`docs/screenshots/03-simulacoes.png`
`docs/screenshots/03-simulacoes(1).png`

Na simulação da Lei dos Grandes Números, as frequências relativas das cinco pessoas começam com bastante oscilação, mas vão se aproximando de 0,5 (50%) conforme o número de jogadas aumenta. No print, por volta de 600–800 jogadas as linhas já aparecem visualmente bastante próximas da linha de 50%, ficando ainda mais estáveis com milhares de jogadas.

No TCL, o histograma das médias apresenta um formato aproximadamente de sino, acompanhado pela curva normal. Isso ilustra que, conforme o tamanho dos grupos aumenta, a distribuição das médias tende a ficar mais próxima de uma distribuição normal.


### Módulo 4 — Distribuições teóricas
`docs/screenshots/04-distribuicoes.png`

A variável escolhida foi Idade da Pessoa. O gráfico compara os dados reais com um molde em forma de sino (Normal) e um molde em queda rápida (Exponencial). Visualmente, os dados apresentam concentração maior aproximadamente entre 30 e 50 anos e uma cauda à direita, portanto o formato real não se encaixa perfeitamente em uma distribuição Normal, embora a curva Normal acompanhe parte do comportamento central.


### Módulo 5 — Correlação e regressão
`docs/screenshots/05-regressao.png`

As duas variáveis comparadas foram Idade da Pessoa e Dinheiro na Conta (Saldo). A aplicação mostrou r = 0,10 e R² = 1,0%, indicando uma relação linear muito fraca entre as variáveis. A equação apresentada foi Y = 214,5 + (28,038 × X).

Como exemplo, usando X = 40 anos, a equação fornece:

Ŷ = 214,5 + (28,038 × 40) = 1.336,02


### Módulo 6 — Descobertas
`docs/screenshots/06-descobertas.png`
`docs/screenshots/06-descobertas(1).png`

## 6. As três descobertas


**Descoberta 1 — Poucas pessoas aceitam a proposta.**
Apenas **[PREENCHER]%** das 45.211 pessoas contatadas aceitaram o depósito a
prazo (gráfico de barras "Aceitou vs Recusou"). *Limite honesto: é a taxa de
conversão desta campanha específica (2008–2010, Portugal) — não deve ser
generalizada para outros bancos, países ou períodos.*

**Descoberta 2 — Conversas mais longas têm mais sucesso.**
Entre quem aceitou, a ligação durou em média **[PREENCHER]** segundos;
entre quem recusou, **[PREENCHER]** segundos (gráfico "Tempo Médio da
Conversa"). *Limite honesto: é associação, não causa — pode ser que quem já
está mais interessado deixe a ligação se estender, e não que a ligação longa
"convença" a pessoa.*

**Descoberta 3 — Histórico de campanha anterior importa.**
Entre quem já tinha aceitado uma campanha anterior (`poutcome = success`),
**[PREENCHER]%** aceitou de novo, contra a taxa geral de **[PREENCHER]%**.
*Limite honesto: correlação não implica causalidade — clientes que aceitam
uma vez podem simplesmente confiar mais no banco, ou ter perfil financeiro
diferente, sem que a campanha anterior seja a causa direta.*

## 7. Limitações gerais

- Os dados vão até 2010; padrões de comportamento financeiro e de
  telemarketing mudaram bastante desde então (ex.: crescimento do
  WhatsApp/e-mail como canal), então as descobertas não devem ser lidas como
  válidas para o presente.
- O dataset é de **um único banco português** — resultados não generalizam
  automaticamente para outras instituições ou países.
- Todas as afirmações de "associação" no relatório são exatamente isso:
  correlação, não causalidade comprovada.

---

### Passo a passo para preencher os `[PREENCHER]`

1. Rode `streamlit run app.py` (ou abra a versão publicada).
2. Para cada módulo, clique na aba correspondente, espere carregar, e tire o
   print (salve com o nome exato indicado, dentro de `docs/screenshots/`).
3. Copie os números/textos exatos que a aplicação mostrar (ela já escreve as
   frases de interpretação prontas — é só colar aqui).
4. Rode `python -m pytest -v`, tire o print do terminal e salve como
   `07-testes.png`.
5. Confira se todas as imagens abrem corretamente no GitHub antes de gerar o
   PDF final.
