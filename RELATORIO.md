# Relatório Técnico — Laboratório Estatístico Interativo

**Disciplina:** Matemática e Estatística para Computação
**Integrante:** Sara Martins Oliveira de Sousa — RA 72650204
**Repositório:** <https://github.com/saritaa1906/laboratorio-estatistico-academico>
**Vídeo:** https://youtu.be/QshIyZCvTrk
---

## 1. Dataset e justificativa

Usei o **Bank Marketing** do UCI Machine Learning Repository ([link original](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing)), com **45.211 registros** de campanhas de telemarketing de um banco português oferecendo depósitos a prazo. A base tem 7 variáveis numéricas (`age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`) e 10 categóricas (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome` e o alvo binário `y`), atendendo com folga o mínimo exigido (1.000 registros, 4 numéricas, 2 categóricas).

Escolhi esse dataset porque ele levanta uma pergunta de negócio real e intuitiva — *o que faz uma pessoa aceitar guardar dinheiro no banco quando ligam oferecendo?* — algo com o qual lido de perto na minha rotina de trabalho. Além disso, as variáveis têm formatos bem diferentes (`age` é aproximadamente simétrica; `balance` e `duration` são fortemente assimétricas à direita), o que torna os Módulos 3 e 4 (TCL e ajuste de distribuições) ricos para discussão.

## 2. Decisões de tratamento dos dados

- **Nulos:** o dataset oficial não tem valores ausentes (`Has Missing Values? No`, conforme a página do UCI); nenhuma linha foi removida.
- **Categoria `unknown`:** em `job`, `education`, `contact` e `poutcome` é uma resposta válida do levantamento, não um nulo; foi mantida como categoria própria para não reduzir a amostra sem necessidade.
- **Tipagem:** as colunas numéricas são convertidas para `float` e passadas ao núcleo como listas puras (`.dropna().astype(float).tolist()`), separando "carregar" (Pandas) de "calcular" (`minhastats.py`).
- **`pdays = -1`:** código do dataset para "nunca contatado antes"; não é erro nem outlier de digitação e foi mantido.
- **Gráficos:** o gráfico de dispersão usa no máximo 10.000 pontos apenas por desempenho de renderização; todas as estatísticas são calculadas sobre a coluna completa. Nos histogramas de distribuições teóricas o eixo vai até o percentil 99 por legibilidade, mas as densidades usam o *n* total.
- **Sem dados sintéticos:** se o dataset oficial não puder ser carregado, a aplicação exibe erro e para; ela nunca substitui os dados por dados inventados.

## 3. Núcleo estatístico — fórmulas implementadas

Todas as funções estão em `minhastats.py`, em Python puro (somente `sum`, `len`, `sorted`, `min`, `max` e `math`). NumPy/SciPy aparecem **apenas** em `tests/test_minhastats.py`, como referência. A aplicação (`app.py`) usa NumPy só para sortear números aleatórios nas simulações, nunca para calcular estatísticas.

**Média:**
$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

**Mediana:** valor central da amostra ordenada; média dos dois centrais se $n$ é par.

**Moda:** valor(es) de maior frequência (lista, pois pode haver mais de uma).

**Amplitude:** $\max(x) - \min(x)$

**Variância amostral (correção de Bessel) e populacional:**
$$s^2 = \frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n-1} \qquad\qquad \sigma^2 = \frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n}$$

**Desvio-padrão:** $s = \sqrt{s^2}$ (e $\sigma = \sqrt{\sigma^2}$).

**Percentil $p$** (interpolação linear, mesma convenção do `numpy.percentile`): posição $h = \dfrac{(n-1)\,p}{100}$ no vetor ordenado $x_{(0)} \le \dots \le x_{(n-1)}$, com
$$P_p = x_{(\lfloor h \rfloor)} + (h - \lfloor h \rfloor)\,\bigl(x_{(\lfloor h \rfloor+1)} - x_{(\lfloor h \rfloor)}\bigr)$$
Os quartis são $Q_1 = P_{25}$, $Q_2 = P_{50}$, $Q_3 = P_{75}$.

**Regra do IQR (outliers):** $IQR = Q_3 - Q_1$; é outlier todo valor fora de
$$\left[\,Q_1 - 1{,}5\cdot IQR \;;\; Q_3 + 1{,}5\cdot IQR\,\right]$$

**Coeficiente de variação:**
$$CV = \frac{s}{|\bar{x}|}\times 100$$

**Covariância amostral:**
$$\text{cov}(x,y) = \frac{\sum_{i=1}^n (x_i-\bar{x})(y_i-\bar{y})}{n-1}$$

**Correlação de Pearson:**
$$r = \frac{\text{cov}(x,y)}{s_x \, s_y}$$

**Regressão linear (mínimos quadrados):**
$$b_1 = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2} \qquad b_0 = \bar{y} - b_1\bar{x} \qquad R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$$

**Assimetria (Fisher, momentos populacionais):**
$$g_1 = \frac{m_3}{m_2^{3/2}}, \qquad m_k = \frac{1}{n}\sum_{i=1}^n (x_i-\bar{x})^k$$
$g_1 > 0$: cauda à direita; $g_1 < 0$: cauda à esquerda. Classificação usada na interpretação automática: $|g_1|<0{,}5$ aproximadamente simétrica; $0{,}5\le|g_1|<1$ moderadamente assimétrica; $|g_1|\ge 1$ fortemente assimétrica.

**Regra de Sturges** (nº de classes): $k = \lceil 1 + 3{,}322\cdot\log_{10}(n) \rceil$. A largura das classes é $h=(\max-\min)/k$; a densidade de cada classe é $f_j = \dfrac{n_j}{n\,h}$, para que o histograma possa ser comparado com curvas teóricas.

**Densidade Normal:**
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}}\,e^{-\frac{(x-\mu)^2}{2\sigma^2}}, \qquad \mu=\bar{x},\ \sigma=\text{desvio-padrão populacional}$$

**Densidade Exponencial:** $f(x) = \lambda e^{-\lambda x}$ para $x \ge 0$, com $\lambda = 1/\bar{x}$ (média dos valores positivos).

**Densidade Uniforme:** $f(x) = \dfrac{1}{b-a}$ em $[a,b]$, com $a=\min$ e $b=\max$.

**Teorema Central do Limite (simulação):** para amostras de tamanho $n$ de uma população com média $\mu$ e desvio $\sigma$, as médias amostrais aproximam-se de $\mathcal{N}\!\left(\mu,\ \sigma/\sqrt{n}\right)$.

## 4. Validação: núcleo próprio × NumPy/SciPy

Executada com `python -m pytest -v` (arquivo `tests/test_minhastats.py`, **42 testes**). Amostra de referência: `gamma(2, 9, n=500)` (seed 0) para as medidas de uma variável e o par `(X, Y)` linear com ruído gaussiano (`Y = 3,44·X + 7,1 + ruído`, seeds 2 e 3, n=200) para covariância/correlação/regressão.

| Função | Resultado (minhastats) | Referência (NumPy/SciPy) | Diferença | Tolerância (rtol) |
|---|---|---|---|---|
| `media` | 17.032957 | 17.032957 | 3,6e-15 | 1e-9 |
| `mediana` | 14.146618 | 14.146618 | 0 | 1e-9 |
| `variancia` (amostral, n−1) | 138.086307 | 138.086307 (`np.var(ddof=1)`) | 0 | 1e-9 |
| `variancia` (populacional, n) | 137.810135 | 137.810135 (`np.var(ddof=0)`) | 0 | 1e-9 |
| `desvio_padrao` (amostral) | 11.751013 | 11.751013 | 0 | 1e-9 |
| `percentil(25)` | 8.521233 | 8.521233 | 0 | 1e-6 |
| `percentil(50)` | 14.146618 | 14.146618 | 0 | 1e-6 |
| `percentil(75)` | 22.294118 | 22.294118 | 0 | 1e-6 |
| `amplitude` | 75.618765 | 75.618765 | 0 | 1e-9 |
| `coeficiente_variacao` (%) | 68.989860 | 68.989860 | 1,4e-14 | 1e-9 |
| `coeficiente_assimetria` ($g_1$) | 1.423140 | 1.423140 (`scipy.stats.skew`) | 1,6e-15 | 1e-9 |
| `covariancia` | 30.789230 | 30.789230 (`np.cov`) | 7,1e-15 | 1e-6 |
| `correlacao_pearson` | 0.959062 | 0.959062 (`np.corrcoef`) | 3,3e-16 | 1e-6 |
| `regressao_linear` ($b_0$) | 6.734216 | 6.734216 (`np.polyfit`) | 2,7e-15 | 1e-6 |
| `regressao_linear` ($b_1$) | 3.541661 | 3.541661 (`np.polyfit`) | 0 | 1e-6 |
| `densidade_normal` | 0.039894 | 0.039894 (`scipy.stats.norm.pdf`) | 6,9e-18 | 1e-9 |
| `densidade_exponencial` | 0.060653 | 0.060653 (`scipy.stats.expon.pdf`) | 6,9e-18 | 1e-9 |
| `densidade_uniforme` | — | `scipy.stats.uniform.pdf` | ≈ 0 | 1e-9 |
| `num_classes_sturges(500)` | 10 | $\lceil 1+3{,}322\log_{10}500\rceil = 10$ | — | exato |
| `tabela_frequencias` | contagens por classe | `np.histogram` (mesmas classes) | 0 | exato |

As diferenças de 1e-14 a 1e-18 são erro de arredondamento de ponto flutuante (a ordem das somas em Python e NumPy não é idêntica), muito abaixo das tolerâncias documentadas.

Casos-limite cobertos com `pytest.raises(ValueError)`: amostra vazia em todas as medidas; variância amostral com menos de 2 valores; coeficiente de variação com média zero; correlação/regressão com variável constante; covariância com vetores de tamanhos diferentes; percentil fora de [0, 100]; assimetria e tabela de frequências de variável constante.

![Testes passando](docs/screenshots/07-testes.png)

## 5. Os módulos, um a um

### Módulo 0 — Dados
![Aba de dados](docs/screenshots/01-dados.png)

Mostra o total de registros (45.211), as 7 variáveis numéricas e as 10 categóricas, o tratamento aplicado e os 100 primeiros registros.

### Módulo 2 — Estatística descritiva
![Descritiva](docs/screenshots/02-descritiva.png)
![Descritiva](docs/screenshots/02-descritiva(1).png)

Para a variável escolhida a aplicação exibe: medidas de tendência central e dispersão (calculadas por `minhastats`), **tabela de frequências por classes** (Sturges) com frequência absoluta, relativa e acumulada, histograma com média e mediana marcadas, boxplot com os limites do IQR e a **interpretação textual automática** (assimetria por $g_1$, proporção de outliers e nível de variabilidade pelo CV). Exemplo com `age`: média 40,94 anos, mediana 39, desvio-padrão 10,62; o IQR detecta 487 outliers e $g_1$ ≈ 0,68 indica assimetria moderada à direita. Para variáveis categóricas há tabela de frequências (absoluta, relativa e acumulada), barras, pizza (≤ 10 categorias) e a moda.

### Módulo 3 — Simulação de Monte Carlo
![Simulações](docs/screenshots/03-simulacoes.png)
![Simulações](docs/screenshots/03-simulacoes(1).png)

**(a) Lei dos Grandes Números:** lançamentos de moeda simulados, com controle do nº de lançamentos, do nº de simulações, da probabilidade $p$ e da semente. Com poucos lançamentos as frequências oscilam muito; com milhares, todas as linhas se aproximam de $p$ (o app mostra numericamente a maior distância a $p$ com 10 lançamentos e com $n$ lançamentos).

**(b) Teorema Central do Limite:** sorteio repetido de amostras de tamanho $n$ de uma variável do dataset (por padrão `duration`, fortemente assimétrica), com a média de cada amostra calculada por `ms.media`. Sliders controlam $n$ e o nº de repetições. A aplicação compara a média e o desvio das médias com os valores teóricos $\mu$ e $\sigma/\sqrt{n}$, mostra a queda da assimetria e sobrepõe a curva $\mathcal{N}(\mu,\sigma/\sqrt{n})$: com $n$ pequeno as médias ainda são assimétricas; com $n=30$ ou mais formam um sino — mesmo com dados originais longe de normais.

### Módulo 4 — Distribuições teóricas
![Distribuições](docs/screenshots/04-distribuicoes.png)

O usuário escolhe a variável e as curvas candidatas (Normal, Exponencial, Uniforme), com parâmetros estimados dos dados e histograma em densidade. A aplicação calcula o **erro médio entre histograma e curva** e gera uma discussão automática (regra empírica 68–95, assimetria e CV).

Discussão para `age`: os dados se concentram entre 30 e 50 anos, com cauda à direita ($g_1\approx 0{,}68$); a Normal acompanha razoavelmente a parte central, mas não a cauda, e a Exponencial não serve (o CV de `age` é ≈ 26%, muito abaixo dos 100% da Exponencial). Para `duration` ou `balance` o contraste se inverte: a Normal falha porque a variável tem forte cauda à direita, e uma Gama/Exponencial descreveria melhor.

### Módulo 5 — Correlação e regressão
![Regressão](docs/screenshots/05-regressao.png)
![Regressão](docs/screenshots/05-regressao(1).png)

Dispersão, reta de mínimos quadrados, equação, $r$, $R^2$, **interpretação dos coeficientes** ("cada unidade a mais de X está *associada*, em média, a $b_1$ unidades a mais/menos de Y"), alerta permanente de que **correlação não implica causalidade** (com exemplo do próprio dataset: `duration` × aceitação) e predição interativa com **aviso de extrapolação** fora da faixa observada de X. Há também a matriz de correlação de todas as variáveis numéricas, calculada com `ms.correlacao_pearson`.

Exemplo: `age` × `balance` → $r = 0{,}10$, $R^2 = 1{,}0\%$, $\hat{y} = 214{,}5 + 28{,}038\,x$: relação linear muito fraca; para $x=40$ anos, $\hat{y}\approx 1.336{,}02$. Ou seja, a idade praticamente não explica o saldo.

### Módulo 6 — Descobertas
![Descobertas](docs/screenshots/06-descobertas.png)
![Descobertas](docs/screenshots/06-descobertas(1).png)
![Descobertas](docs/screenshots/06-descobertas(2).png)

## 6. As três descobertas

**Descoberta 1 — Poucas pessoas aceitam a proposta.** Apenas 11,7% dos 45.211 clientes contatados aceitaram o depósito a prazo (gráfico "Aceitaram × recusaram"). *Limite:* é a taxa desta campanha (Portugal, 2008–2010); não deve ser generalizada.

**Descoberta 2 — Conversas mais longas acompanham a aceitação.** Entre quem aceitou, a ligação durou em média 537 s; entre quem recusou, 221 s (gráfico com média e mediana). *Limite:* é associação, não causa — clientes já interessados tendem a conversar mais.

**Descoberta 3 — O histórico de campanha anterior importa.** Entre quem já tinha aceitado uma campanha anterior (`poutcome = success`), 64,7% aceitou de novo, contra 11,7% na taxa geral. *Limite:* correlação não implica causalidade — quem aceita uma vez pode confiar mais no banco ou ter perfil financeiro diferente.

## 7. Limitações gerais

- Os dados vão até 2010; o comportamento financeiro e de telemarketing mudou desde então (WhatsApp, e-mail, aplicativos), então as descobertas não valem automaticamente para hoje.
- O dataset é de um único banco português; os resultados não generalizam para outras instituições ou países.
- Todas as afirmações de "associação" são correlação, não causalidade comprovada.
