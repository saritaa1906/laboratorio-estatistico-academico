# Relatório - Laboratório Estatístico Interativo

**Disciplina:** Matemática e Estatística para Computação  
**Integrante:** Sara Martins Oliveira de Sousa - RA 72650204

## Objetivo

Desenvolver um laboratório estatístico interativo em Python para explorar dados reais, aplicar estatística descritiva e probabilidade e construir, sem funções estatísticas prontas, os cálculos que sustentam a análise.

## Base de dados

Foi utilizado o dataset público [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank%2Bmarket), da UCI Machine Learning Repository. A base contém **45.211 registros** de campanhas telefônicas, sete variáveis numéricas e dez variáveis categóricas.

A base foi escolhida por reunir volume suficiente para simulações, variáveis de naturezas diferentes para as visualizações e uma variável-resposta categórica (`y`, adesão ou não ao depósito a prazo). O carregamento é feito diretamente da fonte oficial por `data_sources.py`, sem cópias manuais do dataset.

## Implementação

O núcleo `minhastats.py` implementa média, mediana, moda, amplitude, quartis, variância, desvio-padrão, coeficiente de variação, covariância, correlação de Pearson e regressão por mínimos quadrados. A interface está em `app.py`, o carregamento reproduzível está em `data_sources.py` e a validação automatizada está em `tests/test_minhastats.py`.

As medidas exibidas ao usuário são produzidas pelas funções próprias. NumPy, SciPy e `statistics` são utilizados somente como referência nos testes.

### Fórmulas utilizadas

```text
Média:              x̄ = (Σ xi) / n
Variância amostral: s² = Σ(xi - x̄)² / (n - 1)
Variância popul.:   σ² = Σ(xi - μ)² / n
Desvio-padrão:      s = √s²
Covariância:        cov(X,Y) = Σ[(xi - x̄)(yi - ȳ)] / (n - 1)
Pearson:            r = cov(X,Y) / (sx sy)
Reta de regressão:  ŷ = a + bx
Inclinação:         b = Σ[(xi - x̄)(yi - ȳ)] / Σ(xi - x̄)²
Intercepto:         a = ȳ - b x̄
```

Os percentis usam interpolação linear compatível com o padrão do NumPy. O R² é calculado por `1 - SQresíduos/SQtotal`.

### Validação

Os testes comparam média, mediana, moda, amplitude, variâncias, desvios-padrão, quartis, percentis, coeficiente de variação, covariância, correlação e regressão com NumPy/SciPy/`statistics`. A tolerância relativa documentada para números de ponto flutuante é `1e-12`.

Também são testados casos inválidos: amostra vazia, variância amostral com um único valor, correlação com variável constante, percentil fora do intervalo e vetores de tamanhos diferentes. A execução é feita com `python -m pytest -q`.

## Módulos interativos

### Estatística descritiva

Para variáveis numéricas, a aplicação apresenta medidas de posição e dispersão calculadas pelo núcleo próprio, histograma, boxplot, tabela de frequência por classes, regra de Sturges para o número de classes e detecção de outliers pelo intervalo interquartil (IQR). A relação entre média e mediana gera uma interpretação automática de assimetria.

Para variáveis categóricas, são exibidas frequência absoluta, frequência relativa, gráfico de barras e, quando legível, gráfico de pizza.

### Probabilidade e simulação

A Lei dos Grandes Números é demonstrada por lançamentos simulados de moeda: a frequência relativa de cara tende a 0,5 com o aumento do número de lançamentos. O usuário controla lançamentos, experimentos e semente aleatória.

No Teorema Central do Limite, a aplicação coleta amostras repetidas com reposição de uma variável numérica do dataset, calcula as médias com `minhastats.media` e sobrepõe uma curva Normal. O usuário controla o tamanho das amostras e o número de repetições.

### Distribuições teóricas

O histograma de uma variável numérica é comparado com curvas Normal e Exponencial, cujos parâmetros são estimados a partir dos dados. A tela informa uma distância KS descritiva para orientar a comparação visual. Essa medida é interpretada com cautela, pois os parâmetros foram estimados na própria amostra.

### Correlação e regressão linear

O usuário escolhe X e Y. O painel mostra dispersão, coeficiente de Pearson calculado pela biblioteca própria, reta de mínimos quadrados, equação, R² e uma predição para um valor de X informado na interface. A interpretação apresenta sentido e intensidade da associação e alerta explicitamente que correlação não implica causalidade.

## Descobertas

1. **Adesão é minoritária.** Somente **11,70%** das campanhas registraram `yes`. Por isso, contagens absolutas devem ser interpretadas junto com proporções.
2. **Duração e desfecho estão associados descritivamente.** Ligações com adesão duraram, em média, **537,3 segundos**, contra **221,2 segundos** sem adesão. Esse resultado não autoriza concluir que aumentar artificialmente a duração causa adesão, pois a duração pode ser conhecida apenas após o resultado da chamada.
3. **Os canais apresentam taxas de adesão diferentes.** `cellular` registrou **14,92%**, `telephone` **13,42%** e `unknown` **4,07%** de adesão. A diferença sugere associação, mas pode refletir perfis de clientes, período da campanha e estratégias de contato distintas.

Os valores e gráficos dessas três descobertas são gerados na aba **Descobertas** da aplicação. Para a versão final, devem ser incluídas capturas de tela dessa aba e das demais análises relevantes no repositório e no vídeo.

## Reprodutibilidade

O dataset é baixado diretamente da fonte oficial. As dependências estão registradas em `requirements.txt` e os cálculos próprios são conferidos por testes automatizados.

Antes do envio, devem estar disponíveis publicamente:

1. o repositório GitHub com código, README, relatório, testes e capturas da aplicação;
2. o link original do dataset da UCI;
3. o vídeo de demonstração de 3 a 5 minutos, não listado no YouTube ou compartilhado no Drive;
4. o PDF de envio com identificação, os três links e um resumo executivo de uma página.
