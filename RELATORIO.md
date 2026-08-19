# Relatório — Laboratório Estatístico

**Integrante:** Sara Martins Oliveira de Sousa — RA 72650204

## Base de dados

Foi utilizado o dataset público Bank Marketing, da UCI Machine Learning Repository. A base contém 45.211 registros de campanhas telefônicas, sete variáveis numéricas e dez variáveis categóricas.

## Implementação

O núcleo `minhastats.py` implementa média, mediana, moda, amplitude, quartis, variância, desvio-padrão, coeficiente de variação, covariância, correlação de Pearson e regressão por mínimos quadrados. A aplicação apresenta estatística descritiva, simulações da Lei dos Grandes Números e do Teorema Central do Limite, comparações de distribuições e análise de regressão.

## Descobertas

O painel calcula a proporção de adesões, compara a duração média das chamadas conforme o desfecho e avalia diferenças entre canais de contato. As interpretações são descritivas: associação não implica causalidade.

## Reprodutibilidade

O dataset é baixado diretamente da fonte oficial. As dependências estão registradas em `requirements.txt` e os cálculos próprios são conferidos por testes automatizados.
