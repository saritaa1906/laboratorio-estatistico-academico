# Laboratório Estatístico Interativo

**Disciplina:** Matemática e Estatística para Computação  
**Integrante:** Sara Martins Oliveira de Sousa - RA 72650204

Aplicação web em Python e Streamlit que transforma um conjunto de dados real em um laboratório interativo de estatística. O foco do projeto é implementar os cálculos fundamentais em uma biblioteca própria e validá-los contra referências consolidadas.

## Dataset

O projeto utiliza exclusivamente o dataset público [Bank Marketing - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank%2Bmarket). A base reúne **45.211 registros**, sete variáveis numéricas e dez variáveis categóricas sobre campanhas de marketing bancário por telefone.

## Funcionalidades

- **Dados reais:** carregamento reproduzível da fonte oficial UCI.
- **Núcleo estatístico:** média, mediana, moda, amplitude, variâncias, desvios-padrão, percentis, quartis, coeficiente de variação, covariância, Pearson e regressão linear.
- **Descritiva:** tabelas de frequências, histogramas, boxplots, gráficos de barras/pizza, classes, IQR e interpretação automática.
- **Simulações:** Lei dos Grandes Números e Teorema Central do Limite, com parâmetros controláveis.
- **Distribuições:** comparação visual da variável com as distribuições Normal e Exponencial.
- **Regressão:** dispersão, reta de mínimos quadrados, equação, Pearson, R² e predição interativa.
- **Descobertas:** três achados descritivos sobre adesão, duração das chamadas e canal de contato.

## Estrutura do repositório

```text
app.py                    # Interface Streamlit e visualizações
minhastats.py             # Núcleo matemático implementado pela autora
data_sources.py           # Carregamento do dataset oficial
tests/test_minhastats.py  # Validação automatizada contra NumPy/SciPy
RELATORIO.md              # Relatório técnico da atividade
requirements.txt          # Dependências para reprodução
```

## Como executar

Pré-requisito: Python 3.10 ou superior.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Após o comando, abra no navegador o endereço apresentado pelo Streamlit, normalmente `http://localhost:8501`.

## Como validar os cálculos

```powershell
python -m pytest -q
```

Os testes comparam os resultados de `minhastats.py` com NumPy, SciPy e `statistics`, usando tolerância relativa de `1e-12` nos cálculos de ponto flutuante.

## Evidências da aplicação

Antes da entrega, inclua neste README de 3 a 6 capturas de tela reais da aplicação em funcionamento: uma visão geral, uma análise descritiva, uma simulação, uma comparação de distribuições, uma regressão e a aba de descobertas. Isso facilita a correção mesmo sem executar o projeto.

## Entrega

O relatório técnico está em [RELATORIO.md](RELATORIO.md). O envio no ambiente virtual deve ser feito por meio de um PDF com a identificação, os links do dataset, repositório e vídeo, além de um resumo executivo.
