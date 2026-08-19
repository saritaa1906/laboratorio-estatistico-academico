# Laboratório Estatístico

Projeto acadêmico em Python e Streamlit desenvolvido por **Sara Martins Oliveira de Sousa — RA 72650204**.

O laboratório utiliza exclusivamente o dataset público [Bank Marketing, da UCI](https://archive.ics.uci.edu/dataset/222/bank%2Bmarket), com 45.211 registros, sete variáveis numéricas e dez categóricas.

## Conteúdo

- estatística descritiva implementada em `minhastats.py`;
- tabelas de frequência, histogramas e boxplots;
- Lei dos Grandes Números e Teorema Central do Limite;
- distribuições Normal e Exponencial;
- correlação de Pearson e regressão linear;
- três descobertas interpretadas;
- testes automatizados dos cálculos.

## Executar no VS Code

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Também é possível pressionar `F5` e selecionar **Laboratório acadêmico**.

## Testes

```powershell
python -m pytest -q
```
