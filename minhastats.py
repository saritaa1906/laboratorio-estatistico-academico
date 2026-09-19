"""Núcleo estatístico — Etapa 2 do guia.

Regra de ouro: nenhuma função aqui usa numpy, pandas ou scipy para CALCULAR.
sum, len, sorted, min, max, math (sqrt/exp/log/pi) são Python puro e por isso
permitidos pelo enunciado. NumPy/SciPy só aparecem em test_minhastats.py, como
referência para validar estes resultados — nunca dentro da própria conta.
"""
from math import ceil, exp, log10, pi, sqrt

# ==========================================
# Utilitário interno
# ==========================================
def _numeros(valores):
    dados = [float(v) for v in valores]
    if not dados:
        raise ValueError("A amostra não pode estar vazia.")
    return dados


# ==========================================
# Medidas de tendência central
# ==========================================
def media(valores):
    """Média aritmética: soma de todos os valores dividida pela contagem."""
    dados = _numeros(valores)
    return sum(dados) / len(dados)


def mediana(valores):
    """Valor central da amostra ordenada (média dos dois centrais se n for par)."""
    dados = sorted(_numeros(valores))
    n = len(dados)
    meio = n // 2
    return dados[meio] if n % 2 else (dados[meio - 1] + dados[meio]) / 2


def moda(valores):
    """Valor(es) de maior frequência. Pode haver mais de um — devolve lista ordenada."""
    dados = _numeros(valores)
    contagens = {}
    for valor in dados:
        contagens[valor] = contagens.get(valor, 0) + 1
    maior = max(contagens.values())
    return sorted(valor for valor, contagem in contagens.items() if contagem == maior)


# ==========================================
# Medidas de dispersão
# ==========================================
def amplitude(valores):
    """Máximo menos mínimo."""
    dados = _numeros(valores)
    return max(dados) - min(dados)


def variancia(valores, amostral=True):
    """Variância. amostral=True divide por n-1 (correção de Bessel); False, por n."""
    dados = _numeros(valores)
    divisor = len(dados) - 1 if amostral else len(dados)
    if divisor <= 0:
        raise ValueError("Variância amostral exige pelo menos dois valores.")
    centro = media(dados)
    return sum((valor - centro) ** 2 for valor in dados) / divisor


def desvio_padrao(valores, amostral=True):
    """Raiz quadrada da variância."""
    return sqrt(variancia(valores, amostral=amostral))


def percentil(valores, p):
    """Percentil com interpolação linear (mesma convenção do numpy.percentile)."""
    if not 0 <= p <= 100:
        raise ValueError("O percentil deve estar entre 0 e 100.")
    dados = sorted(_numeros(valores))
    posicao = (len(dados) - 1) * p / 100
    inferior = int(posicao)
    superior = min(inferior + 1, len(dados) - 1)
    fracao = posicao - inferior
    return dados[inferior] + fracao * (dados[superior] - dados[inferior])


def quartis(valores):
    """Retorna (Q1, Q2, Q3)."""
    return percentil(valores, 25), percentil(valores, 50), percentil(valores, 75)


def limites_outliers_iqr(valores, k=1.5):
    """Limites [Q1 - k*IQR, Q3 + k*IQR] pela regra do IQR (padrão k=1.5)."""
    q1, _, q3 = quartis(valores)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def contar_outliers_iqr(valores, k=1.5):
    """Quantos valores caem fora dos limites do IQR."""
    dados = _numeros(valores)
    inferior, superior = limites_outliers_iqr(dados, k=k)
    return sum(1 for v in dados if v < inferior or v > superior)


def coeficiente_variacao(valores, amostral=True):
    """CV = desvio-padrão / |média| × 100."""
    centro = media(valores)
    if centro == 0:
        raise ValueError("O coeficiente de variação é indefinido para média zero.")
    return desvio_padrao(valores, amostral=amostral) / abs(centro) * 100


# ==========================================
# Duas variáveis: covariância, correlação, regressão
# ==========================================
def covariancia(x, y, amostral=True):
    """Covariância amostral (padrão) ou populacional entre dois vetores pareados."""
    xs, ys = _numeros(x), _numeros(y)
    if len(xs) != len(ys):
        raise ValueError("As variáveis devem ter o mesmo tamanho.")
    divisor = len(xs) - 1 if amostral else len(xs)
    if divisor <= 0:
        raise ValueError("Covariância amostral exige pelo menos dois pares.")
    mx, my = media(xs), media(ys)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / divisor


def correlacao_pearson(x, y):
    """Correlação linear de Pearson: cov(x,y) / (desvio_x * desvio_y)."""
    sx, sy = desvio_padrao(x), desvio_padrao(y)
    if sx == 0 or sy == 0:
        raise ValueError("Correlação é indefinida para variável constante.")
    return covariancia(x, y) / (sx * sy)


def regressao_linear(x, y):
    """Mínimos quadrados simples. Retorna (intercepto b0, inclinação b1, R²)."""
    xs, ys = _numeros(x), _numeros(y)
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("A regressão exige ao menos dois pares de mesmo tamanho.")
    mx, my = media(xs), media(ys)
    denominador = sum((valor - mx) ** 2 for valor in xs)
    if denominador == 0:
        raise ValueError("Não há regressão quando X é constante.")
    inclinacao = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / denominador
    intercepto = my - inclinacao * mx
    previstos = [intercepto + inclinacao * valor for valor in xs]
    total = sum((valor - my) ** 2 for valor in ys)
    residuos = sum((real - previsto) ** 2 for real, previsto in zip(ys, previstos))
    r2 = 1 - residuos / total if total else 1.0
    return intercepto, inclinacao, r2


def prever(intercepto, inclinacao, x, x_min=None, x_max=None):
    """y = b0 + b1*x. Se x_min/x_max forem dados e x estiver fora, avisa (extrapolação)."""
    fora_da_faixa = (
        (x_min is not None and x < x_min) or (x_max is not None and x > x_max)
    )
    return intercepto + inclinacao * x, fora_da_faixa


# ==========================================
# Regra de Sturges (nº de classes de um histograma)
# ==========================================
def num_classes_sturges(n):
    """k = 1 + 3,322*log10(n), arredondado para cima."""
    if n <= 0:
        raise ValueError("n deve ser positivo.")
    return max(1, ceil(1 + 3.322 * log10(n)))


# ==========================================
# Densidades teóricas (para sobrepor aos histogramas do Módulo 4)
# ==========================================
def densidade_normal(x, mu, sigma):
    """f(x) da Normal(mu, sigma) — fórmula fechada, sem scipy."""
    if sigma <= 0:
        raise ValueError("sigma deve ser positivo.")
    return (1 / (sigma * sqrt(2 * pi))) * exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def densidade_exponencial(x, taxa):
    """f(x) da Exponencial(taxa), taxa = 1/média. Zero para x < 0."""
    if taxa <= 0:
        raise ValueError("taxa deve ser positiva.")
    return 0.0 if x < 0 else taxa * exp(-taxa * x)


def grade_linear(inicio, fim, pontos=200):
    """Equivalente ao numpy.linspace, em Python puro — só para posicionar o eixo X."""
    if pontos < 2:
        return [inicio]
    passo = (fim - inicio) / (pontos - 1)
    return [inicio + i * passo for i in range(pontos)]


def frequencia_relativa_acumulada(sequencia_binaria):
    """Para a Lei dos Grandes Números: fração acumulada de 1s a cada nova jogada."""
    acumulada = []
    soma = 0
    for i, valor in enumerate(sequencia_binaria, start=1):
        soma += valor
        acumulada.append(soma / i)
    return acumulada
