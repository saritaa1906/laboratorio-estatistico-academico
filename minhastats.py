"""Núcleo estatístico implementado sem funções estatísticas prontas."""

from math import sqrt


def _numeros(valores):
    dados = [float(v) for v in valores]
    if not dados:
        raise ValueError("A amostra não pode estar vazia.")
    return dados


def media(valores):
    """Calcula a média aritmética."""
    dados = _numeros(valores)
    return sum(dados) / len(dados)


def mediana(valores):
    """Calcula a mediana após ordenar os valores."""
    dados = sorted(_numeros(valores))
    n = len(dados)
    meio = n // 2
    return dados[meio] if n % 2 else (dados[meio - 1] + dados[meio]) / 2


def moda(valores):
    """Retorna todas as modas, em ordem crescente."""
    dados = _numeros(valores)
    contagens = {}
    for valor in dados:
        contagens[valor] = contagens.get(valor, 0) + 1
    maior = max(contagens.values())
    return sorted(valor for valor, contagem in contagens.items() if contagem == maior)


def amplitude(valores):
    """Calcula máximo menos mínimo."""
    dados = _numeros(valores)
    return max(dados) - min(dados)


def variancia(valores, amostral=True):
    """Calcula variância amostral (n-1) ou populacional (n)."""
    dados = _numeros(valores)
    divisor = len(dados) - 1 if amostral else len(dados)
    if divisor <= 0:
        raise ValueError("Variância amostral exige pelo menos dois valores.")
    centro = media(dados)
    return sum((valor - centro) ** 2 for valor in dados) / divisor


def desvio_padrao(valores, amostral=True):
    """Calcula a raiz quadrada da variância."""
    return sqrt(variancia(valores, amostral=amostral))


def percentil(valores, p):
    """Percentil com interpolação linear, compatível com o padrão do NumPy."""
    if not 0 <= p <= 100:
        raise ValueError("O percentil deve estar entre 0 e 100.")
    dados = sorted(_numeros(valores))
    posicao = (len(dados) - 1) * p / 100
    inferior = int(posicao)
    superior = min(inferior + 1, len(dados) - 1)
    fracao = posicao - inferior
    return dados[inferior] + fracao * (dados[superior] - dados[inferior])


def quartis(valores):
    """Retorna primeiro, segundo e terceiro quartis."""
    return percentil(valores, 25), percentil(valores, 50), percentil(valores, 75)


def coeficiente_variacao(valores, amostral=True):
    """Calcula o coeficiente de variação percentual."""
    centro = media(valores)
    if centro == 0:
        raise ValueError("O coeficiente de variação é indefinido para média zero.")
    return desvio_padrao(valores, amostral=amostral) / abs(centro) * 100


def covariancia(x, y, amostral=True):
    """Calcula covariância amostral ou populacional entre pares."""
    xs, ys = _numeros(x), _numeros(y)
    if len(xs) != len(ys):
        raise ValueError("As variáveis devem ter o mesmo tamanho.")
    divisor = len(xs) - 1 if amostral else len(xs)
    if divisor <= 0:
        raise ValueError("Covariância amostral exige pelo menos dois pares.")
    mx, my = media(xs), media(ys)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / divisor


def correlacao_pearson(x, y):
    """Calcula o coeficiente de correlação linear de Pearson."""
    sx, sy = desvio_padrao(x), desvio_padrao(y)
    if sx == 0 or sy == 0:
        raise ValueError("Correlação é indefinida para variável constante.")
    return covariancia(x, y) / (sx * sy)


def regressao_linear(x, y):
    """Retorna intercepto, inclinação e R² pelos mínimos quadrados."""
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
