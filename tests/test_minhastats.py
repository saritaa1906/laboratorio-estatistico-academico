"""Validação do núcleo estatístico contra NumPy/SciPy (Etapa 2.5 do guia).

NumPy e SciPy aparecem aqui só como REFERÊNCIA — nunca dentro de minhastats.py.
Tolerância documentada: rtol=1e-9 para somas/médias, rtol=1e-6 para percentis
(interpolação pode divergir na última casa dependendo da ordem de operações).
"""
import math

import numpy as np
import pytest
from scipy import stats

import minhastats as ms

DADOS = np.random.default_rng(0).gamma(2, 9, 500).tolist()          # variável assimétrica
DADOS_SIMETRICOS = np.random.default_rng(1).normal(70, 10, 500).tolist()
X = np.random.default_rng(2).uniform(0, 10, 200).tolist()
Y = [3.44 * xi + 7.1 + ruido for xi, ruido in zip(X, np.random.default_rng(3).normal(0, 3, 200))]


# ---------- tendência central ----------
def test_media():
    assert np.isclose(ms.media(DADOS), np.mean(DADOS), rtol=1e-9)


def test_mediana_par_e_impar():
    assert np.isclose(ms.mediana(DADOS), np.median(DADOS), rtol=1e-9)
    assert ms.mediana([1, 3, 2]) == 2
    assert ms.mediana([1, 2, 3, 4]) == 2.5


def test_moda():
    resultado = ms.moda([1, 2, 2, 3, 3, 3])
    assert resultado == [3]
    # múltiplas modas devem vir como lista ordenada
    assert ms.moda([1, 1, 2, 2]) == [1, 2]


# ---------- dispersão ----------
def test_amplitude():
    assert ms.amplitude(DADOS) == max(DADOS) - min(DADOS)


def test_variancia_amostral():
    assert np.isclose(ms.variancia(DADOS, amostral=True), np.var(DADOS, ddof=1), rtol=1e-9)


def test_variancia_populacional():
    assert np.isclose(ms.variancia(DADOS, amostral=False), np.var(DADOS, ddof=0), rtol=1e-9)


def test_variancia_amostral_exige_dois_valores():
    with pytest.raises(ValueError):
        ms.variancia([5.0], amostral=True)


def test_desvio_padrao():
    assert np.isclose(ms.desvio_padrao(DADOS), np.std(DADOS, ddof=1), rtol=1e-9)


def test_percentil():
    for p in (10, 25, 50, 75, 90):
        assert np.isclose(ms.percentil(DADOS, p), np.percentile(DADOS, p), rtol=1e-6)


def test_percentil_faixa_invalida():
    with pytest.raises(ValueError):
        ms.percentil(DADOS, 150)


def test_quartis():
    q1, q2, q3 = ms.quartis(DADOS)
    ref = np.percentile(DADOS, [25, 50, 75])
    assert np.allclose([q1, q2, q3], ref, rtol=1e-6)


def test_limites_outliers_iqr():
    q1, _, q3 = ms.quartis(DADOS)
    iqr = q3 - q1
    inferior, superior = ms.limites_outliers_iqr(DADOS)
    assert np.isclose(inferior, q1 - 1.5 * iqr)
    assert np.isclose(superior, q3 + 1.5 * iqr)


def test_contar_outliers_iqr_nao_negativo():
    assert ms.contar_outliers_iqr(DADOS) >= 0
    assert ms.contar_outliers_iqr([1, 2, 3, 4, 5]) == 0  # sem outliers óbvios


def test_coeficiente_variacao():
    esperado = (np.std(DADOS, ddof=1) / np.mean(DADOS)) * 100
    assert np.isclose(ms.coeficiente_variacao(DADOS), esperado, rtol=1e-9)


def test_coeficiente_variacao_media_zero():
    with pytest.raises(ValueError):
        ms.coeficiente_variacao([-1, 0, 1])


# ---------- duas variáveis ----------
def test_covariancia():
    esperado = np.cov(X, Y, ddof=1)[0, 1]
    assert np.isclose(ms.covariancia(X, Y), esperado, rtol=1e-6)


def test_covariancia_tamanhos_diferentes():
    with pytest.raises(ValueError):
        ms.covariancia([1, 2, 3], [1, 2])


def test_correlacao_pearson():
    esperado = np.corrcoef(X, Y)[0, 1]
    assert np.isclose(ms.correlacao_pearson(X, Y), esperado, rtol=1e-6)


def test_correlacao_variavel_constante():
    with pytest.raises(ValueError):
        ms.correlacao_pearson([5, 5, 5], [1, 2, 3])


def test_regressao_linear_contra_numpy_polyfit():
    inclinacao_np, intercepto_np = np.polyfit(X, Y, 1)
    intercepto, inclinacao, r2 = ms.regressao_linear(X, Y)
    assert np.isclose(intercepto, intercepto_np, rtol=1e-6)
    assert np.isclose(inclinacao, inclinacao_np, rtol=1e-6)
    assert 0 <= r2 <= 1


def test_regressao_x_constante_falha():
    with pytest.raises(ValueError):
        ms.regressao_linear([2, 2, 2], [1, 2, 3])


def test_prever_extrapolacao():
    intercepto, inclinacao, _ = ms.regressao_linear(X, Y)
    dentro, fora = ms.prever(intercepto, inclinacao, min(X), x_min=min(X), x_max=max(X))
    assert fora is False
    _, fora2 = ms.prever(intercepto, inclinacao, max(X) + 100, x_min=min(X), x_max=max(X))
    assert fora2 is True


# ---------- utilidades de visualização ----------
def test_num_classes_sturges():
    assert ms.num_classes_sturges(1000) == math.ceil(1 + 3.322 * math.log10(1000))
    assert ms.num_classes_sturges(1) == 1


def test_densidade_normal_contra_scipy():
    mu, sigma = 70.0, 10.0
    for x in (50, 60, 70, 80, 90):
        esperado = stats.norm.pdf(x, mu, sigma)
        assert np.isclose(ms.densidade_normal(x, mu, sigma), esperado, rtol=1e-9)


def test_densidade_exponencial_contra_scipy():
    taxa = 0.1
    for x in (0, 5, 10, 20):
        esperado = stats.expon.pdf(x, scale=1 / taxa)
        assert np.isclose(ms.densidade_exponencial(x, taxa), esperado, rtol=1e-9)
    assert ms.densidade_exponencial(-1, taxa) == 0.0


def test_grade_linear_contra_numpy_linspace():
    grade = ms.grade_linear(0, 10, 50)
    esperado = np.linspace(0, 10, 50)
    assert np.allclose(grade, esperado)


def test_frequencia_relativa_acumulada():
    seq = [1, 0, 1, 1, 0]
    esperado = np.cumsum(seq) / np.arange(1, len(seq) + 1)
    assert np.allclose(ms.frequencia_relativa_acumulada(seq), esperado)


# ---------- robustez com amostra vazia ----------
@pytest.mark.parametrize("funcao", [ms.media, ms.mediana, ms.moda, ms.amplitude, ms.variancia, ms.desvio_padrao])
def test_funcoes_rejeitam_amostra_vazia(funcao):
    with pytest.raises(ValueError):
        funcao([])
