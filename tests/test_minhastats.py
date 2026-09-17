import numpy as np
import pytest
from statistics import multimode
from scipy import stats
import minhastats as ms

DADOS = [2, 3, 3, 5, 8, 13]

def test_medidas_centrais():
    assert ms.media(DADOS) == pytest.approx(np.mean(DADOS), rel=1e-12)
    assert ms.mediana(DADOS) == pytest.approx(np.median(DADOS), rel=1e-12)
    assert ms.moda(DADOS) == [float(v) for v in multimode(DADOS)]

def test_dispersao_amostral_e_populacional():
    assert ms.amplitude(DADOS) == np.ptp(DADOS)
    assert ms.variancia(DADOS, True) == pytest.approx(np.var(DADOS, ddof=1), rel=1e-12)
    assert ms.variancia(DADOS, False) == pytest.approx(np.var(DADOS), rel=1e-12)
    assert ms.desvio_padrao(DADOS, True) == pytest.approx(np.std(DADOS, ddof=1), rel=1e-12)
    assert ms.desvio_padrao(DADOS, False) == pytest.approx(np.std(DADOS, ddof=0), rel=1e-12)

def test_quartis_e_coeficiente_de_variacao():
    assert ms.quartis(DADOS) == pytest.approx(tuple(np.percentile(DADOS, [25, 50, 75])))
    referencia_cv = np.std(DADOS, ddof=1) / abs(np.mean(DADOS)) * 100
    assert ms.coeficiente_variacao(DADOS) == pytest.approx(referencia_cv)

@pytest.mark.parametrize("p", [0, 10, 25, 50, 75, 90, 100])
def test_percentis(p):
    assert ms.percentil(DADOS, p) == pytest.approx(np.percentile(DADOS, p), rel=1e-12)

def test_covariancia_correlacao_regressao():
    x, y = [1, 2, 3, 4, 5], [2, 5, 5, 8, 10]
    assert ms.covariancia(x, y) == pytest.approx(np.cov(x, y, ddof=1)[0, 1])
    assert ms.covariancia(x, y, amostral=False) == pytest.approx(np.cov(x, y, ddof=0)[0, 1])
    assert ms.correlacao_pearson(x, y) == pytest.approx(stats.pearsonr(x, y).statistic)
    a, b, r2 = ms.regressao_linear(x, y)
    ref = stats.linregress(x, y)
    assert a == pytest.approx(ref.intercept)
    assert b == pytest.approx(ref.slope)
    assert r2 == pytest.approx(ref.rvalue ** 2)

def test_erros_documentados():
    with pytest.raises(ValueError): ms.media([])
    with pytest.raises(ValueError): ms.variancia([1], True)
    with pytest.raises(ValueError): ms.correlacao_pearson([1, 1], [2, 3])
    with pytest.raises(ValueError): ms.percentil([1, 2], 101)
    with pytest.raises(ValueError): ms.covariancia([1, 2], [1])
    with pytest.raises(ValueError): ms.coeficiente_variacao([-1, 1])
