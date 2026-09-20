"""Laboratório Estatístico Interativo — interface (Streamlit)"""

import inspect

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import minhastats as ms

try:
    from data_sources import carregar_uci
except ImportError:  # sem dados reais o app NÃO inventa dados sintéticos
    carregar_uci = None

# ==========================================================================
# TRADUÇÕES (apenas apresentação)
# ==========================================================================
COLUNAS_PT = {
    "age": "Idade",
    "job": "Profissão",
    "marital": "Estado civil",
    "education": "Escolaridade",
    "default": "Possui dívida em atraso?",
    "balance": "Saldo em conta (balance)",
    "housing": "Possui financiamento imobiliário?",
    "loan": "Possui empréstimo pessoal?",
    "contact": "Canal de contato",
    "day": "Dia do mês da ligação",
    "month": "Mês da ligação",
    "duration": "Duração da ligação em segundos (duration)",
    "campaign": "Nº de contatos nesta campanha (campaign)",
    "pdays": "Dias desde o último contato (pdays)",
    "previous": "Nº de contatos anteriores (previous)",
    "poutcome": "Resultado da campanha anterior",
    "y": "Aceitou o depósito a prazo?",
}

TRADUCOES_VALORES = {
    "marital": {"married": "Casado(a)", "single": "Solteiro(a)", "divorced": "Divorciado(a)"},
    "education": {"tertiary": "Ensino superior", "secondary": "Ensino médio",
                  "primary": "Ensino fundamental", "unknown": "Não informado"},
    "job": {
        "management": "Gerência", "technician": "Técnico(a)", "entrepreneur": "Empreendedor(a)",
        "blue-collar": "Trabalho manual", "retired": "Aposentado(a)", "admin.": "Administrativo",
        "services": "Serviços", "self-employed": "Autônomo(a)", "unemployed": "Desempregado(a)",
        "housemaid": "Trabalho doméstico", "student": "Estudante", "unknown": "Não informado",
    },
    "default": {"yes": "Sim", "no": "Não"},
    "housing": {"yes": "Sim", "no": "Não"},
    "loan": {"yes": "Sim", "no": "Não"},
    "month": {
        "jan": "Janeiro", "feb": "Fevereiro", "mar": "Março", "apr": "Abril",
        "may": "Maio", "jun": "Junho", "jul": "Julho", "aug": "Agosto",
        "sep": "Setembro", "oct": "Outubro", "nov": "Novembro", "dec": "Dezembro",
    },
    "y": {"yes": "Sim, aceitou", "no": "Não aceitou"},
}


def _largura(funcao):
    """Compatibilidade entre versões do Streamlit: `use_container_width` (antigo) ou `width="stretch"` (novo)."""
    try:
        parametros = inspect.signature(funcao).parameters
    except (TypeError, ValueError):
        return {"use_container_width": True}
    return {"use_container_width": True} if "use_container_width" in parametros else {"width": "stretch"}


def exibir_grafico(fig, onde=st):
    onde.plotly_chart(fig, **_largura(onde.plotly_chart))


def exibir_tabela(dados, onde=st, **extra):
    onde.dataframe(dados, **_largura(onde.dataframe), **extra)


def nome(coluna):
    return COLUNAS_PT.get(coluna, coluna)


def fmt(valor, casas=2):
    """Formata número no padrão brasileiro (1.234,56)."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def barras_densidade(tabela, rotulo, cor=None, opacidade=0.65):
    """Barras de densidade a partir de ms.tabela_frequencias (histograma feito pelo NOSSO núcleo)."""
    largura = tabela[0]["fim"] - tabela[0]["inicio"]
    return go.Bar(
        x=[linha["ponto_medio"] for linha in tabela],
        y=[linha["densidade"] for linha in tabela],
        width=largura, name=rotulo, opacity=opacidade,
        marker_color=cor,
    )


# ==========================================================================
# PÁGINA E ESTILO
# ==========================================================================
st.set_page_config(page_title="Laboratório Estatístico Interativo", layout="wide")

st.markdown("""
<style>
    .stApp { background: #0e1117; color: #f0f0f0; font-size: 1.05rem; }
    [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
    h1, h2, h3, h4 { color: #ffffff !important; font-family: 'Segoe UI', Arial, sans-serif; }
    [data-testid="stMetric"] { background: #1f242d; border: 2px solid #30363d; padding: 16px; border-radius: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { background: #161b22; border-radius: 8px 8px 0 0; color: #ffffff; font-size: 1.05rem; padding: 12px 18px; }
</style>
""", unsafe_allow_html=True)

st.title("Laboratório Estatístico Interativo")
st.caption("Dataset: Bank Marketing (UCI) · Trabalho feito por: Sara Martins Oliveira de Sousa — RA: 72650204")
st.info(
    "**O que é este laboratório?** Uma aplicação que carrega dados reais de campanhas de telemarketing "
    "de um banco e permite explorá-los com estatística descritiva, simulação, distribuições teóricas e "
    "regressão. **Todas as medidas exibidas são calculadas pela biblioteca própria `minhastats.py`.**"
)
st.divider()


# ==========================================================================
# DADOS (Pandas só carrega e organiza)
# ==========================================================================
@st.cache_data(ttl=300, show_spinner="Carregando o dataset oficial...")
def obter_dados():
    if carregar_uci is None:
        raise RuntimeError("módulo data_sources.py não encontrado — o dataset real não pode ser carregado.")
    return carregar_uci()


try:
    df = obter_dados()
except Exception as erro:
    st.error(f"Não foi possível carregar o dataset: {erro}")
    st.stop()

numericas = [c for c in df.select_dtypes(include="number").columns if not c.lower().startswith("id")]
categoricas = list(df.select_dtypes(exclude="number").columns)
total_exibido = df.attrs.get("total_registros", len(df))

st.sidebar.markdown("### Sobre o laboratório")
st.sidebar.info("Dados reais de ligações de um banco português oferecendo depósito a prazo.")
st.sidebar.metric("Registros analisados", fmt(total_exibido, 0))
st.sidebar.metric("Variáveis numéricas", len(numericas))
st.sidebar.metric("Variáveis categóricas", len(categoricas))


def indice_padrao(opcoes, preferida, alternativa=0):
    return opcoes.index(preferida) if preferida in opcoes else min(alternativa, len(opcoes) - 1)


# ==========================================================================
# ABA 1 — DADOS
# ==========================================================================
def aba_dados():
    st.subheader("Módulo 0 — Dados reais")
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", fmt(total_exibido, 0))
    c2.metric("Variáveis numéricas", len(numericas))
    c3.metric("Variáveis categóricas", len(categoricas))

    st.markdown(
        "**Tratamento dos dados:** o dataset oficial não possui valores ausentes; a categoria "
        "`unknown` foi mantida como categoria válida; `pdays = -1` (nunca contatado) foi mantido, "
        "pois é um código do dataset e não um erro."
    )
    df_exibicao = df.copy()
    for coluna, mapa in TRADUCOES_VALORES.items():
        if coluna in df_exibicao.columns:
            df_exibicao[coluna] = df_exibicao[coluna].map(mapa).fillna(df_exibicao[coluna])
    df_exibicao = df_exibicao.rename(columns=COLUNAS_PT)
    st.markdown("##### Primeiros 100 registros")
    exibir_tabela(df_exibicao.head(100))


# ==========================================================================
# ABA 2 — ESTATÍSTICA DESCRITIVA
# ==========================================================================
def interpretar_forma(g1, media_v, mediana_v, n_out, n, cv):
    """Interpretação textual automática (assimetria, outliers, variabilidade)."""
    if abs(g1) < 0.5:
        forma = ("A distribuição é **aproximadamente simétrica** "
                 f"(assimetria g₁ = {fmt(g1)}; média ≈ mediana).")
    else:
        intensidade = "moderadamente" if abs(g1) < 1 else "fortemente"
        if g1 > 0:
            forma = (f"A distribuição é **{intensidade} assimétrica à direita** (g₁ = {fmt(g1)}): "
                     f"há uma cauda de valores altos que puxam a média ({fmt(media_v)}) para cima "
                     f"da mediana ({fmt(mediana_v)}).")
        else:
            forma = (f"A distribuição é **{intensidade} assimétrica à esquerda** (g₁ = {fmt(g1)}): "
                     f"valores baixos puxam a média ({fmt(media_v)}) para baixo da mediana ({fmt(mediana_v)}).")
    pct = n_out / n * 100
    if n_out == 0:
        outl = "Pela regra do IQR **não há outliers**."
    else:
        outl = (f"Pela regra do IQR há **{fmt(n_out, 0)} outliers ({fmt(pct)}% dos dados)**"
                + (" — uma cauda pesada, então a mediana descreve o 'típico' melhor que a média." if pct > 5 else "."))
    if cv < 15:
        var = f"A variabilidade relativa é **baixa** (CV = {fmt(cv)}%)."
    elif cv <= 30:
        var = f"A variabilidade relativa é **moderada** (CV = {fmt(cv)}%)."
    else:
        var = f"A variabilidade relativa é **alta** (CV = {fmt(cv)}%)."
    return forma, outl, var


def aba_descritiva():
    st.subheader("Módulo 2 — Estatística descritiva interativa")
    tipo = st.radio("Tipo de variável", ["Numérica", "Categórica"], horizontal=True)

    if tipo == "Numérica":
        if not numericas:
            st.warning("O dataset não possui variáveis numéricas.")
            return
        col = st.selectbox("Escolha a variável numérica:", numericas, key="desc_num", format_func=nome)
        valores = df[col].dropna().astype(float).tolist()
        n = len(valores)

        try:
            q1, q2, q3 = ms.quartis(valores)
            lim_i, lim_s = ms.limites_outliers_iqr(valores)
            n_out = ms.contar_outliers_iqr(valores)
            media_v, mediana_v = ms.media(valores), ms.mediana(valores)
            dp = ms.desvio_padrao(valores)
            cv = ms.coeficiente_variacao(valores)
            g1 = ms.coeficiente_assimetria(valores)
            tabela_freq = ms.tabela_frequencias(valores)  # k por Sturges
        except ValueError as erro:
            st.warning(f"Não foi possível resumir esta variável: {erro}")
            return

        modas = ms.moda(valores)
        if len(modas) == n:
            moda_txt = "não há moda (todos os valores distintos)"
        else:
            moda_txt = ", ".join(fmt(m) for m in modas[:3]) + (f" (+{len(modas) - 3} empatadas)" if len(modas) > 3 else "")

        medidas = pd.DataFrame({
            "Medida": [
                "n (observações)", "Média", "Mediana", "Moda", "Amplitude",
                "Variância amostral (n−1)", "Variância populacional (n)",
                "Desvio-padrão amostral", "Desvio-padrão populacional",
                "Q1 (25%)", "Q3 (75%)", "IQR (Q3 − Q1)",
                "Coeficiente de variação", "Assimetria (g₁)",
            ],
            "Valor": [
                fmt(n, 0), fmt(media_v), fmt(mediana_v), moda_txt, fmt(ms.amplitude(valores)),
                fmt(ms.variancia(valores, amostral=True)), fmt(ms.variancia(valores, amostral=False)),
                fmt(dp), fmt(ms.desvio_padrao(valores, amostral=False)),
                fmt(q1), fmt(q3), fmt(q3 - q1),
                fmt(cv) + "%", fmt(g1),
            ],
        })

        esq, dir_ = st.columns([1, 1])
        with esq:
            st.markdown("##### Medidas de tendência central e dispersão")
            exibir_tabela(medidas, hide_index=True)
        with dir_:
            st.markdown(f"##### Tabela de frequências ({len(tabela_freq)} classes — regra de Sturges)")
            tabela_df = pd.DataFrame({
                "Classe": [f"[{fmt(l['inicio'])} ; {fmt(l['fim'])}" + ("]" if i == len(tabela_freq) - 1 else ")")
                           for i, l in enumerate(tabela_freq)],
                "Frequência": [l["freq"] for l in tabela_freq],
                "Freq. relativa (%)": [round(l["freq_rel"] * 100, 2) for l in tabela_freq],
                "Freq. acumulada (%)": [round(l["freq_acum"] * 100, 2) for l in tabela_freq],
            })
            exibir_tabela(tabela_df, hide_index=True)
            st.caption(f"k = ⌈1 + 3,322·log₁₀({fmt(n, 0)})⌉ = {len(tabela_freq)} classes de mesma largura.")

        g1_col, g2_col = st.columns(2)
        with g1_col:
            largura = tabela_freq[0]["fim"] - tabela_freq[0]["inicio"]
            fig_hist = go.Figure(go.Bar(
                x=[l["ponto_medio"] for l in tabela_freq], y=[l["freq"] for l in tabela_freq],
                width=largura, name="Frequência", marker_color="#7e22ce"))
            fig_hist.add_vline(x=media_v, line_color="#ffcc00", annotation_text="média")
            fig_hist.add_vline(x=mediana_v, line_color="#28a745", line_dash="dash", annotation_text="mediana")
            fig_hist.update_layout(template="plotly_dark", title=f"Histograma — {nome(col)}",
                                   xaxis_title=nome(col), yaxis_title="Frequência")
            exibir_grafico(fig_hist, st)
        with g2_col:
            fig_box = px.box(pd.DataFrame({col: valores}), y=col, points="outliers",
                             title=f"Boxplot — {nome(col)}", labels={col: nome(col)})
            fig_box.add_hline(y=lim_s, line_dash="dot", line_color="#ff3333",
                              annotation_text=f"Q3 + 1,5·IQR = {fmt(lim_s)}")
            fig_box.add_hline(y=lim_i, line_dash="dot", line_color="#ff3333",
                              annotation_text=f"Q1 − 1,5·IQR = {fmt(lim_i)}")
            fig_box.update_layout(template="plotly_dark")
            exibir_grafico(fig_box, st)

        forma, outl, var = interpretar_forma(g1, media_v, mediana_v, n_out, n, cv)
        st.success(f"**Interpretação automática**\n\n- {forma}\n- {outl}\n- {var}")

    else:
        if not categoricas:
            st.warning("O dataset não possui variáveis categóricas.")
            return
        col = st.selectbox("Escolha a variável categórica:", categoricas, key="desc_cat", format_func=nome)
    valores_categoricos = df[col].fillna("Não informado").astype(str).tolist()

    freq = ms.frequencia_categorica(valores_categoricos)

    freq = pd.DataFrame({
        "Categoria": [linha["categoria"] for linha in freq],
        "Frequência": [linha["frequencia"] for linha in freq],
        "Freq. relativa (%)": [round(linha["freq_rel"] * 100, 2) for linha in freq],
        "Freq. acumulada (%)": [round(linha["freq_acum"] * 100, 2) for linha in freq],
    })

    if col in TRADUCOES_VALORES:
        freq["Categoria"] = freq["Categoria"].map(
            TRADUCOES_VALORES[col]
        ).fillna(freq["Categoria"])
        exibir_tabela(freq, hide_index=True)
        a, b = st.columns(2)
        fig_bar = px.bar(freq.head(20), x="Categoria", y="Frequência", title=f"Frequência — {nome(col)}")
        fig_bar.update_layout(template="plotly_dark")
        exibir_grafico(fig_bar, a)
        if len(freq) <= 10:
            fig_pie = px.pie(freq, names="Categoria", values="Frequência", title="Proporção por categoria")
            fig_pie.update_layout(template="plotly_dark")
            exibir_grafico(fig_pie, b)
        moda_cat = freq.iloc[0]
        st.success(f"**Interpretação:** a categoria mais frequente (moda) é **{moda_cat['Categoria']}**, "
                   f"com {fmt(moda_cat['Freq. relativa (%)'])}% dos registros.")


# ==========================================================================
# ABA 3 — PROBABILIDADE E SIMULAÇÃO
# ==========================================================================
@st.cache_data(show_spinner="Sorteando amostras...")
def simular_tcl(coluna, tamanho, repeticoes, semente):
    """Sorteia `repeticoes` amostras de `tamanho` e devolve a média de cada uma (ms.media)."""
    base = df[coluna].dropna().astype(float).to_numpy()
    rng = np.random.default_rng(int(semente))
    amostras = rng.choice(base, size=(repeticoes, tamanho), replace=True)  # NumPy só sorteia
    return [ms.media(linha.tolist()) for linha in amostras]


def aba_simulacoes():
    st.subheader("Módulo 3 — Probabilidade e simulação de Monte Carlo")

    # ---- (a) Lei dos Grandes Números ----
    st.markdown("### (a) Lei dos Grandes Números")
    st.write("A frequência relativa de caras oscila muito no início e converge para a probabilidade teórica.")
    c1, c2, c3, c4 = st.columns(4)
    n = c1.slider("Nº de lançamentos", 100, 10000, 2000, 100)
    repet = c2.slider("Nº de simulações (linhas)", 1, 20, 5)
    p = c3.slider("Probabilidade de cara (p)", 0.05, 0.95, 0.50, 0.05)
    semente = c4.number_input("Semente aleatória", min_value=0, value=42, step=1)
    log = st.checkbox("Eixo X em escala logarítmica", value=True)

    rng = np.random.default_rng(int(semente))
    fig = go.Figure()
    eixo_n = list(range(1, n + 1))
    finais, apos_10 = [], []
    for i in range(repet):
        lancamentos = (rng.random(n) < p).astype(int).tolist()
        freq = ms.frequencia_relativa_acumulada(lancamentos)
        finais.append(freq[-1])
        apos_10.append(freq[min(9, n - 1)])
        fig.add_scatter(x=eixo_n, y=freq, name=f"Simulação {i + 1}", opacity=0.7)
    fig.add_hline(y=p, line_dash="dash", line_color="#ffcc00", line_width=3,
                  annotation_text=f"probabilidade teórica = {fmt(p)}")
    if log:
        fig.update_xaxes(type="log")
    fig.update_layout(template="plotly_dark", xaxis_title="Nº de lançamentos",
                      yaxis_title="Frequência relativa de caras", yaxis_range=[0, 1])
    exibir_grafico(fig, st)
    dist10 = max(abs(v - p) for v in apos_10)
    distn = max(abs(v - p) for v in finais)
    st.info(f"Com 10 lançamentos, a maior distância até p foi **{fmt(dist10, 3)}**; "
            f"com {fmt(n, 0)} lançamentos caiu para **{fmt(distn, 3)}**. "
            "Cada simulação oscila diferente no início, mas todas convergem para p.")

    st.divider()

    # ---- (b) Teorema Central do Limite ----
    st.markdown("### (b) Teorema Central do Limite (com os dados do dataset)")
    st.write("Sorteamos amostras repetidas da variável escolhida e calculamos a média de cada uma "
             "com `ms.media`. Escolha uma variável **bem assimétrica** (ex.: duração da ligação ou saldo).")
    if not numericas:
        st.warning("Sem variáveis numéricas.")
        return
    col = st.selectbox("Variável:", numericas, key="tcl", format_func=nome,
                       index=indice_padrao(numericas, "duration"))
    t1, t2 = st.columns(2)
    tamanho = t1.slider("Tamanho de cada amostra (n)", 1, 200, 30)
    repeticoes = t2.slider("Nº de repetições (amostras)", 100, 5000, 1000, 100)

    base = df[col].dropna().astype(float).tolist()
    try:
        mu_pop = ms.media(base)
        sigma_pop = ms.desvio_padrao(base, amostral=False)
        g1_orig = ms.coeficiente_assimetria(base)
    except ValueError as erro:
        st.warning(f"Variável inadequada para o experimento: {erro}")
        return

    medias = simular_tcl(col, tamanho, repeticoes, semente)
    media_medias = ms.media(medias)
    desvio_medias = ms.desvio_padrao(medias, amostral=False)
    erro_padrao = sigma_pop / tamanho ** 0.5
    try:
        g1_medias = ms.coeficiente_assimetria(medias)
    except ValueError:
        g1_medias = 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Média das médias", fmt(media_medias))
    m2.metric("Desvio-padrão das médias", fmt(desvio_medias))
    m3.metric("Assimetria dos dados originais", fmt(g1_orig))
    m4.metric("Assimetria das médias", fmt(g1_medias))
    st.caption(f"Teoria: a média das médias deve ficar perto de μ = {fmt(mu_pop)} e o desvio das médias perto de "
               f"σ/√n = {fmt(erro_padrao)}.")

    corte = ms.percentil(base, 99)
    try:
        tab_orig = ms.tabela_frequencias(base, k=40, inicio=min(base), fim=corte)
        tab_med = ms.tabela_frequencias(medias, k=35)
    except ValueError as erro:
        st.warning(f"Não foi possível montar os histogramas: {erro}")
        return

    esq, dir_ = st.columns(2)
    fig_o = go.Figure(barras_densidade(tab_orig, "Dados originais", "#7e22ce"))
    fig_o.update_layout(template="plotly_dark", title=f"Dados originais (até o percentil 99) — {nome(col)}",
                        xaxis_title=nome(col), yaxis_title="Densidade")
    exibir_grafico(fig_o, esq)

    eixo = ms.grade_linear(tab_med[0]["inicio"], tab_med[-1]["fim"], 250)
    fig_m = go.Figure(barras_densidade(tab_med, f"Médias de amostras (n = {tamanho})", "#7e22ce"))
    if erro_padrao > 0:
        fig_m.add_scatter(x=eixo, y=[ms.densidade_normal(x, mu_pop, erro_padrao) for x in eixo],
                          name="Normal(μ, σ/√n)", line={"color": "#f59e0b", "width": 3})
    fig_m.update_layout(template="plotly_dark", title=f"Distribuição das {fmt(repeticoes, 0)} médias amostrais",
                        xaxis_title="Média da amostra", yaxis_title="Densidade")
    exibir_grafico(fig_m, dir_)

    st.success("**Leitura:** com n pequeno as médias ainda lembram a forma assimétrica dos dados; conforme n cresce, "
               "a assimetria das médias tende a 0 e o histograma se aproxima do sino Normal com desvio σ/√n — "
               "**é por isso que a Normal aparece em todo lugar**, mesmo quando os dados originais não são normais.")


# ==========================================================================
# ABA 4 — DISTRIBUIÇÕES TEÓRICAS
# ==========================================================================
def aba_distribuicoes():
    st.subheader("Módulo 4 — Distribuições teóricas sobre o histograma")
    if not numericas:
        st.warning("Sem variáveis numéricas.")
        return
    c1, c2 = st.columns([1, 1])
    col = c1.selectbox("Variável:", numericas, key="dist", format_func=nome,
                       index=indice_padrao(numericas, "age"))
    escolhidas = c2.multiselect("Distribuições candidatas:", ["Normal", "Exponencial", "Uniforme"],
                                default=["Normal", "Exponencial"])

    dados = df[col].dropna().astype(float).tolist()
    try:
        vmin, vmax = min(dados), max(dados)
        mu = ms.media(dados)
        sigma = ms.desvio_padrao(dados, amostral=False)
        corte = ms.percentil(dados, 99)
        g1 = ms.coeficiente_assimetria(dados)
        cv = ms.coeficiente_variacao(dados)
        tabela = ms.tabela_frequencias(dados, k=40, inicio=vmin, fim=corte)
    except ValueError as erro:
        st.warning(f"Não foi possível ajustar distribuições a esta variável: {erro}")
        return

    positivos = [v for v in dados if v > 0]
    lam = 1 / ms.media(positivos) if positivos else None
    eixo = ms.grade_linear(vmin, corte, 300)

    densidades = {}
    if "Normal" in escolhidas:
        densidades["Normal"] = (lambda x: ms.densidade_normal(x, mu, sigma),
                                f"Normal(μ={fmt(mu)}, σ={fmt(sigma)})", "#ef4444")
    if "Exponencial" in escolhidas and lam:
        densidades["Exponencial"] = (lambda x: ms.densidade_exponencial(x, lam),
                                     f"Exponencial(λ={lam:.4f})", "#22c55e")
    if "Uniforme" in escolhidas and vmax > vmin:
        densidades["Uniforme"] = (lambda x: ms.densidade_uniforme(x, vmin, vmax),
                                  f"Uniforme({fmt(vmin)}; {fmt(vmax)})", "#38bdf8")

    fig = go.Figure(barras_densidade(tabela, "Dados reais (histograma)", "#a78bfa", 0.55))
    for chave, (f, rotulo, cor) in densidades.items():
        fig.add_scatter(x=eixo, y=[f(x) for x in eixo], name=rotulo, line={"color": cor, "width": 3})
    fig.update_layout(template="plotly_dark", xaxis_title=nome(col), yaxis_title="Densidade",
                      title=f"{nome(col)} — histograma × curvas teóricas (eixo até o percentil 99)")
    exibir_grafico(fig, st)
    st.caption("Parâmetros estimados a partir dos dados: Normal (μ = média, σ = desvio populacional); "
               "Exponencial (λ = 1/média dos valores positivos); Uniforme (mín, máx). "
               "O histograma está em densidade, na mesma escala das curvas.")

    # ---- qualidade do ajuste (numérica) ----
    if densidades:
        max_dens = max(l["densidade"] for l in tabela)
        linhas = []
        for chave, (f, rotulo, _) in densidades.items():
            erros = [abs(l["densidade"] - f(l["ponto_medio"])) for l in tabela]
            linhas.append({"Distribuição": rotulo,
                           "Erro médio |histograma − curva| (% da maior barra)": round(ms.media(erros) / max_dens * 100, 2)})
        st.markdown("##### Qualidade do ajuste (quanto menor o erro, melhor)")
        exibir_tabela(pd.DataFrame(linhas), hide_index=True)

    # ---- discussão automática ----
    p1 = ms.proporcao_entre(dados, mu - sigma, mu + sigma) * 100
    p2 = ms.proporcao_entre(dados, mu - 2 * sigma, mu + 2 * sigma) * 100
    texto = [f"- **Forma dos dados:** assimetria g₁ = {fmt(g1)}, coeficiente de variação = {fmt(cv)}%, "
             f"valor mínimo = {fmt(vmin)}."]
    if "Normal" in densidades:
        boa = abs(g1) < 0.5 and abs(p1 - 68.27) < 5 and abs(p2 - 95.45) < 3
        texto.append(
            f"- **Normal:** a regra empírica prevê 68,3% dos dados em μ±1σ e 95,4% em μ±2σ; nos dados há "
            f"{fmt(p1)}% e {fmt(p2)}%. "
            + ("Como a assimetria é pequena e as proporções são próximas, **o ajuste é razoável**."
               if boa else
               "Como há assimetria/caudas diferentes do sino, **a Normal ajusta mal**"
               + (" — a variável tem cauda à direita e uma Gama/Exponencial descreveria melhor." if g1 > 0.5 else ".")))
    if "Exponencial" in densidades:
        compat = vmin >= 0 and abs(cv - 100) < 25 and g1 > 1
        texto.append(
            f"- **Exponencial:** a teórica tem CV = 100% e assimetria = 2; nos dados CV = {fmt(cv)}% e g₁ = {fmt(g1)}. "
            + ("Os indicadores são compatíveis, então **o ajuste é plausível**."
               if compat else
               "Os indicadores diferem, então **a Exponencial não descreve bem esta variável**"
               + (" (há valores negativos; ela só existe para x ≥ 0)." if vmin < 0 else ".")))
    if "Uniforme" in densidades:
        texto.append("- **Uniforme:** só ajusta bem se o histograma for aproximadamente plano entre o mínimo e o máximo; "
                     "compare visualmente com as barras.")
    st.success("**Discussão automática do ajuste**\n\n" + "\n".join(texto))


# ==========================================================================
# ABA 5 — CORRELAÇÃO E REGRESSÃO
# ==========================================================================
@st.cache_data(show_spinner="Calculando a matriz de correlação (Pearson próprio)...")
def matriz_correlacao(colunas):
    limpo = df[list(colunas)].dropna().astype(float)
    listas = {c: limpo[c].tolist() for c in colunas}
    matriz = []
    for a in colunas:
        linha = []
        for b in colunas:
            try:
                linha.append(1.0 if a == b else ms.correlacao_pearson(listas[a], listas[b]))
            except ValueError:
                linha.append(float("nan"))
        matriz.append(linha)
    return matriz


def aba_regressao():
    st.subheader("Módulo 5 — Correlação e regressão linear simples")
    if len(numericas) < 2:
        st.warning("São necessárias ao menos duas variáveis numéricas.")
        return
    a, b = st.columns(2)
    xcol = a.selectbox("Variável X (explicativa):", numericas, key="rx", format_func=nome,
                       index=indice_padrao(numericas, "age"))
    ycol = b.selectbox("Variável Y (resposta):", numericas, key="ry", format_func=nome,
                       index=indice_padrao(numericas, "balance", 1))
    if xcol == ycol:
        st.warning("Escolha variáveis diferentes: X = Y gera uma correlação trivial de 1.")
        return

    pares = df[[xcol, ycol]].dropna()
    xs, ys = pares[xcol].astype(float).tolist(), pares[ycol].astype(float).tolist()
    try:
        b0, b1, r2 = ms.regressao_linear(xs, ys)
        r = ms.correlacao_pearson(xs, ys)
    except ValueError as erro:
        st.warning(f"Não foi possível ajustar a reta: {erro}")
        return
    x_min, x_max = min(xs), max(xs)

    amostra = pares.sample(min(10000, len(pares)), random_state=42)
    fig = px.scatter(amostra, x=xcol, y=ycol, opacity=0.35,
                     title=f"{nome(xcol)} × {nome(ycol)} (gráfico com até 10.000 pontos; cálculos com todos)",
                     labels={xcol: nome(xcol), ycol: nome(ycol)})
    fig.add_scatter(x=[x_min, x_max], y=[b0 + b1 * x_min, b0 + b1 * x_max],
                    name="Reta de mínimos quadrados", line={"color": "#ff3333", "width": 3})
    fig.update_layout(template="plotly_dark")
    exibir_grafico(fig, st)

    m1, m2, m3 = st.columns(3)
    m1.metric("Correlação de Pearson (r)", fmt(r, 3))
    m2.metric("Coeficiente de determinação (R²)", fmt(r2 * 100, 1) + "%")
    m3.metric("Equação da reta", f"ŷ = {fmt(b0)} + {fmt(b1, 4)}·x")

    forca = "muito forte" if abs(r) >= 0.7 else "moderada" if abs(r) >= 0.4 else "fraca" if abs(r) >= 0.1 else "praticamente nula"
    sentido = "positiva" if r > 0 else "negativa" if r < 0 else "nula"
    mais_menos = "a mais" if b1 > 0 else "a menos"
    st.markdown("##### Interpretação dos coeficientes")
    st.info(
        f"- **Inclinação (b₁ = {fmt(b1, 4)}):** cada unidade a mais de *{nome(xcol)}* está **associada**, em média, "
        f"a {fmt(abs(b1), 4)} unidades {mais_menos} de *{nome(ycol)}*.\n"
        f"- **Intercepto (b₀ = {fmt(b0)}):** valor médio previsto de *{nome(ycol)}* quando X = 0"
        + (" (X = 0 está dentro da faixa observada)." if x_min <= 0 <= x_max
           else f" — atenção: X = 0 está **fora** da faixa observada [{fmt(x_min)} ; {fmt(x_max)}], "
                "então b₀ é só um parâmetro da reta, sem significado prático.") + "\n"
        f"- **Correlação:** {forca}, {sentido} (r = {fmt(r, 3)}).\n"
        f"- **R² = {fmt(r2 * 100, 1)}%:** a variação de X explica essa fração da variação de Y; "
        f"os {fmt((1 - r2) * 100, 1)}% restantes dependem de outros fatores."
    )
    st.warning(
        "⚠️ **Correlação não implica causalidade.** A reta descreve uma *associação* nos dados, não uma relação de causa e efeito. "
        "Exemplo deste dataset: ligações mais longas (`duration`) aparecem junto com mais aceitações, mas isso não prova que "
        "'ligar por mais tempo convence' — clientes já interessados podem simplesmente conversar mais."
    )

    st.divider()
    st.markdown("##### Predição interativa")
    entrada = st.number_input(f"Digite X = {nome(xcol)}:", value=float(ms.media(xs)), format="%.2f")
    y_prev, fora = ms.prever(b0, b1, entrada, x_min=x_min, x_max=x_max)
    st.success(f"ŷ = {fmt(b0)} + {fmt(b1, 4)} × {fmt(entrada)} = **{fmt(y_prev)}** ({nome(ycol)})")
    if fora:
        st.error(f"⚠️ **Extrapolação:** X = {fmt(entrada)} está fora da faixa observada "
                 f"[{fmt(x_min)} ; {fmt(x_max)}]. A reta só foi validada dentro desse intervalo — "
                 "a previsão não é confiável.")

    with st.expander("Ver matriz de correlação de todas as variáveis numéricas"):
        matriz = matriz_correlacao(tuple(numericas))
        fig_c = px.imshow(matriz, x=[nome(c) for c in numericas], y=[nome(c) for c in numericas],
                          zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f")
        fig_c.update_layout(template="plotly_dark", height=600)
        exibir_grafico(fig_c, st)


# ==========================================================================
# ABA 6 — DESCOBERTAS
# ==========================================================================
def aba_descobertas():
    st.subheader("Módulo 6 — As três descobertas")
    if "y" not in df.columns:
        st.warning("A coluna alvo `y` não está disponível.")
        return
    aceitaram = df[df["y"] == "yes"]
    recusaram = df[df["y"] == "no"]
    total, n_aceitou = len(df), len(aceitaram)
    taxa = n_aceitou / total * 100

    # ---- Descoberta 1 ----
    st.success(f"**1. Poucas pessoas aceitam a proposta:** apenas **{fmt(taxa, 1)}%** "
               f"({fmt(n_aceitou, 0)} de {fmt(total, 0)}) aceitaram o depósito a prazo.")
    desfechos = pd.DataFrame({"Decisão": ["Aceitou", "Não aceitou"], "Clientes": [n_aceitou, len(recusaram)]})
    fig1 = px.bar(desfechos, x="Decisão", y="Clientes", title="Aceitaram × recusaram")
    fig1.update_layout(template="plotly_dark")
    exibir_grafico(fig1, st)
    st.caption("Limite: é a taxa desta campanha (Portugal, 2008–2010); não se generaliza a outros bancos ou períodos.")

    st.divider()

    # ---- Descoberta 2 ----
    d_sim = aceitaram["duration"].astype(float).tolist()
    d_nao = recusaram["duration"].astype(float).tolist()
    med_sim, med_nao = ms.media(d_sim), ms.media(d_nao)
    mdn_sim, mdn_nao = ms.mediana(d_sim), ms.mediana(d_nao)
    st.info(f"**2. Conversas mais longas acompanham a aceitação:** quem aceitou conversou em média "
            f"**{fmt(med_sim / 60, 1)} min** ({fmt(med_sim, 0)} s; mediana {fmt(mdn_sim, 0)} s), contra "
            f"**{fmt(med_nao / 60, 1)} min** ({fmt(med_nao, 0)} s; mediana {fmt(mdn_nao, 0)} s) de quem recusou.")
    comp = pd.DataFrame({
        "Decisão": ["Aceitou", "Aceitou", "Não aceitou", "Não aceitou"],
        "Medida": ["Média", "Mediana", "Média", "Mediana"],
        "Segundos": [med_sim, mdn_sim, med_nao, mdn_nao],
    })
    fig2 = px.bar(comp, x="Decisão", y="Segundos", color="Medida", barmode="group",
                  title="Duração da ligação (segundos): média e mediana")
    fig2.update_layout(template="plotly_dark")
    exibir_grafico(fig2, st)
    st.caption("Limite: é associação, não causa — clientes já interessados tendem a conversar mais.")

    st.divider()

    # ---- Descoberta 3 ----
    if "poutcome" in df.columns:
        anterior = df[df["poutcome"] == "success"]
        if len(anterior) > 0:
            taxa_ret = len(anterior[anterior["y"] == "yes"]) / len(anterior) * 100
            st.warning(f"**3. O histórico de campanha anterior importa:** entre os {fmt(len(anterior), 0)} clientes "
                       f"que já tinham aceitado antes (`poutcome = success`), **{fmt(taxa_ret, 1)}%** aceitaram "
                       f"de novo, contra a taxa geral de **{fmt(taxa, 1)}%**.")
            comp3 = pd.DataFrame({"Grupo": ["Taxa geral", "Já aceitou antes"], "Taxa de aceitação (%)": [taxa, taxa_ret]})
            fig3 = px.bar(comp3, x="Grupo", y="Taxa de aceitação (%)", text="Taxa de aceitação (%)",
                          title="Taxa de aceitação: geral × quem já aceitou antes")
            fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig3.update_layout(template="plotly_dark", yaxis_range=[0, 80])
            exibir_grafico(fig3, st)
            st.caption("Limite: correlação não implica causalidade — quem aceita uma vez pode simplesmente confiar mais "
                       "no banco ou ter perfil financeiro diferente.")


# ==========================================================================
# ABAS
# ==========================================================================
tab_dados, tab_descritiva, tab_simulacoes, tab_distribuicoes, tab_regressao, tab_descobertas = st.tabs([
    "0. Dados", "2. Descritiva", "3. Simulação (LGN e TCL)",
    "4. Distribuições", "5. Correlação e regressão", "6. Descobertas",
])
with tab_dados:
    aba_dados()
with tab_descritiva:
    aba_descritiva()
with tab_simulacoes:
    aba_simulacoes()
with tab_distribuicoes:
    aba_distribuicoes()
with tab_regressao:
    aba_regressao()
with tab_descobertas:
    aba_descobertas()
