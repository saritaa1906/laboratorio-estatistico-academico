from math import ceil, log2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm, expon

import minhastats as ms
from data_sources import carregar_uci

st.set_page_config(page_title="Laboratório Estatístico", page_icon="📊", layout="wide")
st.markdown("""
<style>
.stApp { background: #0b0b0b; color: #f7f7f7; }
[data-testid="stSidebar"] { background: #050505; border-right: 1px solid #D4AF37; }
h1, h2, h3 { color: #D4AF37 !important; }
[data-testid="stMetric"] { background:#151515; border:1px solid #6f5b1c; padding:16px; border-radius:10px; }
.stButton > button { background:#D4AF37; color:#050505; border:0; font-weight:700; }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { background:#151515; border-radius:8px 8px 0 0; }
</style>
""", unsafe_allow_html=True)
st.title("📊 Laboratório Estatístico")
st.caption("Sara Martins Oliveira de Sousa · RA 72650204")

@st.cache_data(ttl=300, show_spinner="Atualizando os dados...")
def obter_dados():
    return carregar_uci()

st.sidebar.markdown("### Dataset público UCI")
if st.sidebar.button("Atualizar agora"):
    st.cache_data.clear(); st.rerun()

try:
    df = obter_dados()
except Exception as erro:
    st.error(f"Não foi possível carregar os dados: {erro}"); st.stop()

numericas = [c for c in df.select_dtypes(include="number").columns if not c.lower().startswith("id")]
categoricas = list(df.select_dtypes(exclude="number").columns)
total_exibido = df.attrs.get("total_registros", len(df))
st.sidebar.metric("Registros", f"{total_exibido:,}".replace(",", "."))
st.sidebar.caption("Atualização automática a cada 5 minutos; botão para atualização imediata.")

inicio, descritiva, simulacoes, distribuicoes, regressao, descobertas = st.tabs([
    "Dados", "Descritiva", "Simulações", "Distribuições", "Regressão", "Descobertas"
])

with inicio:
    st.subheader("Visão geral")
    a, b, c = st.columns(3)
    a.metric("Registros", total_exibido); b.metric("Variáveis numéricas", len(numericas)); c.metric("Variáveis categóricas", len(categoricas))
    st.dataframe(df.head(100), width="stretch")
    st.info("A análise utiliza exclusivamente o dataset público Bank Marketing da UCI.")

with descritiva:
    tipo = st.radio("Tipo de variável", ["Numérica", "Categórica"], horizontal=True)
    if tipo == "Numérica" and numericas:
        col = st.selectbox("Variável", numericas, key="desc_num")
        valores = df[col].dropna().astype(float).tolist()
        q1, q2, q3 = ms.quartis(valores); iqr = q3-q1
        limite_i, limite_s = q1-1.5*iqr, q3+1.5*iqr
        modas = ms.moda(valores)
        tabela = pd.DataFrame({"Medida":["Média","Mediana","Moda","Amplitude","Variância amostral","Desvio-padrão amostral","Q1","Q3","CV (%)"],
          "Valor":[ms.media(valores),ms.mediana(valores),modas[0],ms.amplitude(valores),ms.variancia(valores),ms.desvio_padrao(valores),q1,q3,ms.coeficiente_variacao(valores)]})
        amostra_grafico = df[[col]].dropna().sample(min(50000, len(valores)), random_state=42)
        l, r = st.columns(2); l.dataframe(tabela, hide_index=True, width="stretch")
        l.plotly_chart(px.histogram(amostra_grafico, x=col, nbins=30, title="Histograma"), width="stretch")
        r.plotly_chart(px.box(amostra_grafico, y=col, title="Boxplot"), width="stretch")
        outliers = sum(v < limite_i or v > limite_s for v in valores)
        assimetria = "à direita" if ms.media(valores) > ms.mediana(valores) else "à esquerda" if ms.media(valores) < ms.mediana(valores) else "aproximadamente simétrica"
        st.write(f"Foram detectados **{outliers} outliers** pela regra do IQR. A relação entre média e mediana sugere distribuição **{assimetria}**.")
        numero_classes = max(1, ceil(1 + log2(len(valores))))
        classes = pd.cut(pd.Series(valores), bins=numero_classes).value_counts(sort=False)
        st.dataframe(pd.DataFrame({"Classe":classes.index.astype(str),"Frequência":classes.values,"Frequência relativa (%)":classes.values/len(valores)*100}), hide_index=True)
    elif categoricas:
        col = st.selectbox("Variável", categoricas, key="desc_cat")
        freq = df[col].fillna("Ausente").astype(str).value_counts().reset_index()
        freq.columns = ["Categoria", "Frequência"]
        freq["Frequência relativa (%)"] = freq["Frequência"] / freq["Frequência"].sum() * 100
        st.dataframe(freq, hide_index=True)
        grafico_cat = freq.head(20)
        a, b = st.columns(2)
        a.plotly_chart(px.bar(grafico_cat, x="Categoria", y="Frequência", title="Frequência por categoria"), width="stretch")
        if len(freq) <= 10:
            b.plotly_chart(px.pie(freq, names="Categoria", values="Frequência", title="Participação relativa"), width="stretch")
        else:
            b.info("O gráfico de pizza é omitido quando há mais de 10 categorias para preservar a legibilidade.")

with simulacoes:
    st.subheader("Lei dos Grandes Números")
    n = st.slider("Número de lançamentos", 100, 10000, 2000, 100)
    repeticoes_lgn = st.slider("Número de experimentos", 1, 20, 5)
    semente = st.number_input("Semente aleatória", min_value=0, value=42, step=1)
    rng = np.random.default_rng(int(semente)); fig = go.Figure()
    for indice in range(repeticoes_lgn):
        moedas = rng.integers(0, 2, n)
        acumulada = np.cumsum(moedas) / np.arange(1, n+1)
        fig.add_scatter(y=acumulada, name=f"Experimento {indice + 1}", opacity=.65)
    fig.add_scatter(y=[.5]*n, name="Probabilidade teórica", line={"color":"#D4AF37", "width":3})
    st.plotly_chart(fig, width="stretch")
    st.write("À medida que o número de lançamentos aumenta, as frequências relativas tendem a se aproximar da probabilidade teórica de 0,5.")
    st.subheader("Teorema Central do Limite")
    col = st.selectbox("Variável do dataset", numericas, key="tcl")
    tamanho = st.slider("Tamanho da amostra", 2, 200, 30); repeticoes = st.slider("Número de amostras", 100, 5000, 1000, 100)
    base = df[col].dropna().astype(float).to_numpy(); medias=[]
    for _ in range(repeticoes): medias.append(ms.media(rng.choice(base, tamanho, replace=True).tolist()))
    media_medias = ms.media(medias); desvio_medias = ms.desvio_padrao(medias, amostral=False)
    eixo = np.linspace(min(medias), max(medias), 250)
    fig_tcl = go.Figure()
    fig_tcl.add_histogram(x=medias, nbinsx=35, histnorm="probability density", name="Médias amostrais")
    fig_tcl.add_scatter(x=eixo, y=norm.pdf(eixo, media_medias, desvio_medias), name="Normal de referência", line={"color":"#D4AF37"})
    st.plotly_chart(fig_tcl, width="stretch")
    st.write(f"Média original: **{ms.media(base):.3f}** · Média das médias: **{media_medias:.3f}** · Desvio das médias: **{desvio_medias:.3f}**")
    st.write("Ao aumentar o tamanho da amostra, a distribuição das médias tende a ficar mais concentrada e próxima da forma Normal prevista pelo TCL.")

with distribuicoes:
    col = st.selectbox("Variável contínua", numericas, key="dist")
    dados = df[col].dropna().astype(float).to_numpy(); positivos = dados[dados >= 0]
    x = np.linspace(float(np.min(dados)), float(np.percentile(dados, 99)), 300)
    mu, sigma = ms.media(dados), ms.desvio_padrao(dados, amostral=False)
    dados_grafico = np.random.default_rng(42).choice(dados, min(50000, len(dados)), replace=False)
    fig = go.Figure(); fig.add_histogram(x=dados_grafico, histnorm="probability density", nbinsx=40, name="Dados", opacity=.55)
    fig.add_scatter(x=x, y=norm.pdf(x, mu, sigma), name=f"Normal (μ={mu:.2f}, σ={sigma:.2f})")
    if len(positivos):
        escala = ms.media(positivos); fig.add_scatter(x=x[x>=0], y=expon.pdf(x[x>=0], scale=escala), name=f"Exponencial (escala={escala:.2f})")
    st.plotly_chart(fig, width="stretch")
    ordenados = np.sort(dados_grafico)
    ecdf_superior = np.arange(1, len(ordenados) + 1) / len(ordenados)
    ecdf_inferior = np.arange(0, len(ordenados)) / len(ordenados)
    cdf_normal = norm.cdf(ordenados, mu, sigma)
    d_normal = max(np.max(ecdf_superior - cdf_normal), np.max(cdf_normal - ecdf_inferior))
    texto_ajuste = f"Distância KS descritiva da Normal: **{d_normal:.3f}**."
    if len(positivos) == len(dados):
        cdf_exp = expon.cdf(ordenados, scale=ms.media(positivos))
        d_exp = max(np.max(ecdf_superior - cdf_exp), np.max(cdf_exp - ecdf_inferior))
        texto_ajuste += f" Distância da Exponencial: **{d_exp:.3f}**. Neste comparativo, a menor distância indica maior proximidade empírica."
    st.write(texto_ajuste)
    st.caption("As distâncias são descritivas porque os parâmetros foram estimados na própria amostra; não constituem, isoladamente, prova de ajuste perfeito.")

with regressao:
    xcol = st.selectbox("Variável X", numericas, key="rx"); ycol = st.selectbox("Variável Y", numericas, index=min(1,len(numericas)-1), key="ry")
    pares = df[[xcol,ycol]].dropna(); xs, ys = pares[xcol].astype(float).tolist(), pares[ycol].astype(float).tolist()
    try:
        intercepto, inclinacao, r2 = ms.regressao_linear(xs,ys); correlacao=ms.correlacao_pearson(xs,ys)
        pares_grafico = pares.sample(min(10000, len(pares)), random_state=42)
        fig=px.scatter(pares_grafico,x=xcol,y=ycol,opacity=.35); linha_x=[min(xs),max(xs)]; fig.add_scatter(x=linha_x,y=[intercepto+inclinacao*v for v in linha_x],name="Reta")
        st.plotly_chart(fig,width="stretch"); st.write(f"**ŷ = {intercepto:.4f} + {inclinacao:.4f}x** · Pearson: **{correlacao:.4f}** · R²: **{r2:.4f}**")
        intensidade = "forte" if abs(correlacao) >= .7 else "moderada" if abs(correlacao) >= .4 else "fraca"
        sentido = "positiva" if inclinacao > 0 else "negativa" if inclinacao < 0 else "nula"
        st.write(f"A inclinação indica que, para cada unidade adicional de **{xcol}**, a estimativa de **{ycol}** varia {inclinacao:.4f} unidade(s). A associação linear é **{intensidade} e {sentido}**; o modelo explica aproximadamente **{r2 * 100:.1f}%** da variação observada em Y.")
        entrada=st.number_input("Digite X para prever Y",value=float(ms.media(xs))); st.metric("Predição de Ŷ",f"{intercepto+inclinacao*entrada:.4f}")
        st.warning("Correlação não implica causalidade. A predição é uma estimativa descritiva dentro do contexto observado.")
    except ValueError as erro: st.warning(str(erro))

with descobertas:
    st.subheader("Três descobertas do relatório")
    positivos = df[df["y"] == "yes"]
    negativos = df[df["y"] == "no"]
    taxa = len(positivos) / len(df) * 100
    st.markdown(f"**1. Apenas {taxa:.2f}% das campanhas terminaram em adesão.** O desfecho é desequilibrado, então a quantidade absoluta de recusas deve ser interpretada junto com a taxa.")
    desfechos = df["y"].value_counts().rename_axis("adesao").reset_index(name="registros")
    st.plotly_chart(px.bar(desfechos, x="adesao", y="registros", color_discrete_sequence=["#D4AF37"]), width="stretch")

    duracao_sim = ms.media(positivos["duration"])
    duracao_nao = ms.media(negativos["duration"])
    st.markdown(f"**2. Ligações com adesão duraram, em média, {duracao_sim:.1f} segundos; sem adesão, {duracao_nao:.1f} segundos.** A associação não prova que prolongar artificialmente uma chamada causará adesão.")
    medias_duracao = pd.DataFrame({"adesao": ["yes", "no"], "duracao_media": [duracao_sim, duracao_nao]})
    st.plotly_chart(px.bar(medias_duracao, x="adesao", y="duracao_media", color_discrete_sequence=["#D4AF37"]), width="stretch")

    canais = []
    for canal, grupo in df.groupby("contact"):
        canais.append({"canal": canal, "taxa_adesao": sum(grupo["y"] == "yes") / len(grupo) * 100})
    taxas_canal = pd.DataFrame(canais).sort_values("taxa_adesao", ascending=False)
    melhor = taxas_canal.iloc[0]
    pior = taxas_canal.iloc[-1]
    st.markdown(f"**3. O canal {melhor['canal']} apresentou {melhor['taxa_adesao']:.2f}% de adesão, contra {pior['taxa_adesao']:.2f}% em {pior['canal']}.** A diferença pode refletir seleção de público e disponibilidade de contato, não apenas o canal.")
    st.plotly_chart(px.bar(taxas_canal, x="canal", y="taxa_adesao", color_discrete_sequence=["#D4AF37"]), width="stretch")

