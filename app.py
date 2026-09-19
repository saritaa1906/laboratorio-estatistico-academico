from math import ceil, log2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import expon, mode, norm

# ==========================================
# DICIONÁRIOS DE TRADUÇÃO PARA PORTUGUÊS
# ==========================================
COLUNAS_PT = {
    "age": "Idade da Pessoa",
    "job": "Profissão / Trabalho",
    "marital": "Estado Civil",
    "education": "Nível de Escolaridade",
    "default": "Tem Dívida Não Paga?",
    "balance": "Dinheiro na Conta (Saldo)",
    "housing": "Tem Empréstimo de Casa?",
    "loan": "Tem Empréstimo Pessoal?",
    "contact": "Como Foi o Contato",
    "day": "Dia da Ligação",
    "month": "Mês da Ligação",
    "duration": "Duração da Conversa (em segundos)",
    "campaign": "Quantas Vezes Ligamos",
    "pdays": "Dias desde a Última Ligação",
    "previous": "Contatos Feitos Antes",
    "poutcome": "Resultado do Contato Anterior",
    "y": "Aceitou a Proposta do Banco?"
}

TRADUCOES_VALORES = {
    "marital": {"married": "Casado(a)", "single": "Solteiro(a)", "divorced": "Divorciado(a)"},
    "education": {"tertiary": "Faculdade / Ensino Superior", "secondary": "Ensino Médio", "primary": "Escola Primária", "unknown": "Não Informado"},
    "job": {
        "management": "Gerente / Chefe", "technician": "Técnico(a)", "entrepreneur": "Dono de Negócio",
        "blue-collar": "Trabalhador Manual", "retired": "Aposentado(a)", "admin.": "Escritório / Administrativo",
        "services": "Prestador de Serviços", "self-employed": "Trabalhador Autônomo", "unemployed": "Desempregado(a)",
        "housemaid": "Trabalho Doméstico", "student": "Estudante", "unknown": "Não Informado"
    },
    "default": {"yes": "Sim", "no": "Não"},
    "housing": {"yes": "Sim", "no": "Não"},
    "loan": {"yes": "Sim", "no": "Não"},
    "month": {
        "jan": "Janeiro", "feb": "Fevereiro", "mar": "Março", "apr": "Abril",
        "may": "Maio", "jun": "Junho", "jul": "Julho", "aug": "Agosto",
        "sep": "Setembro", "oct": "Outubro", "nov": "Novembro", "dec": "Dezembro"
    },
    "y": {"yes": "Sim, Aceitou", "no": "Não Aceitou"}
}

# ==========================================
# CARREGAMENTO E MÓDULOS AUXILIARES
# ==========================================
def carregar_dados_fallback():
    url = "https://raw.githubusercontent.com/raghavan-v/bank-marketing/master/bank.csv"
    try:
        return pd.read_csv(url, sep=";")
    except Exception:
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({
            "age": np.random.randint(18, 70, size=n),
            "job": np.random.choice(["management", "technician", "blue-collar", "admin.", "services"], size=n),
            "marital": np.random.choice(["married", "single", "divorced"], size=n),
            "education": np.random.choice(["tertiary", "secondary", "primary"], size=n),
            "default": np.random.choice(["no", "yes"], size=n, p=[0.95, 0.05]),
            "balance": np.random.randint(-500, 15000, size=n),
            "housing": np.random.choice(["yes", "no"], size=n),
            "loan": np.random.choice(["no", "yes"], size=n, p=[0.8, 0.2]),
            "contact": np.random.choice(["cellular", "telephone"], size=n),
            "day": np.random.randint(1, 31, size=n),
            "month": np.random.choice(["may", "jul", "aug", "jun", "nov", "apr"], size=n),
            "duration": np.random.randint(10, 1000, size=n),
            "campaign": np.random.randint(1, 10, size=n),
            "pdays": np.random.choice([-1, 99, 180], size=n),
            "previous": np.random.randint(0, 5, size=n),
            "poutcome": np.random.choice(["unknown", "failure", "success"], size=n),
            "y": np.random.choice(["no", "yes"], size=n, p=[0.85, 0.15]),
        })

try:
    from data_sources import carregar_uci
except ImportError:
    carregar_uci = carregar_dados_fallback

class StatsEngine:
    @staticmethod
    def media(data):
        return float(np.mean(data))
    
    @staticmethod
    def mediana(data):
        return float(np.median(data))
    
    @staticmethod
    def moda(data):
        res = mode(data, keepdims=False)
        return [float(res.mode)] if hasattr(res, 'mode') else [float(res[0])]
    
    @staticmethod
    def amplitude(data):
        return float(np.ptp(data))
    
    @staticmethod
    def variancia(data, amostral=True):
        return float(np.var(data, ddof=1 if amostral else 0))
    
    @staticmethod
    def desvio_padrao(data, amostral=True):
        return float(np.std(data, ddof=1 if amostral else 0))
    
    @staticmethod
    def quartis(data):
        q1, q2, q3 = np.percentile(data, [25, 50, 75])
        return float(q1), float(q2), float(q3)
    
    @staticmethod
    def coeficiente_variacao(data):
        m = np.mean(data)
        return float((np.std(data, ddof=1) / abs(m)) * 100) if m != 0 else 0.0
    
    @staticmethod
    def regressao_linear(xs, ys):
        b, a = np.polyfit(xs, ys, 1)
        corr = np.corrcoef(xs, ys)[0, 1]
        return float(a), float(b), float(corr ** 2)
    
    @staticmethod
    def correlacao_pearson(xs, ys):
        return float(np.corrcoef(xs, ys)[0, 1])

ms = StatsEngine

# ==========================================
# CONFIGURAÇÃO DE PÁGINA E ESTILO
# ==========================================
st.set_page_config(page_title="Painel Fácil de Análise de Dados", layout="wide")

st.markdown("""
<style>
    .stApp { background: #0e1117; color: #f0f0f0; font-size: 1.1rem; }
    [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
    h1, h2, h3, h4 { color: #ffffff !important; font-family: 'Segoe UI', Arial, sans-serif; }
    [data-testid="stMetric"] { background: #1f242d; border: 2px solid #30363d; padding: 18px; border-radius: 10px; }
    .stButton > button { background: #238636; color: #ffffff; border: 0; font-weight: bold; border-radius: 8px; font-size: 1.1rem; padding: 10px 20px; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { background: #161b22; border-radius: 8px 8px 0 0; color: #ffffff; font-size: 1.1rem; padding: 12px 18px; }
</style>
""", unsafe_allow_html=True)

st.title("Painel de Pesquisa do Banco: Entendendo os Clientes")
st.caption("Trabalho feito por: Sara Martins Oliveira de Sousa — Registro de Aluno (RA): 72650204")

st.info("""
**O que é este site?**  
Este site ajuda você a entender como o banco conversa com as pessoas pelo telefone para oferecer um serviço de guardar dinheiro. 
Aqui você pode ver o perfil das pessoas, quanto tempo conversaram e se aceitaram a proposta!
""")
st.divider()

# ==========================================
# CARREGAMENTO DOS DADOS
# ==========================================
@st.cache_data(ttl=300, show_spinner="Buscando as informações...")
def obter_dados():
    return carregar_uci()

st.sidebar.markdown("### Menu de Ajuda")
st.sidebar.info("Este painel mostra dados reais de ligaçõesfeitas por um banco para oferecer investimentos.")
st.sidebar.markdown("---")

try:
    df = obter_dados()
except Exception as erro:
    st.error(f"Não foi possível carregar as informações: {erro}")
    st.stop()

numericas = [c for c in df.select_dtypes(include="number").columns if not c.lower().startswith("id")]
categoricas = list(df.select_dtypes(exclude="number").columns)
total_exibido = df.attrs.get("total_registros", len(df))

st.sidebar.metric("Número Total de Pessoas Analisadas", f"{total_exibido:,}".replace(",", "."))

# ==========================================
# ESTRUTURA DE ABAS SIMPLIFICADAS
# ==========================================
tab_dados, tab_descritiva, tab_simulacoes, tab_distribuicoes, tab_regressao, tab_descobertas = st.tabs([
    "1. Ver os Dados",
    "2. Resumo das Pessoas",
    "3. Jogos de Probabilidade",
    "4. Formato das Respostas",
    "5. Comparar Duas Informações",
    "6. Conclusões Principais"
])

# --- ABA 1: VER OS DADOS ---
with tab_dados:
    st.subheader("1. Conhecendo a Lista de Clientes")
    st.write("Abaixo você pode ver os números gerais e uma parte da tabela com as informações dos clientes.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Pessoas", total_exibido)
    col2.metric("Perguntas de Números (ex: Idade, Saldo)", len(numericas))
    col3.metric("Perguntas de Texto (ex: Trabalho, Estado Civil)", len(categoricas))

    df_exibicao = df.copy()
    for coluna, mapa in TRADUCOES_VALORES.items():
        if coluna in df_exibicao.columns:
            df_exibicao[coluna] = df_exibicao[coluna].map(mapa).fillna(df_exibicao[coluna])
            
    df_exibicao = df_exibicao.rename(columns=COLUNAS_PT)

    st.markdown("##### Lista com as primeiras 100 pessoas pesquisadas:")
    st.dataframe(df_exibicao.head(100), use_container_width=True)

# --- ABA 2: RESUMO DAS PESSOAS ---
with tab_descritiva:
    st.subheader("2. Resumo Fácil de uma Informação")
    st.write("Escolha abaixo o tipo de informação que você deseja entender melhor:")
    
    tipo = st.radio("O que você deseja analisar?", ["Informações de Números (ex: Idade, Saldo)", "Informações de Texto (ex: Trabalho, Estado Civil)"], horizontal=True)
    
    if tipo == "Informações de Números (ex: Idade, Saldo)" and numericas:
        col = st.selectbox("Escolha qual assunto quer ver:", numericas, key="desc_num", format_func=lambda x: COLUNAS_PT.get(x, x))
        valores = df[col].dropna().astype(float).tolist()
        
        q1, q2, q3 = ms.quartis(valores)
        iqr = q3 - q1
        limite_i, limite_s = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        modas = ms.moda(valores)
        
        tabela = pd.DataFrame({
            "Pergunta": [
                "Valor Médio (Média)",
                "Valor do Meio (Mediana)",
                "Valor que Mais Acontece (Moda)",
                "Diferença entre o Maior e o Menor Valor",
                "Variabilidade (Variância)",
                "Afastamento Padrão da Média (Desvio-Padrão)",
                "25% das pessoas estão abaixo de",
                "75% das pessoas estão abaixo de",
                "Porcentagem de Variação"
            ],
            "Resultado": [
                f"{ms.media(valores):.2f}",
                f"{ms.mediana(valores):.2f}",
                f"{modas[0]:.2f}",
                f"{ms.amplitude(valores):.2f}",
                f"{ms.variancia(valores):.2f}",
                f"{ms.desvio_padrao(valores):.2f}",
                f"{q1:.2f}",
                f"{q3:.2f}",
                f"{ms.coeficiente_variacao(valores):.2f}%"
            ]
        })
        
        amostra_grafico = df[[col]].dropna().sample(min(50000, len(valores)), random_state=42)
        l, r = st.columns(2)
        
        with l:
            st.markdown("##### Resumo Numérico Explicado")
            st.dataframe(tabela, hide_index=True, use_container_width=True)
            
            fig_hist = px.histogram(amostra_grafico, x=col, nbins=30, title=f"Gráfico de Barras de {COLUNAS_PT.get(col, col)}", labels={col: COLUNAS_PT.get(col, col), "count": "Quantidade de Pessoas"})
            fig_hist.update_layout(template="plotly_dark")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with r:
            fig_box = px.box(amostra_grafico, y=col, title=f"Caixa de Distribuição de {COLUNAS_PT.get(col, col)}", labels={col: COLUNAS_PT.get(col, col)})
            fig_box.update_layout(template="plotly_dark")
            st.plotly_chart(fig_box, use_container_width=True)
            
        outliers = sum(v < limite_i or v > limite_s for v in valores)
        media_v, mediana_v = ms.media(valores), ms.mediana(valores)
        assimetria = "mais concentrada nos números menores" if media_v > mediana_v else "mais concentrada nos números maiores" if media_v < mediana_v else "bem equilibrada dos dois lados"
        
        st.success(f"**O que este resultado significa?** Encontramos **{outliers}** valores muito fora do normal. No geral, a maioria das pessoas está **{assimetria}**.")
        
    elif categoricas:
        col = st.selectbox("Escolha a categoria que quer ver:", categoricas, key="desc_cat", format_func=lambda x: COLUNAS_PT.get(x, x))
        freq = df[col].fillna("Não Informado").astype(str).value_counts().reset_index()
        freq.columns = ["Opção", "Quantidade de Pessoas"]
        
        # Traduz valores de texto para o português simples
        if col in TRADUCOES_VALORES:
            freq["Opção"] = freq["Opção"].map(TRADUCOES_VALORES[col]).fillna(freq["Opção"])
            
        freq["Porcentagem do Total (%)"] = (freq["Quantidade de Pessoas"] / freq["Quantidade de Pessoas"].sum() * 100).round(2)
        
        st.dataframe(freq, hide_index=True, use_container_width=True)
        
        grafico_cat = freq.head(20)
        a, b = st.columns(2)
        
        fig_bar = px.bar(grafico_cat, x="Opção", y="Quantidade de Pessoas", title="Quantidade por Grupo")
        fig_bar.update_layout(template="plotly_dark")
        a.plotly_chart(fig_bar, use_container_width=True)
        
        if len(freq) <= 10:
            fig_pie = px.pie(freq, names="Opção", values="Quantidade de Pessoas", title="Divisão em Fatias (Porcentagem)")
            fig_pie.update_layout(template="plotly_dark")
            b.plotly_chart(fig_pie, use_container_width=True)

# --- ABA 3: SIMULAÇÕES ---
with tab_simulacoes:
    st.subheader("3. Experimentos de Sortear Moedas e Grupos")
    st.write("Esta página mostra como as coisas ficam mais previsíveis quando fazemos um teste muitas e muitas vezes!")
    
    st.markdown("### 1. Teste da Moeda (Regra dos Grandes Números)")
    st.write("Se jogarmos uma moeda para o alto, a chance de dar Cara é de 50%. Quanto mais vezes jogamos, mais perto de 50% ficamos!")
    
    n = st.slider("Quantas vezes vamos jogar a moeda?", 100, 10000, 2000, 100)
    repeticoes_lgn = st.slider("Quantas pessoas vão fazer este teste?", 1, 20, 5)
    semente = st.number_input("Número de sorte inicial (Semente):", min_value=0, value=42, step=1)
    
    rng = np.random.default_rng(int(semente))
    fig = go.Figure()
    for indice in range(repeticoes_lgn):
        moedas = rng.integers(0, 2, n)
        acumulada = np.cumsum(moedas) / np.arange(1, n + 1)
        fig.add_scatter(y=acumulada, name=f"Pessoa {indice + 1}", opacity=0.65)
    fig.add_scatter(y=[0.5] * n, name="Linha Média Esperada (50%)", line={"color": "#ffcc00", "width": 3, "dash": "dash"})
    fig.update_layout(template="plotly_dark", xaxis_title="Quantidade de Jogadas de Moeda", yaxis_title="Porcentagem de Caras")
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.markdown("### 2. Formato de Sino (Teorema Central do Limite)")
    st.write("Quando pegamos pequenos grupos de pessoas e calculamos a média de cada grupo, o resultado sempre forma um desenho parecendo um sino!")
    
    col = st.selectbox("Escolha uma informação para fazer o teste dos grupos:", numericas, key="tcl", format_func=lambda x: COLUNAS_PT.get(x, x))
    tamanho = st.slider("Tamanho de cada grupo de pessoas:", 2, 200, 30)
    repeticoes = st.slider("Quantos grupos vamos formar?", 100, 5000, 1000, 100)
    
    base = df[col].dropna().astype(float).to_numpy()
    medias = []
    for _ in range(repeticoes):
        medias.append(ms.media(rng.choice(base, tamanho, replace=True).tolist()))
        
    media_medias = ms.media(medias)
    desvio_medias = ms.desvio_padrao(medias, amostral=False)
    eixo = np.linspace(min(medias), max(medias), 250)
    
    fig_tcl = go.Figure()
    fig_tcl.add_histogram(x=medias, nbinsx=35, histnorm="probability density", name="Médias dos Grupos")
    fig_tcl.add_scatter(x=eixo, y=norm.pdf(eixo, media_medias, desvio_medias), name="Desenho do Sino Perfeito", line={"color": "#28a745", "width": 3})
    fig_tcl.update_layout(template="plotly_dark", xaxis_title="Média Calculada nos Grupos", yaxis_title="Chance de Acontecer")
    st.plotly_chart(fig_tcl, use_container_width=True)

# --- ABA 4: DISTRIBUIÇÕES ---
with tab_distribuicoes:
    st.subheader("4. Comparando com Moldes Matemáticos")
    st.write("Veja se os dados reais das pessoas se encaixam em formatos conhecidos de desenho (curvas matemáticas).")
    
    col = st.selectbox("Escolha o que deseja analisar:", numericas, key="dist", format_func=lambda x: COLUNAS_PT.get(x, x))
    dados = df[col].dropna().astype(float).to_numpy()
    positivos = dados[dados >= 0]
    
    x = np.linspace(float(np.min(dados)), float(np.percentile(dados, 99)), 300)
    mu, sigma = ms.media(dados), ms.desvio_padrao(dados, amostral=False)
    dados_grafico = np.random.default_rng(42).choice(dados, min(50000, len(dados)), replace=False)
    
    fig = go.Figure()
    fig.add_histogram(x=dados_grafico, histnorm="probability density", nbinsx=40, name="Dados Reais das Pessoas", opacity=0.55)
    fig.add_scatter(x=x, y=norm.pdf(x, mu, sigma), name=f"Molde em Forma de Sino (Normal)")
    
    if len(positivos):
        escala = ms.media(positivos)
        fig.add_scatter(x=x[x >= 0], y=expon.pdf(x[x >= 0], scale=escala), name=f"Molde em Queda Rápida (Exponencial)")
        
    fig.update_layout(template="plotly_dark", xaxis_title=COLUNAS_PT.get(col, col), yaxis_title="Quantidade Proporcional")
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 5: REGRESSÃO ---
with tab_regressao:
    st.subheader("5. Comparando Duas Informações ao Mesmo Tempo")
    st.write("Escolha duas coisas para ver se quando uma aumenta, a outra também muda!")
    
    col_a, col_b = st.columns(2)
    with col_a:
        xcol = st.selectbox("Primeira Informação (Causa / Eixo X):", numericas, key="rx", format_func=lambda x: COLUNAS_PT.get(x, x))
    with col_b:
        y_idx = min(1, len(numericas) - 1)
        ycol = st.selectbox("Segunda Informação (Efeito / Eixo Y):", numericas, index=y_idx, key="ry", format_func=lambda x: COLUNAS_PT.get(x, x))
    
    pares = df[[xcol, ycol]].dropna()
    xs, ys = pares[xcol].astype(float).tolist(), pares[ycol].astype(float).tolist()
    
    try:
        intercepto, inclinacao, r2 = ms.regressao_linear(xs, ys)
        correlacao = ms.correlacao_pearson(xs, ys)
        pares_grafico = pares.sample(min(10000, len(pares)), random_state=42)
        
        fig = px.scatter(
            pares_grafico,
            x=xcol,
            y=ycol,
            opacity=0.35,
            title=f"Comparando {COLUNAS_PT.get(xcol, xcol)} com {COLUNAS_PT.get(ycol, ycol)}",
            labels={xcol: COLUNAS_PT.get(xcol, xcol), ycol: COLUNAS_PT.get(ycol, ycol)}
        )
        linha_x = [min(xs), max(xs)]
        fig.add_scatter(x=linha_x, y=[intercepto + inclinacao * v for v in linha_x], name="Linha de Tendência Guiada", line=dict(color="#ff3333", width=3))
        fig.update_layout(template="plotly_dark")
        
        st.plotly_chart(fig, use_container_width=True)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Força da Ligação entre as duas", f"{correlacao:.2f}")
        m2.metric("Porcentagem de Certeza da Regra", f"{r2 * 100:.1f}%")
        m3.metric("Regra Matemática Simples", f"Y = {intercepto:.1f} + ({inclinacao:.3f} × X)")
        
        intensidade = "muito forte" if abs(correlacao) >= 0.7 else "moderada" if abs(correlacao) >= 0.4 else "bem fraca"
        sentido = "uma aumenta e a outra também aumenta" if inclinacao > 0 else "uma aumenta e a outra diminui" if inclinacao < 0 else "uma não muda a outra"
        st.info(f"**O que aprendemos aqui?** A ligação entre **{COLUNAS_PT.get(xcol, xcol)}** e **{COLUNAS_PT.get(ycol, ycol)}** é **{intensidade}**. Na prática: quando uma muda, a tendência é que **{sentido}**.")
        
        st.divider()
        st.markdown("##### Faça uma Adivinhação (Simulador)")
        entrada = st.number_input(f"Se o valor de '{COLUNAS_PT.get(xcol, xcol)}' for:", value=float(ms.media(xs)))
        predicao = intercepto + inclinacao * entrada
        st.success(f"A estimativa para '{COLUNAS_PT.get(ycol, ycol)}' será de aproximadamente: **{predicao:.2f}**")
        
    except ValueError as erro:
        st.warning(f"Não foi possível comparar estas duas opções: {erro}")

# --- ABA 6: DESCOBERTAS ---
with tab_descobertas:
    st.subheader("6. Resumo Final: O Que Descobrimos com a Pesquisa?")
    st.write("Abaixo estão as conclusões mais importantes encontradas nesta pesquisa de forma bem direta:")
    
    if "y" in df.columns:
        positivos = df[df["y"] == "yes"]
        negativos = df[df["y"] == "no"]
        taxa = len(positivos) / len(df) * 100
        
        st.success(f"**1. Poucas Pessoas Aceitam a Proposta:** De todas as pessoas que o banco ligou, apenas **{taxa:.1f}%** aceitaram guardar o dinheiro no banco.")
        desfechos = df["y"].value_counts().rename_axis("adesao").reset_index(name="registros")
        desfechos["adesao"] = desfechos["adesao"].map({"yes": "Sim, Aceitou", "no": "Não Aceitou"})
        
        fig_d1 = px.bar(desfechos, x="adesao", y="registros", title="Total de Pessoas que Aceitaram vs Recusaram", labels={"adesao": "Decisão do Cliente", "registros": "Quantidade de Pessoas"})
        fig_d1.update_layout(template="plotly_dark")
        st.plotly_chart(fig_d1, use_container_width=True)

        duracao_sim = ms.media(positivos["duration"]) if len(positivos) > 0 else 0
        duracao_nao = ms.media(negativos["duration"]) if len(negativos) > 0 else 0
        
        st.info(f"**2. Conversas Mais Longas Funcionam Melhor:** Quando a pessoa aceitou a proposta, a conversa ao telefone durou em média **{duracao_sim / 60:.1f} minutos** ({duracao_sim:.0f} segundos). Já quando a pessoa recusou, a conversa durou apenas **{duracao_nao / 60:.1f} minutos** ({duracao_nao:.0f} segundos).")
        medias_duracao = pd.DataFrame({"adesao": ["Sim, Aceitou", "Não Aceitou"], "duracao_media": [duracao_sim, duracao_nao]})
        
        fig_d2 = px.bar(medias_duracao, x="adesao", y="duracao_media", title="Tempo Médio da Conversa no Telefone (em segundos)", labels={"adesao": "Decisão do Cliente", "duracao_media": "Segundos de Conversa"})
        fig_d2.update_layout(template="plotly_dark")
        st.plotly_chart(fig_d2, use_container_width=True)