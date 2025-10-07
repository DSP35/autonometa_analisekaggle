import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
from pathlib import Path
import io
import sys
import tempfile
import traceback
import base64
import logging
from ydata_profiling import ProfileReport

# --- Integração com LangChain e Gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.tools import tool
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.agents import AgentExecutor
from langchain.schema import HumanMessage, AIMessage

# --- Configuração Inicial e Chave da API ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("ERRO: A GEMINI_API_KEY não foi encontrada. Defina-a no Streamlit Secrets (st.secrets) com o nome 'GEMINI_API_KEY'.")
    st.stop()

# --- Estados e Funções de Suporte ---

if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None
if 'agente' not in st.session_state:
    st.session_state.agente = None
if 'NOME_DO_ARQUIVO_ATUAL' not in st.session_state:
    st.session_state.NOME_DO_ARQUIVO_ATUAL = "Nenhum arquivo carregado"
if "historico_mensagens" not in st.session_state:
    st.session_state.historico_mensagens = []

def obter_dados_para_ferramenta_cara(df_geral: pd.DataFrame, limite: int = 100000) -> pd.DataFrame:
    if df_geral.shape[0] > limite:
        return df_geral.sample(n=limite, random_state=42).reset_index(drop=True)
    return df_geral

def interpretar_comando_para_grafico(entrada: str) -> tuple:
    resultado = re.search(r'(\w+)\s*\(([^)]*)\)', entrada.strip().lower())
    if resultado:
        categoria = resultado.group(1)
        parametros = [p.strip().strip("'\"") for p in resultado.group(2).split(',') if p.strip()]
        eixo_x = parametros[0] if len(parametros) > 0 else None
        eixo_y = parametros[1] if len(parametros) > 1 and parametros[1] != 'none' else None
        return categoria, eixo_x, eixo_y
    fragmentos = [p.strip().strip("'\"") for p in entrada.split(',') if p.strip()]
    if len(fragmentos) >= 2:
        return fragmentos[1], fragmentos[0], None
    elif len(fragmentos) == 1:
        return 'hist', fragmentos[0], None
    return None, None, None

# --- Ferramentas para o Agente ---

@st.cache_data
def carregar_arquivo_csv(arquivo_enviado):
    bytes_conteudo = arquivo_enviado.getvalue()
    try:
        conteudo = bytes_conteudo.decode('utf-8-sig')
    except:
        conteudo = bytes_conteudo.decode('latin1')
    return pd.read_csv(io.StringIO(conteudo), sep=None, engine='python')

@st.cache_data
def criar_relatorio_perfil(amostra_df):
    return ProfileReport(
        amostra_df,
        title=f"Relatório de Perfil de Dados - {st.session_state.NOME_DO_ARQUIVO_ATUAL}",
        html={"style": {"full_width": True}},
        sort=None,
        lazy=True
    )

@tool
def otimizar_tipos_de_dados_para_memoria() -> str:
    """Otimiza os tipos de dados numéricos do DataFrame para reduzir uso de memória."""
    df = st.session_state.dataframe
    if df is None:
        return "Erro: DataFrame não carregado."
    memoria_inicial = df.memory_usage(deep=True).sum()
    df_otimizado = df.copy()
    for coluna in df_otimizado.columns:
        tipo_coluna = df_otimizado[coluna].dtype
        if np.issubdtype(tipo_coluna, np.integer):
            df_otimizado[coluna] = pd.to_numeric(df_otimizado[coluna], downcast='integer')
        elif np.issubdtype(tipo_coluna, np.floating):
            df_otimizado[coluna] = pd.to_numeric(df_otimizado[coluna], downcast='float')
    memoria_final = df_otimizado.memory_usage(deep=True).sum()
    memoria_economizada = (memoria_inicial - memoria_final) / (1024**2)
    porcentagem_economizada = (memoria_inicial - memoria_final) / memoria_inicial * 100
    if porcentagem_economizada > 0.1:
        st.session_state.dataframe = df_otimizado
        return f"Sucesso: Tipos de dados otimizados. Memória economizada: {memoria_economizada:.2f} MB ({porcentagem_economizada:.2f}%)."
    else:
        return f"Aviso: Otimização não significativa. Memória economizada: {memoria_economizada:.2f} MB."

@tool
def gerar_perfil_de_dados_e_salvar_html() -> str:
    """Gera um relatório HTML de perfil de dados (ydata-profiling)."""
    df = st.session_state.dataframe
    if df is None: return "Erro: O DataFrame não está carregado."
    amostra_perfil = obter_dados_para_ferramenta_cara(df, threshold=50000)
    try:
        perfil = criar_relatorio_perfil(amostra_perfil)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as arquivo_temp:
            perfil.to_file(arquivo_temp.name)
            caminho_arquivo = arquivo_temp.name
        st.session_state.caminho_relatorio_perfil = caminho_arquivo
        return f"Sucesso: Relatório de perfil de dados gerado ({amostra_perfil.shape[0]} linhas)."
    except Exception as e:
        return f"Erro CRÍTICO ao gerar o perfil de dados. Detalhe: {e}"

@tool
def gerar_visualizacao(comando_grafico: str) -> str:
    """
    Gera um gráfico PNG a partir de um comando simples.
    O formato do comando deve ser: 'tipo_de_grafico(coluna_x, coluna_y)' ou 'coluna_x, tipo_de_grafico'.
    Tipos suportados: hist, scatter, line, box, bar.
    Exemplos: 'hist(Amount)', 'scatter(Preço, Quantidade)', 'Categoria, bar'.
    """
    df = st.session_state.dataframe
    if df is None: return "Erro: O DataFrame não está carregado."
    dados_para_plot = obter_dados_para_ferramenta_cara(df)
    tipo_plot, eixo_x, eixo_y = interpretar_comando_para_grafico(comando_grafico)
    if not eixo_x or not tipo_plot: return "Erro de Parsing: Comando inválido."
    eixo_x_original = eixo_x
    colunas_no_df = {col.lower(): col for col in dados_para_plot.columns}
    if eixo_x.lower() in colunas_no_df:
        eixo_x = colunas_no_df[eixo_x.lower()]
    else:
        return f"Erro: A coluna '{eixo_x_original}' não existe."
    if eixo_y and eixo_y.lower() in colunas_no_df:
        eixo_y = colunas_no_df[eixo_y.lower()]
    elif eixo_y:
        return f"Erro: A coluna Y '{eixo_y}' não existe."
    plt.figure(figsize=(12, 7))
    buffer_img = io.BytesIO()
    try:
        titulo_base = f" (Amostra de {dados_para_plot.shape[0]} linhas)" if dados_para_plot.shape[0] != df.shape[0] else ""
        if tipo_plot == 'hist':
            sns.histplot(dados_para_plot[eixo_x].dropna(), kde=True)
            plt.title(f'Distribuição de {eixo_x}{titulo_base}')
        elif tipo_plot == 'scatter':
            if not eixo_y: return "Erro: Scatter requer duas colunas."
            sns.scatterplot(x=dados_para_plot[eixo_x], y=dados_para_plot[eixo_y])
            plt.title(f'Dispersão {eixo_x} vs {eixo_y}{titulo_base}')
        elif tipo_plot == 'line':
            if not eixo_y: return "Erro: Line requer duas colunas."
            sns.lineplot(x=dados_para_plot[eixo_x], y=dados_para_plot[eixo_y])
            plt.title(f'Evolução de {eixo_y} por {eixo_x}{titulo_base}')
        elif tipo_plot == 'box':
            if eixo_y and dados_para_plot[eixo_y].nunique() < 50:
                sns.boxplot(x=dados_para_plot[eixo_y], y=dados_para_plot[eixo_x])
                plt.xlabel(eixo_y)
                if dados_para_plot[eixo_y].nunique() > 10:
                    plt.xticks(rotation=45, ha='right')
            else:
                sns.boxplot(y=dados_para_plot[eixo_x])
            plt.title(f'Boxplot de {eixo_x}{titulo_base}')
            plt.ylabel(eixo_x)
        elif tipo_plot == 'bar':
            qtd_unica = dados_para_plot[eixo_x].nunique()
            if qtd_unica > 50:
                return f"Erro: Muitas categorias ({qtd_unica}) para gráfico de barras."
            sns.countplot(x=dados_para_plot[eixo_x], order=dados_para_plot[eixo_x].value_counts().index)
            plt.title(f'Contagem de {eixo_x}{titulo_base}')
            plt.xlabel(eixo_x)
            plt.ylabel('Contagem')
            if qtd_unica > 10:
                plt.xticks(rotation=45, ha='right')
        else:
            return f"Erro: Tipo '{tipo_plot}' não suportado."
        plt.tight_layout()
        plt.savefig(buffer_img, format='png')
        buffer_img.seek(0)
        plt.close()
        st.session_state.buffer_grafico = buffer_img.getvalue()
        st.session_state.nome_arquivo_grafico = f"grafico_{tipo_plot}_{eixo_x}.png"
        return f"Sucesso: Gráfico '{tipo_plot}' gerado."
    except Exception as e:
        plt.close()
        return f"Erro inesperado ao gerar gráfico: {e}"
    finally:
        plt.close()

# --- Configuração do Agente ---

def criar_agente(df: pd.DataFrame):
    modelo_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=2048,
        api_key=GEMINI_API_KEY
    )
    lista_ferramentas = [otimizar_tipos_de_dados_para_memoria, gerar_perfil_de_dados_e_salvar_html, gerar_visualizacao]

    PREFIXO_PERSONALIZADO = """
Você é um analista de dados especializado em Exploratory Data Analysis (EDA). Sua função é analisar o DataFrame fornecido de forma profunda e iterativa, explorando aspectos como: estatísticas descritivas (média, mediana, desvio padrão, quartis), distribuições de variáveis, correlações entre colunas, identificação de outliers, valores faltantes, padrões temporais ou categóricos, e insights acionáveis baseados nos dados.

Responda exclusivamente a perguntas relacionadas ao dataset, fornecendo análises claras, concisas e baseadas em evidências. Sempre considere o contexto completo da conversa para tirar conclusões cumulativas, evitando repetições desnecessárias.

Pense passo a passo antes de responder:
1. Entenda a query do usuário e relacione com o histórico da conversa.
2. Verifique se a análise pode ser feita diretamente com consultas ao DataFrame (ex.: df.describe(), df.corr()).
3. Use ferramentas (como otimização de tipos, geração de perfil de dados ou visualizações) SOMENTE se explicitamente solicitado pelo usuário ou se for essencial para responder com precisão (ex.: "gere um gráfico" ou quando a query exige visualização para clareza). Evite chamadas desnecessárias.
4. Interprete os resultados e forneça insights úteis, sugerindo próximos passos de análise se relevante.

Histórico da conversa:
{conversation_history}
    """
    SUFIXO_PERSONALIZADO = ""

    historico_chat = StreamlitChatMessageHistory(key="historico_mensagens")

    memoria_conversa = ConversationBufferWindowMemory(
        k=10,
        chat_memory=historico_chat,
        memory_key="conversation_history",
        return_messages=True
    )

    df_para_agente = df if df.shape[0] < 50000 else df.sample(n=50000, random_state=42).reset_index(drop=True)

    estrutura_agente = create_pandas_dataframe_agent(
        modelo_llm,
        df_para_agente,
        verbose=False,
        agent_type="openai-tools",
        extra_tools=lista_ferramentas,
        allow_dangerous_code=True,
        prefix=PREFIXO_PERSONALIZADO,
        suffix=SUFIXO_PERSONALIZADO
    )

    executor_agente = AgentExecutor(
        agent=estrutura_agente.agent,
        tools=estrutura_agente.tools,
        memory=memoria_conversa,
        verbose=False,
        handle_parsing_errors=True,
    )

    return executor_agente

# --- Interface do Streamlit ---

st.set_page_config(layout="wide")
st.title("🤖 Agente de Análise de Dados com Gemini")

st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <img src="https://i.imgur.com/oH1wbZ4.png" width="150">
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader("Carregue seu arquivo CSV", type="csv")

if uploaded_file is not None and (st.session_state.dataframe is None or st.session_state.NOME_DO_ARQUIVO_ATUAL != uploaded_file.name):
    st.session_state.historico_mensagens = []
    with st.spinner(f"Carregando {uploaded_file.name} e inicializando o agente..."):
        try:
            df = carregar_arquivo_csv(uploaded_file)
            if df.shape[0] > 10000:
                memoria_inicial = df.memory_usage(deep=True).sum()
                for col in df.columns:
                    tipo_col = df[col].dtype
                    if np.issubdtype(tipo_col, np.integer):
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                    elif np.issubdtype(tipo_col, np.floating):
                        df[col] = pd.to_numeric(df[col], downcast='float')
                memoria_final = df.memory_usage(deep=True).sum()
                economizado = (memoria_inicial - memoria_final) / (1024**2)
                st.info(f"DF otimizado automaticamente: {economizado:.2f} MB economizados.")
            st.session_state.dataframe = df
            st.session_state.NOME_DO_ARQUIVO_ATUAL = uploaded_file.name
            st.session_state.agente = criar_agente(df)
            st.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso. Agente pronto!")
        except Exception as e:
            st.error(f"Erro ao ler o arquivo CSV ou inicializar o agente: {e}")
            st.error(traceback.format_exc())
            st.session_state.dataframe = None

# --- Exibição do Histórico de Mensagens (suporta dict e HumanMessage/AIMessage) ---

for msg in st.session_state.historico_mensagens:
    if isinstance(msg, dict):
        papel = msg.get("role", "assistant")
        texto = msg.get("content", "")
    elif isinstance(msg, HumanMessage):
        papel = "user"
        texto = msg.content
    elif isinstance(msg, AIMessage):
        papel = "assistant"
        texto = msg.content
    else:
        papel = "assistant"
        texto = str(msg)

    with st.chat_message(papel):
        st.markdown(texto)

# --- Seção de Chat ---

if st.session_state.dataframe is not None:
    st.sidebar.markdown(f"**Arquivo carregado:** `{st.session_state.NOME_DO_ARQUIVO_ATUAL}`")
    
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        st.metric(label="Linhas", value=st.session_state.dataframe.shape[0])
    with col_b:
        st.metric(label="Colunas", value=st.session_state.dataframe.shape[1])
    
    st.sidebar.markdown("**Amostragem de dados:**")
    st.sidebar.dataframe(st.session_state.dataframe.head(5))
    
    if 'caminho_relatorio_perfil' in st.session_state:
        with open(st.session_state.caminho_relatorio_perfil, "rb") as arq:
            st.sidebar.download_button(
                label="📥 Baixar Relatório de Perfil (.html)",
                data=arq,
                file_name="relatorio_perfil.html",
                mime="text/html"
            )
        try:
            os.remove(st.session_state.caminho_relatorio_perfil)
        except OSError:
            pass
        del st.session_state.caminho_relatorio_perfil

if st.session_state.agente:
    consulta = st.chat_input("Digite sua pergunta de análise de dados aqui...")
    if consulta:
        with st.chat_message("user"):
            st.markdown(consulta)
        
        with st.chat_message("assistant"):
            with st.spinner("O Agente está pensando..."):
                captura_log = io.StringIO()
                sys.stdout = captura_log
                try:
                    resultado = st.session_state.agente.invoke({"input": consulta})
                    sys.stdout = sys.__stdout__
                    conteudo_assistente = resultado['output']

                    if 'buffer_grafico' in st.session_state and st.session_state.buffer_grafico:
                        st.markdown("---")
                        st.subheader("Gráfico Gerado")
                        base64_img = base64.b64encode(st.session_state.buffer_grafico).decode()
                        markdown_img = f'![{st.session_state.nome_arquivo_grafico}](data:image/png;base64,{base64_img})'
                        
                        conteudo_assistente += f"\n\n{markdown_img}"
                        
                        if st.session_state.historico_mensagens and isinstance(st.session_state.historico_mensagens[-1], AIMessage):
                            st.session_state.historico_mensagens[-1].content = conteudo_assistente
                        
                        del st.session_state.buffer_grafico
                        del st.session_state.nome_arquivo_grafico

                        st.download_button(
                            label="📥 Download Gráfico",
                            data=st.session_state.buffer_grafico,
                            file_name=st.session_state.nome_arquivo_grafico,
                            mime="image/png"
                        )

                    st.markdown(conteudo_assistente)

                except Exception as e:
                    sys.stdout = sys.__stdout__
                    msg_erro = f"❌ Erro na execução do Agente: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                    
                    if 'ultimo_erro' not in st.session_state:
                        st.session_state.ultimo_erro = []
                    st.session_state.ultimo_erro.append(msg_erro)
                    
                    st.error(msg_erro)
                    st.exception(e)
                    
                    logging.error(msg_erro)
else:
    st.warning("Por favor, carregue um arquivo CSV na barra lateral para começar a análise.")
