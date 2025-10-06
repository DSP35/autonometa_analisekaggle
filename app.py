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
from ydata_profiling import ProfileReport

# --- LangChain / Gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.tools import tool
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.agents import AgentExecutor
from langchain.schema import HumanMessage, AIMessage

# --- 1. CONFIGURAÇÃO INICIAL E CHAVE API ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("ERRO: A GEMINI_API_KEY não foi encontrada. Defina-a no Streamlit Secrets (st.secrets) com o nome 'GEMINI_API_KEY'.")
    st.stop()

# --- 2. ESTADO E FUNÇÕES AUXILIARES ---

if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'NOME_DO_ARQUIVO_REFERENCIA' not in st.session_state:
    st.session_state.NOME_DO_ARQUIVO_REFERENCIA = "Nenhum arquivo carregado"
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_data_for_high_cost_tool(df_global: pd.DataFrame, threshold: int = 100000) -> pd.DataFrame:
    if df_global.shape[0] > threshold:
        return df_global.sample(n=threshold, random_state=42).reset_index(drop=True)
    return df_global

def parse_comando_grafico(comando: str) -> tuple:
    match = re.search(r'(\w+)\s*\(([^)]*)\)', comando.strip().lower())
    if match:
        tipo = match.group(1)
        args = [a.strip().strip("'\"") for a in match.group(2).split(',') if a.strip()]
        coluna_x = args[0] if len(args) > 0 else None
        coluna_y = args[1] if len(args) > 1 and args[1] != 'none' else None
        return tipo, coluna_x, coluna_y
    parts = [p.strip().strip("'\"") for p in comando.split(',') if p.strip()]
    if len(parts) >= 2:
        return parts[1], parts[0], None
    elif len(parts) == 1:
        return 'hist', parts[0], None
    return None, None, None

# --- 3. FERRAMENTAS DO AGENTE ---

@tool
def otimizar_tipos_de_dados_para_memoria() -> str:
    """Otimiza os tipos de dados numéricos do DataFrame para reduzir uso de memória."""
    df = st.session_state.df
    if df is None:
        return "Erro: DataFrame não carregado."
    initial_mem = df.memory_usage(deep=True).sum()
    df_optimized = df.copy()
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        if np.issubdtype(col_type, np.integer):
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        elif np.issubdtype(col_type, np.floating):
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    final_mem = df_optimized.memory_usage(deep=True).sum()
    mem_saved = (initial_mem - final_mem) / (1024**2)
    percentage_saved = (initial_mem - final_mem) / initial_mem * 100
    if percentage_saved > 0.1:
        st.session_state.df = df_optimized
        return f"Sucesso: Tipos de dados otimizados. Memória economizada: {mem_saved:.2f} MB ({percentage_saved:.2f}%)."
    else:
        return f"Aviso: Otimização não significativa. Memória economizada: {mem_saved:.2f} MB."

@tool
def gerar_perfil_de_dados_e_salvar_html() -> str:
    """Gera um relatório HTML de perfil de dados (ydata-profiling)."""
    df = st.session_state.df
    if df is None: return "Erro: O DataFrame não está carregado."
    data_to_profile = get_data_for_high_cost_tool(df)
    try:
        profile = ProfileReport(
            data_to_profile,
            title=f"Relatório de Perfil de Dados - {st.session_state.NOME_DO_ARQUIVO_REFERENCIA}",
            html={"style": {"full_width": True}},
            sort=None,
            lazy=True
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            profile.to_file(tmp_file.name)
            output_file_path = tmp_file.name
        st.session_state.profile_report_path = output_file_path
        return f"Sucesso: Relatório de perfil de dados gerado ({data_to_profile.shape[0]} linhas)."
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
    df = st.session_state.df
    if df is None: return "Erro: O DataFrame não está carregado."
    data_to_plot = get_data_for_high_cost_tool(df)
    tipo_grafico, coluna_x, coluna_y = parse_comando_grafico(comando_grafico)
    if not coluna_x or not tipo_grafico: return "Erro de Parsing: Comando inválido."
    coluna_x_original = coluna_x
    colunas_df = {col.lower(): col for col in data_to_plot.columns}
    if coluna_x.lower() in colunas_df:
        coluna_x = colunas_df[coluna_x.lower()]
    else:
        return f"Erro: A coluna '{coluna_x_original}' não existe."
    if coluna_y and coluna_y.lower() in colunas_df:
        coluna_y = colunas_df[coluna_y.lower()]
    elif coluna_y:
        return f"Erro: A coluna Y '{coluna_y}' não existe."
    plt.figure(figsize=(12, 7))
    buffer = io.BytesIO()
    try:
        base_title = f" (Amostra de {data_to_plot.shape[0]} linhas)" if data_to_plot.shape[0] != df.shape[0] else ""
        if tipo_grafico == 'hist':
            sns.histplot(data_to_plot[coluna_x].dropna(), kde=True)
            plt.title(f'Distribuição de {coluna_x}{base_title}')
        elif tipo_grafico == 'scatter':
            if not coluna_y: return "Erro: Scatter requer duas colunas."
            sns.scatterplot(x=data_to_plot[coluna_x], y=data_to_plot[coluna_y])
            plt.title(f'Dispersão {coluna_x} vs {coluna_y}{base_title}')
        elif tipo_grafico == 'line':
            if not coluna_y: return "Erro: Line requer duas colunas."
            sns.lineplot(x=data_to_plot[coluna_x], y=data_to_plot[coluna_y])
            plt.title(f'Evolução de {coluna_y} por {coluna_x}{base_title}')
        elif tipo_grafico == 'box':
            if coluna_y and data_to_plot[coluna_y].nunique() < 50:
                sns.boxplot(x=data_to_plot[coluna_y], y=data_to_plot[coluna_x])
                plt.xlabel(coluna_y)
                if data_to_plot[coluna_y].nunique() > 10:
                    plt.xticks(rotation=45, ha='right')
            else:
                sns.boxplot(y=data_to_plot[coluna_x])
            plt.title(f'Boxplot de {coluna_x}{base_title}')
            plt.ylabel(coluna_x)
        elif tipo_grafico == 'bar':
            unique_count = data_to_plot[coluna_x].nunique()
            if unique_count > 50:
                return f"Erro: Muitas categorias ({unique_count}) para gráfico de barras."
            sns.countplot(x=data_to_plot[coluna_x], order=data_to_plot[coluna_x].value_counts().index)
            plt.title(f'Contagem de {coluna_x}{base_title}')
            plt.xlabel(coluna_x)
            plt.ylabel('Contagem')
            if unique_count > 10:
                plt.xticks(rotation=45, ha='right')
        else:
            return f"Erro: Tipo '{tipo_grafico}' não suportado."
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        st.session_state.graph_buffer = buffer.getvalue()
        st.session_state.graph_filename = f"grafico_{tipo_grafico}_{coluna_x}.png"
        return f"Sucesso: Gráfico '{tipo_grafico}' gerado."
    except Exception as e:
        plt.close()
        return f"Erro inesperado ao gerar gráfico: {e}"
    finally:
        plt.close()

# --- 4. CRIAÇÃO DO AGENTE ---

def create_agent(df: pd.DataFrame):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=2048,
        api_key=GEMINI_API_KEY
    )
    tools = [otimizar_tipos_de_dados_para_memoria, gerar_perfil_de_dados_e_salvar_html, gerar_visualizacao]

    CUSTOM_PREFIX = """
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
    CUSTOM_SUFFIX = ""

    history = StreamlitChatMessageHistory(key="messages")

    memory = ConversationBufferWindowMemory(
        k=10,
        chat_memory=history,
        memory_key="conversation_history",
        return_messages=True
    )

    agent_framework = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type="openai-tools",
        extra_tools=tools,
        allow_dangerous_code=True,
        prefix=CUSTOM_PREFIX,
        suffix=CUSTOM_SUFFIX
    )

    executor = AgentExecutor(
        agent=agent_framework.agent,
        tools=agent_framework.tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )

    return executor

# --- 5. INTERFACE STREAMLIT ---

st.set_page_config(layout="wide")
st.title("🤖 Agente de Análise de Dados com Gemini")

uploaded_file = st.sidebar.file_uploader("Carregue seu arquivo CSV", type="csv")

if uploaded_file is not None and (st.session_state.df is None or st.session_state.NOME_DO_ARQUIVO_REFERENCIA != uploaded_file.name):
    st.session_state.messages = []
    with st.spinner(f"Carregando {uploaded_file.name} e inicializando o agente..."):
        try:
            content_bytes = uploaded_file.getvalue()
            try:
                content = content_bytes.decode('utf-8-sig')
            except Exception:
                content = content_bytes.decode('latin1')
            df = pd.read_csv(io.StringIO(content), sep=None, engine='python')
            st.session_state.df = df
            st.session_state.NOME_DO_ARQUIVO_REFERENCIA = uploaded_file.name
            st.session_state.agent = create_agent(df)
            st.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso. Agente pronto!")
        except Exception as e:
            st.error(f"Erro ao ler o arquivo CSV ou inicializar o agente: {e}")
            st.error(traceback.format_exc())
            st.session_state.df = None

# --- Bloco Lateral (Sidebar) ---
if st.session_state.df is not None:
    st.sidebar.markdown(f"**Arquivo carregado:** `{st.session_state.NOME_DO_ARQUIVO_REFERENCIA}`")
    st.sidebar.metric(label="Linhas", value=st.session_state.df.shape[0])
    st.sidebar.metric(label="Colunas", value=st.session_state.df.shape[1])
    st.sidebar.markdown("**Amostragem de dados:**")
    st.sidebar.dataframe(st.session_state.df.head(5))
    
    # Botões de download do relatório de perfil (com limpeza)
    if 'profile_report_path' in st.session_state:
        with open(st.session_state.profile_report_path, "rb") as file:
            st.sidebar.download_button(
                label="📥 Baixar Relatório de Perfil (.html)",
                data=file,
                file_name="relatorio_perfil.html",
                mime="text/html"
            )
        try:
            os.remove(st.session_state.profile_report_path)
        except OSError:
            pass
        del st.session_state.profile_report_path
            
# --- HISTÓRICO DE MENSAGENS (suporta dict e HumanMessage/AIMessage) ---

for message in st.session_state.messages:
    if isinstance(message, dict):
        role = message.get("role", "assistant")
        content = message.get("content", "")
    elif isinstance(message, HumanMessage):
        role = "user"
        content = message.content
    elif isinstance(message, AIMessage):
        role = "assistant"
        content = message.content
    else:
        role = "assistant"
        content = str(message)

    with st.chat_message(role):
        st.markdown(content)

# --- BLOCO DE CHAT ---

if st.session_state.agent:
    pergunta = st.chat_input("Digite sua pergunta de análise de dados aqui...")
    if pergunta:
        with st.chat_message("user"):
            st.markdown(pergunta)
        with st.chat_message("assistant"):
            with st.spinner("O Agente está pensando..."):
                output_buffer = io.StringIO()
                sys.stdout = output_buffer
                try:
                    resposta = st.session_state.agent.invoke({"input": pergunta})
                    sys.stdout = sys.__stdout__
                    assistant_message_content = resposta['output']
                    st.markdown(assistant_message_content)
                    with st.expander("Rastreio da Execução (Verbose)"):
                        st.code(output_buffer.getvalue(), language='log')
                except Exception as e:
                    sys.stdout = sys.__stdout__
                    assistant_message_content = f"❌ Erro na execução do Agente: {e}"
                    st.error(assistant_message_content)

        if 'graph_buffer' in st.session_state and st.session_state.graph_buffer:
            st.markdown("---")
            st.subheader("Gráfico Gerado")
            img_base64 = base64.b64encode(st.session_state.graph_buffer).decode()
            st.markdown(f'![{st.session_state.graph_filename}](data:image/png;base64,{img_base64})')
            del st.session_state.graph_buffer
            del st.session_state.graph_filename

        st.rerun()
else:
    st.warning("Por favor, carregue um arquivo CSV na barra lateral para começar a análise.")





