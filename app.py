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
from ydata_profiling import ProfileReport

# Importações LangChain/Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor

# --- 1. CONFIGURAÇÃO INICIAL E CHAVE API (TOTALMENTE GENÉRICA) ---

# Tenta ler a chave do Streamlit Secrets (modo recomendado no Streamlit Cloud)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("ERRO: A GEMINI_API_KEY não foi encontrada. Defina-a no Streamlit Secrets (st.secrets) com o nome 'GEMINI_API_KEY'.")
    st.stop()

# --- 2. VARIÁVEIS DE ESTADO E FUNÇÕES DE AJUDA ---

# Inicialização de estado de sessão
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'NOME_DO_ARQUIVO_REFERENCIA' not in st.session_state:
    st.session_state.NOME_DO_ARQUIVO_REFERENCIA = "Nenhum arquivo carregado"


def get_data_for_high_cost_tool(df_global: pd.DataFrame, threshold: int = 100000) -> pd.DataFrame:
    """Retorna o DataFrame principal ou uma amostra de 100k linhas se ele for muito grande."""
    if df_global.shape[0] > threshold:
        return df_global.sample(n=threshold, random_state=42).reset_index(drop=True)
    return df_global

def parse_comando_grafico(comando: str) -> tuple:
    """Extrai coluna_x, tipo_grafico e coluna_y de uma string de comando."""
    match = re.search(r'(\w+)\s*\(([^)]*)\)', comando.strip().lower())
    if match:
        tipo = match.group(1)
        args = [a.strip().strip("'\"") for a in match.group(2).split(',') if a.strip()]
        coluna_x = args[0] if len(args) > 0 else None
        coluna_y = args[1] if len(args) > 1 else None
        return tipo, coluna_x, coluna_y
    
    parts = [p.strip().strip("'\"") for p in comando.split(',') if p.strip()]
    if len(parts) >= 2:
        return parts[1], parts[0], None
    elif len(parts) == 1:
        return 'hist', parts[0], None
    return None, None, None


# --- 3. FERRAMENTAS DO AGENTE (@tool) ---

@tool
def otimizar_tipos_de_dados_para_memoria() -> str:
    """
    Otimiza os tipos de dados do DataFrame (downcasting de numéricos) para reduzir o consumo de memória.
    Esta otimização é aplicada diretamente ao DataFrame principal.
    :return: Relatório da economia de memória em MB e porcentagem.
    """
    df = st.session_state.df
    if df is None: return "Erro: DataFrame não carregado."
    
    initial_mem = df.memory_usage(deep=True).sum()
    df_optimized = df.copy()

    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        if np.issubdtype(col_type, np.integer):
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        elif np.issubdtype(col_type, np.floating):
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
    final_mem = df_optimized.memory_usage(deep=True).sum()
    mem_saved = (initial_mem - final_mem) / (1024**2) # MB
    percentage_saved = (initial_mem - final_mem) / initial_mem * 100

    if percentage_saved > 0.1:
        st.session_state.df = df_optimized
        return f"Sucesso: Tipos de dados otimizados. Memória economizada: {mem_saved:.2f} MB ({percentage_saved:.2f}%). Otimização aplicada ao DataFrame principal (df)."
    else:
        return f"Aviso: Otimização de tipos de dados não foi significativa. Memória economizada: {mem_saved:.2f} MB."


@tool
def gerar_perfil_de_dados_e_salvar_html() -> str:
    """
    Gera um relatório HTML de perfil de dados (ydata-profiling). Otimiza o uso de memória 
    usando uma amostra interna se o DataFrame principal for muito grande.
    O relatório é salvo em um arquivo temporário e anexado ao Streamlit.
    :return: Confirmação do arquivo salvo e instrução para o usuário baixar.
    """
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
        
        return f"Sucesso: Relatório de perfil de dados (baseado em {data_to_profile.shape[0]} linhas) gerado. O usuário pode baixar o arquivo HTML."

    except Exception as e:
        return f"Erro CRÍTICO ao gerar o perfil de dados. Detalhe: {e}"


@tool
def gerar_visualizacao(comando_grafico: str) -> str:
    """
    Gera um gráfico PNG baseado em um comando simples. Tipos suportados: 'hist', 'scatter', 'box', 'bar', 'line'.
    Exemplos: 'line(Time, Amount)', 'hist(Amount)'.
    :param comando_grafico: Comando do gráfico no formato 'tipo(coluna_x, coluna_y)'.
    :return: Confirmação e o caminho do arquivo PNG gerado.
    """
    df = st.session_state.df
    if df is None: return "Erro: O DataFrame não está carregado. Não é possível gerar o gráfico."
    
    data_to_plot = get_data_for_high_cost_tool(df)

    tipo_grafico, coluna_x, coluna_y = parse_comando_grafico(comando_grafico)
    
    if not coluna_x or not tipo_grafico: return "Erro de Parsing: Comando inválido. Use o formato 'tipo(coluna_x, coluna_y)'. Ex: line(Time, Amount)."
    
    coluna_x_original = coluna_x
    
    colunas_df = {col.lower(): col for col in data_to_plot.columns}
    
    if coluna_x.lower() in colunas_df:
        coluna_x = colunas_df[coluna_x.lower()]
    else:
        return f"Erro: A coluna '{coluna_x_original}' não existe no DataFrame."
        
    if coluna_y and coluna_y.lower() in colunas_df:
        coluna_y = colunas_df[coluna_y.lower()]
    elif coluna_y:
        return f"Erro: A coluna Y '{coluna_y}' não existe no DataFrame."

    plt.figure(figsize=(12, 7))
    buffer = io.BytesIO()
    
    try:
        base_title = f" (Amostra de {data_to_plot.shape[0]} linhas)" if data_to_plot.shape[0] != df.shape[0] else ""

        if tipo_grafico == 'hist':
            sns.histplot(data_to_plot[coluna_x].dropna(), kde=True)
            plt.title(f'Distribuição de {coluna_x}{base_title}')
        elif tipo_grafico == 'scatter':
            if not coluna_y: return "Erro: O tipo 'scatter' requer duas colunas (X e Y)."
            sns.scatterplot(x=data_to_plot[coluna_x], y=data_to_plot[coluna_y])
            plt.title(f'Dispersão entre {coluna_x} e {coluna_y}{base_title}')
        elif tipo_grafico == 'line': 
            if not coluna_y: return "Erro: O tipo 'line' requer duas colunas (X e Y)."
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
            plt.title(f'Boxplot de {coluna_x} vs {coluna_y or "Geral"}{base_title}')
            plt.ylabel(coluna_x)
        elif tipo_grafico == 'bar':
            unique_count = data_to_plot[coluna_x].nunique()
            if unique_count > 50: return f"Erro: A coluna '{coluna_x}' tem muitas categorias ({unique_count}) para um gráfico de barras. Sugestão: tente 'hist'."
            
            sns.countplot(x=data_to_plot[coluna_x], order=data_to_plot[coluna_x].value_counts().index)
            plt.title(f'Contagem de {coluna_x}{base_title}')
            plt.xlabel(coluna_x)
            plt.ylabel('Contagem')
            
            if unique_count > 10:
                plt.xticks(rotation=45, ha='right')
            
        else:
            return f"Erro: Tipo de gráfico '{tipo_grafico}' não suportado. Use 'hist', 'scatter', 'box', 'bar' ou 'line'."

        plt.tight_layout()
        plt.savefig(buffer, format='png')
        plt.close()

        st.session_state.graph_buffer = buffer.getvalue()
        st.session_state.graph_filename = f"grafico_{tipo_grafico}_{coluna_x}.png"
        
        return f"Sucesso: Gráfico '{tipo_grafico}' gerado. O Streamlit irá exibi-lo abaixo."

    except Exception as e:
        plt.close()
        return f"Erro inesperado ao gerar o gráfico. Detalhe: {e}"
    finally:
        plt.close()


# --- 4. FUNÇÃO DE CRIAÇÃO DO AGENTE ---

@st.cache_resource(show_spinner="Inicializando o Agente de IA...")
def create_agent(df: pd.DataFrame):
    """Inicializa o agente de análise de dados com Memória Conversacional."""
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0, 
        max_tokens=2048,
        api_key=GEMINI_API_KEY
    )
    
    CUSTOM_PREFIX = """
    Você é um agente de ANÁLISE DE DADOS. Sua principal função é analisar o DataFrame pandas carregado e gerar visualizações, estatísticas ou perfis.
    Siga as regras rigorosamente.
    """

    # --- CORREÇÃO DE ESCOPO: DEFINIÇÃO DA LISTA DE FERRAMENTAS DENTRO DA FUNÇÃO ---
    tools = [
        otimizar_tipos_de_dados_para_memoria, 
        gerar_perfil_de_dados_e_salvar_html, 
        gerar_visualizacao 
    ]
    
    # --- CONFIGURAÇÃO DE MEMÓRIA ---
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    
    # 1. Cria o Agente base
    agent_framework = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type="openai-tools",
        extra_tools=tools, # 'tools' agora está definido
        handle_parsing_errors=True,
        allow_dangerous_code=True,
        agent_kwargs={"prefix": CUSTOM_PREFIX}
    )
    
    # 2. Envolve o agente em um AgentExecutor com Memória
    executor = AgentExecutor(
        agent=agent_framework.agent,
        tools=agent_framework.tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )
    
    return executor

# --- 5. INTERFACE STREAMLIT PRINCIPAL (main) ---

st.set_page_config(layout="wide")

st.title("🤖 Agente de Análise de Dados com Gemini")

# Inicialização do Histórico de Chat e Flags de Estado 
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized_chat" not in st.session_state:
    st.session_state.initialized_chat = False # NOVO FLAG

# Upload de Arquivo
uploaded_file = st.sidebar.file_uploader("Carregue seu arquivo CSV", type="csv")

# Lógica de carregamento e inicialização do agente
# Esta lógica foca APENAS no carregamento do DF e do Agente.
if uploaded_file is not None and (st.session_state.df is None or st.session_state.NOME_DO_ARQUIVO_REFERENCIA != uploaded_file.name):
    
    # Redefine o estado ao carregar um NOVO arquivo
    st.session_state.messages = [] 
    st.session_state.initialized_chat = False # Reseta o flag para enviar nova saudação
    
    with st.spinner(f"Carregando {uploaded_file.name} e inicializando o agente..."):
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.NOME_DO_ARQUIVO_REFERENCIA = uploaded_file.name
            
            st.session_state.agent = create_agent(df)
            st.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso. Agente pronto!")
            
        except Exception as e:
            st.error(f"Erro ao ler o arquivo CSV: {e}")
            st.session_state.df = None

# --- NOVO BLOCO DE CONTROLE DE MENSAGEM INICIAL ---
# Adiciona mensagem de boas-vindas APENAS UMA VEZ após o agente estar pronto.
if st.session_state.agent is not None and not st.session_state.initialized_chat:
    
    # 1. Adiciona a mensagem de boas-vindas ao histórico
    st.session_state.messages.append({"role": "assistant", "content": f"Olá! Sou o Agente de Análise de Dados. O arquivo `{st.session_state.NOME_DO_ARQUIVO_REFERENCIA}` com {st.session_state.df.shape[0]} linhas foi carregado com sucesso. Como posso ajudar na análise?"})
    
    # 2. Define o flag como True para que não seja mais executado
    st.session_state.initialized_chat = True

# --- Bloco Lateral (Sidebar) ---
if st.session_state.df is not None:
    st.sidebar.markdown(f"**Arquivo carregado:** `{st.session_state.NOME_DO_ARQUIVO_REFERENCIA}`")
    st.sidebar.markdown("**Amostragem de dados:**")
    st.sidebar.dataframe(st.session_state.df.head(5))
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(label="Linhas", value=st.session_state.df.shape[0])
        st.metric(label="Colunas", value=st.session_state.df.shape[1])
        
        # Botões de download do relatório de perfil (com limpeza)
        if 'profile_report_path' in st.session_state:
            with open(st.session_state.profile_report_path, "rb") as file:
                st.download_button(
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


# --- Exibição do Histórico de Chat ---
# Remove os dados da barra lateral (col1, col2) e exibe o histórico no corpo principal
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Campo de Entrada de Chat Fixo na Parte Inferior ---
if st.session_state.agent:
    pergunta = st.chat_input("Digite sua pergunta de análise de dados aqui...")
    
    if pergunta:
        # 1. Adiciona a pergunta do usuário ao histórico e exibe
        st.session_state.messages.append({"role": "user", "content": pergunta})
        with st.chat_message("user"):
            st.markdown(pergunta)

        # 2. Executa o agente
        with st.chat_message("assistant"):
            with st.spinner("O Agente está pensando e analisando os dados..."):
                output_buffer = io.StringIO()
                sys.stdout = output_buffer
                
                try:
                    resposta = st.session_state.agent.invoke({"input": pergunta})
                    sys.stdout = sys.__stdout__
                    
                    output_text = resposta['output']
                    
                    # 3. Exibe a resposta final e a adiciona ao histórico
                    st.markdown(output_text)
                    
                    # 4. Exibe o rastreio (verbose) em um expander
                    with st.expander("Rastreio da Execução (Verbose)"):
                        st.code(output_buffer.getvalue(), language='log')

                except Exception as e:
                    sys.stdout = sys.__stdout__
                    output_text = f"❌ Erro na execução do Agente. Detalhe: {e}"
                    st.error(output_text)

                # Adiciona a resposta (ou erro) ao histórico da sessão
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
        
        # 5. Exibição de Gráfico Gerado (O Streamlit redesenha o histórico acima)
        if 'graph_buffer' in st.session_state and st.session_state.graph_buffer:
            st.markdown("---")
            st.subheader("Gráfico Gerado")
            st.image(st.session_state.graph_buffer, caption=st.session_state.graph_filename)
            del st.session_state.graph_buffer
        
        # Opcional: Reruns para garantir que o chat scroll para baixo
        # st.rerun() # Descomentar se o chat não rolar automaticamente

else:
    st.warning("Por favor, carregue um arquivo CSV na barra lateral para começar a análise.")





