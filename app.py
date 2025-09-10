import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq
import unicodedata
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Chatbot de Manutenção", page_icon="🛠", layout="centered")

@st.cache_data
def carregar_dados():
    return pd.read_excel('manutencao.xlsx')
df = carregar_dados()

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="mixtral-8x7b-32768")

def extrair_nome_equipamento(pergunta):
    prompt_extracao = f"""
                    Extraia apenas o nome do equipamento mencionado na pergunta abaixo.
                    Não adicione nada além do nome exato encontrado.
                    Caso nenhum equipamento seja encontrado responda
                    Não há equipamento mencionado na pergunta.

                    Pergunta: "{pergunta}"
                    """
    resposta = llm.invoke(prompt_extracao)
    return resposta.content.strip()

def remover_acentos(texto):
    if not isinstance(texto, str):
        return texto
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def busca_dados_relevantes_equipamento(nome_equipamento):
    nome_normalizado = remover_acentos(nome_equipamento).lower()
    df_normalizado = df.copy()
    df_normalizado['Nome_Equipamento'] = df_normalizado['Nome_Equipamento'].apply(remover_acentos).str.lower()   
    return df_normalizado[df_normalizado['Nome_Equipamento'] == nome_normalizado]

def perguntar_ia(pergunta):
    nome_equipamento = extrair_nome_equipamento(pergunta)
    if nome_equipamento == 'Não há equipamento mencionado na pergunta.':
        dados_filtrados = df
    else:
        dados_filtrados = busca_dados_relevantes_equipamento(nome_equipamento)

    # if dados_filtrados.empty:
    #     return "Equipamento não encontrado na base"

    dados_texto = dados_filtrados.to_string(index=False)
    data_hoje = datetime.now().strftime("%d/%m/%Y")
    prompt = f"""
                Você é um assistente de manutenção de máquinas.Você devera responder perguntas com base nos dados
                que tiver. Como por exemplo a última vez que foi feita a manutenção em uma máquina, a quanto tempo essa máquina nao recebe manutenções e entre outras.

                A data de hoje é {data_hoje}.                
                Aqui estão os dados de manutenção:

                {dados_texto}

                Com base nesses dados, responda de forma objetiva:
                {pergunta}
            """
    resposta_final = llm.invoke(prompt)
    return resposta_final.content

# pergunta_usuario = "quantas manutenções foram realizadas nos anos de 2024 e 2025?"
# resposta = perguntar_ia(pergunta_usuario)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.container():
    st.title("🛠 Chatbot de Manutenção")

chat_container = st.container()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Digite sua pergunta ou instrução..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    resposta = perguntar_ia(prompt)
    st.session_state.messages.append({"role": "assistant", "content": resposta})
    with st.chat_message("assistant"):
        st.markdown(resposta)