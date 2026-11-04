import streamlit as st
from openai import AzureOpenAI
import tempfile
import os

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Assistente Virtual de Orientação",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Assistente Virtual Inteligente (RAG)")
st.markdown("---")

# ==========================================
# 1. GERENCIAMENTO DE ESTADO (Session State)
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_context" not in st.session_state:
    st.session_state.rag_context = ""

# ==========================================
# 2. ENTRADAS (Sidebar & Configurações)
# ==========================================
with st.sidebar:
    st.header("⚙️ Configurações de Conexão")
    
    # Mantém os valores salvos no session_state para o botão "Novo Chat"
    api_key_input = st.text_input("API Key (Azure OpenAI)", type="password", value=st.session_state.get("api_key", ""))
    endpoint_input = st.text_input("End Point (URL)", value=st.session_state.get("endpoint", ""))
    model_input = st.text_input("Nome do Modelo / Deployment", value=st.session_state.get("model", "gpt-4o"))
    
    st.markdown("---")
    st.header("👤 Persona da IA")
    default_persona = (
        "Você é um assistente virtual informativo. Forneça respostas claras, objetivas e educadas. "
        "⚠️ AVISO IMPORTANTE: Você não substitui o diagnóstico de um profissional de saúde qualificado."
    )
    persona_input = st.text_area("Instruções do Sistema (System Prompt)", value=st.session_state.get("persona", default_persona))

    st.markdown("---")
    st.header("📁 Upload de Documentos (RAG)")
    uploaded_file = st.file_uploader("Envie arquivos (PDF, PNG, JPG)", type=["pdf", "png", "jpg", "jpeg"])

    # Botão para validar/salvar conexão
    if st.button("Salvar Configurações e Iniciar"):
        if api_key_input and endpoint_input and model_input:
            st.session_state.api_key = api_key_input
            st.session_state.endpoint = endpoint_input
            st.session_state.model = model_input
            st.session_state.persona = persona_input
            st.session_state.authenticated = True
            st.success("Conexão configurada com sucesso!")
        else:
            st.error("Preencha todos os campos de conexão (API, Endpoint e Modelo).")

    st.markdown("---")
    # Botão de Novo Chat (mantém as credenciais e persona, limpa apenas o histórico de conversa)
    if st.button("🔄 Iniciar Novo Chat"):
        st.session_state.messages = []
        st.session_state.rag_context = ""
        st.rerun()

# ==========================================
# 3. PROCESSAMENTO DE ARQUIVOS (RAG SIMPLES)
# ==========================================
if uploaded_file is not None and not st.session_state.rag_context:
    with st.spinner("Processando documento para RAG..."):
        # Salva o arquivo temporariamente para leitura
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Exemplo básico de extração de texto para PDF usando pypdf (se instalado)
            if uploaded_file.type == "application/pdf":
                import pypdf
                reader = pypdf.PdfReader(tmp_path)
                text_content = ""
                for page in reader.pages:
                    text_content += page.extract_text() or ""
                st.session_state.rag_context = f"\n\n[Contexto do Documento RAG - {uploaded_file.name}]:\n{text_content[:4000]}"
                st.success("PDF processado e integrado ao contexto com sucesso!")
            else:
                # Para imagens (PNG/JPG), pode-se utilizar descrição ou OCR se necessário
                st.session_state.rag_context = f"\n\n[Contexto do Arquivo de Imagem anexado]: {uploaded_file.name}"
                st.success("Imagem anexada ao contexto com sucesso!")
        except Exception as e:
            st.warning(f"Não foi possível extrair texto automaticamente do arquivo: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# ==========================================
# 4. PROCESSAMENTO DA CONVERSA E SAÍDA
# ==========================================
if not st.session_state.authenticated:
    st.info("👈 Insira suas credenciais da Azure na barra lateral e clique em **Salvar Configurações e Iniciar** para começar.")
else:
    # Exibe o histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada de pergunta do usuário
    if user_prompt := st.chat_input("Digite sua dúvida aqui..."):
        # Adiciona a mensagem do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Processamento via Azure OpenAI
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    # Inicializa o cliente Azure OpenAI
                    client = AzureOpenAI(
                        api_key=st.session_state.api_key,
                        api_version="2024-02-01",  # Ajuste a versão da API conforme necessário
                        azure_endpoint=st.session_state.endpoint
                    )

                    # Monta as mensagens incluindo a persona e o contexto do RAG, se houver
                    system_content = st.session_state.persona
                    if st.session_state.rag_context:
                        system_content += st.session_state.rag_context

                    messages_payload = [{"role": "system", "content": system_content}]
                    
                    # Adiciona o histórico de conversas anterior
                    for m in st.session_state.messages:
                        messages_payload.append({"role": m["role"], "content": m["content"]})

                    # Chamada ao modelo na nuvem
                    response = client.chat.completions.create(
                        model=st.session_state.model,
                        messages=messages_payload,
                        temperature=0.7,
                        max_tokens=1000
                    )

                    assistant_response = response.choices[0].message.content
                    st.markdown(assistant_response)
                    
                    # Salva a resposta no histórico
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                except Exception as e:
                    st.error(f"Erro ao comunicar com a Azure OpenAI: {e}")
