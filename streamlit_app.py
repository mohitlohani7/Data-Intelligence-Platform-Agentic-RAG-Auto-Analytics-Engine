import os
import streamlit as st
from typing import List
from dotenv import load_dotenv

load_dotenv()

# ================= CORE LANGCHAIN =================
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ================= LLM PROVIDERS =================
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

# ================= RAG COMPONENTS =================
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ================= UI & APP STATE =================
st.set_page_config(
    page_title="Data Intelligence Platform — Enterprise RAG",
    layout="wide",
    page_icon="🧠"
)

# ================= PREMIUM CSS =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background-color: #060b17;
        background-image:
            radial-gradient(ellipse at 10% 10%, rgba(78, 205, 196, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 90% 80%, rgba(139, 92, 246, 0.06) 0%, transparent 50%);
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
    }

    h1 {
        background: linear-gradient(135deg, #A0FFED 0%, #4ECDC4 40%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 16, 30, 0.98) 0%, rgba(6, 11, 23, 0.98) 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.04);
    }

    .stButton > button {
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
        color: #0b0f19;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.95rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(78, 205, 196, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(78, 205, 196, 0.5);
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(15, 23, 42, 0.8);
        padding: 0.4rem;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #64748b;
        font-weight: 600;
        border: none !important;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(78, 205, 196, 0.15), rgba(139, 92, 246, 0.1)) !important;
        color: #4ECDC4 !important;
        border: 1px solid rgba(78, 205, 196, 0.2) !important;
    }

    .stChatMessage { background-color: transparent !important; }

    .feature-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(78, 205, 196, 0.15);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(20px);
    }

    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-ready { background: rgba(78,205,196,0.15); color: #4ECDC4; border: 1px solid rgba(78,205,196,0.3); }
    .badge-warn  { background: rgba(251,191,36,0.15);  color: #FBBF24; border: 1px solid rgba(251,191,36,0.3); }
</style>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
for key, default in [
    ("chat_history", []),
    ("vectorstore", None),
    ("docs_ready", False),
    ("embedding_provider", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ================= HELPERS =================
def get_embeddings(provider: str, api_key: str):
    """Return embeddings object based on provider."""
    if provider == "OpenAI":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    elif provider == "Claude":
        # Use OpenAI embeddings if key available, else fallback
        oai_key = os.getenv("OPENAI_API_KEY", "")
        if oai_key:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(api_key=oai_key, model="text-embedding-3-small")
        else:
            # Fallback: Groq doesn't do embeddings, use a simple local approach
            from langchain_community.embeddings import FakeEmbeddings
            st.warning("⚠️ No OpenAI key for embeddings — using demo mode. Add OPENAI_API_KEY for full RAG.")
            return FakeEmbeddings(size=768)
    else:  # Groq
        oai_key = os.getenv("OPENAI_API_KEY", "")
        if oai_key:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(api_key=oai_key, model="text-embedding-3-small")
        else:
            from langchain_community.embeddings import FakeEmbeddings
            st.warning("⚠️ No OpenAI key for embeddings — using demo mode. Add OPENAI_API_KEY for full RAG.")
            return FakeEmbeddings(size=768)


def get_llm(provider: str, model_name: str, api_key: str, temperature: float = 0.2):
    if not api_key:
        st.error(f"⚠️ Please provide an API key for {provider}")
        st.stop()
    if provider == "OpenAI":
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=temperature)
    elif provider == "Claude":
        return ChatAnthropic(model=model_name, api_key=api_key, temperature=temperature)
    else:  # Groq
        return ChatGroq(model=model_name, api_key=api_key, temperature=temperature)


def extract_text_from_pdfs(pdf_docs) -> str:
    text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            st.warning(f"Could not read {pdf.name}: {e}")
    return text


def build_vectorstore(text: str, provider: str, api_key: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)
    embeddings = get_embeddings(provider, api_key)
    return FAISS.from_texts(chunks, embedding=embeddings), len(chunks)


# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    provider = st.selectbox("🤖 LLM Provider", ["Groq (Free)", "OpenAI", "Claude"])

    if provider == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY", "") or st.text_input("OpenAI API Key", type="password")
        model_name = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
        prov_key = "OpenAI"
    elif provider == "Claude":
        api_key = os.getenv("ANTHROPIC_API_KEY", "") or st.text_input("Anthropic API Key", type="password")
        model_name = st.selectbox("Model", ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"])
        prov_key = "Claude"
    else:
        api_key = os.getenv("GROQ_API_KEY", "") or st.text_input("Groq API Key", type="password")
        model_name = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"])
        prov_key = "Groq"

    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.2)

    if api_key:
        st.success(f"✅ {provider} key loaded")
    else:
        st.error(f"⚠️ No {provider} key found")

    st.markdown("---")
    st.markdown("### 📖 Quick Guide")
    st.markdown("""
    1. Select your LLM provider above
    2. Upload PDFs in the RAG Chat tab
    3. Click **Vectorize** to index them
    4. Ask questions!
    
    > 💡 **Tip:** For RAG embeddings, add an OpenAI API key in Streamlit Secrets even if using Groq/Claude for chat.
    """)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#475569; text-align:center;'>
    🔒 Keys are stored in Streamlit Secrets<br>Never exposed in code or GitHub
    </div>
    """, unsafe_allow_html=True)


# ================= MAIN UI =================
st.title("Data Intelligence Platform")
st.markdown('<p style="color:#64748b; margin-top:-0.5rem; margin-bottom:1.5rem;">Enterprise Agentic RAG · Multi-LLM · Semantic Search</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📚 Document RAG Chat", "ℹ️ About"])

# ====================== TAB 1: RAG ======================
with tab1:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### 📁 Knowledge Base")
        pdf_docs = st.file_uploader(
            "Upload PDFs to vectorize",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to chat with"
        )

        if st.button("⚡ Vectorize Documents", key="vectorize_btn"):
            if not pdf_docs:
                st.warning("Upload at least one PDF first.")
            elif not api_key:
                st.error(f"Enter your {provider} API key first.")
            else:
                with st.spinner("📄 Reading & embedding PDFs..."):
                    try:
                        raw_text = extract_text_from_pdfs(pdf_docs)
                        if not raw_text.strip():
                            st.error("Could not extract text from PDFs.")
                        else:
                            vs, n_chunks = build_vectorstore(raw_text, prov_key, api_key)
                            st.session_state.vectorstore = vs
                            st.session_state.docs_ready = True
                            st.session_state.chat_history = []
                            st.success(f"✅ {n_chunks} chunks indexed from {len(pdf_docs)} PDF(s)!")
                    except Exception as e:
                        st.error(f"Vectorization error: {e}")

        if st.session_state.docs_ready:
            st.markdown('<span class="status-badge badge-ready">🟢 Knowledge Base Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-warn">🟡 No Documents Indexed</span>', unsafe_allow_html=True)

        if st.session_state.docs_ready and st.button("🗑️ Clear & Reset", key="clear_btn"):
            st.session_state.vectorstore = None
            st.session_state.docs_ready = False
            st.session_state.chat_history = []
            st.rerun()

    with col_right:
        st.markdown("#### 💬 RAG Chat")

        if not st.session_state.docs_ready:
            st.info("👈 Upload and vectorize PDFs first to start chatting.")
        else:
            # Chat history display
            chat_container = st.container(height=420)
            with chat_container:
                if not st.session_state.chat_history:
                    st.markdown("""
                    <div style='text-align:center; padding:3rem; color:#475569;'>
                        <div style='font-size:2rem; margin-bottom:0.5rem;'>🧠</div>
                        <div>Documents indexed. Ask me anything!</div>
                    </div>
                    """, unsafe_allow_html=True)
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            user_input = st.chat_input("Ask anything about your documents...", key="rag_input")

            if user_input:
                if not api_key:
                    st.error("No API key found.")
                else:
                    st.session_state.chat_history.append({"role": "user", "content": user_input})

                    with st.spinner(f"🤔 Thinking with {provider}..."):
                        try:
                            llm = get_llm(prov_key, model_name, api_key, temperature)
                            retriever = st.session_state.vectorstore.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": 5}
                            )

                            # Build retrieval chain (no agents — more compatible)
                            system_prompt = (
                                "You are an expert document analyst. Use the following retrieved context "
                                "from the user's documents to answer their question accurately and concisely. "
                                "If the answer is not in the context, say so clearly.\n\n"
                                "Context:\n{context}"
                            )
                            qa_prompt = ChatPromptTemplate.from_messages([
                                ("system", system_prompt),
                                ("human", "{input}"),
                            ])

                            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                            response = rag_chain.invoke({"input": user_input})
                            answer = response.get("answer", "No response generated.")

                            st.session_state.chat_history.append({"role": "assistant", "content": answer})
                            st.rerun()

                        except Exception as e:
                            err_msg = f"❌ Error: {str(e)}"
                            st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
                            st.rerun()

# ====================== TAB 2: ABOUT ======================
with tab2:
    st.markdown("## 🧠 About This Platform")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color:#4ECDC4;">📚 Multi-LLM RAG</h4>
            <p style="color:#94a3b8; font-size:0.9rem;">
            Upload any PDF and chat with it using GPT-4o, Claude, or Groq Llama-3.
            Powered by FAISS vector search + semantic retrieval.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color:#8B5CF6;">🔐 Secure by Design</h4>
            <p style="color:#94a3b8; font-size:0.9rem;">
            API keys stored in Streamlit Secrets — never committed to GitHub.
            .env file excluded from version control.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color:#FF6B6B;">⚡ Enterprise Stack</h4>
            <p style="color:#94a3b8; font-size:0.9rem;">
            LangChain · FAISS · OpenAI Embeddings · Groq · Anthropic.
            Production-grade retrieval pipeline with semantic chunking.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### 🛠️ Tech Stack
    | Component | Technology |
    |-----------|-----------|
    | **Frontend** | Streamlit |
    | **RAG Framework** | LangChain |
    | **Vector Store** | FAISS (CPU) |
    | **Embeddings** | OpenAI text-embedding-3-small |
    | **LLM Providers** | OpenAI GPT-4o / Anthropic Claude / Groq Llama-3 |
    | **PDF Parsing** | pypdf |
    | **Deployment** | Streamlit Cloud |
    """)
