import streamlit as st
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
import wikipediaapi
from ddgs import DDGS
import os

# Page config
st.set_page_config(
    page_title="Kushai AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Global */
* { font-family: 'DM Sans', sans-serif !important; }
.stApp { background: #08080f !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid #1e1e2e !important;
}

/* Main header */
.kushai-header {
    text-align: center;
    padding: 2rem 0 1rem;
}

.kushai-logo {
    font-family: 'Syne', sans-serif !important;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -2px;
}

.kushai-sub {
    color: #4b4b6a;
    font-size: 0.85rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* Chat messages */
.user-msg {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 18px 18px 4px 18px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0 0.5rem 3rem;
    color: white;
    font-size: 0.95rem;
    box-shadow: 0 4px 20px rgba(79,70,229,0.3);
}

.bot-msg {
    background: #12121f;
    border: 1px solid #1e1e2e;
    border-radius: 18px 18px 18px 4px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 3rem 0.5rem 0;
    color: #e0e0f0;
    font-size: 0.95rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.msg-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
    opacity: 0.5;
}

/* Input */
.stTextInput input {
    background: #12121f !important;
    border: 1px solid #2e2e42 !important;
    border-radius: 12px !important;
    color: #e0e0f0 !important;
    padding: 0.8rem 1rem !important;
    font-size: 0.95rem !important;
}

.stTextInput input:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 2px rgba(79,70,229,0.2) !important;
}

/* Send button */
.stButton button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.stButton button:hover { opacity: 0.85 !important; }

/* Badge */
.badge {
    display: inline-block;
    background: #12121f;
    border: 1px solid #2e2e42;
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.7rem;
    color: #818cf8;
    margin: 0.2rem;
}

/* Sidebar items */
.sidebar-item {
    background: #12121f;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #e0e0f0;
    font-size: 0.85rem;
}

.sidebar-title {
    color: #4b4b6a;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 1rem 0 0.5rem;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Init
@st.cache_resource
def init_kushai():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    chroma = chromadb.Client()
    
    try:
        collection = chroma.get_collection("kushai")
    except:
        collection = chroma.create_collection("kushai")
        knowledge = """
Kushai is a friendly AI assistant for education, language learning, and entertainment.
Kushai serves users in Sri Lanka and India.
Supported languages: Sinhala, English, Tamil, Japanese and more.
Services: Language Training, Educational Support (History, Literature, CS, Science, Math), Fun and Entertainment.
Kushai will NOT help with hacking, cybercrime, web scraping, adult content, or violence.
Age restriction: Users must be 13 years or older.
User conversations are saved to improve Kushai anonymously.
Kushai Company Mission: Making education fun and accessible for everyone.
Values: Honest, Helpful, Safe, Multilingual.
        """
        chunks = [c.strip() for c in knowledge.split("\n") if c.strip()]
        collection.add(
            documents=chunks,
            embeddings=embedder.encode(chunks).tolist(),
            ids=[f"c_{i}" for i in range(len(chunks))]
        )
    
    wiki = wikipediaapi.Wikipedia(language='en', user_agent='Kushai/1.0')
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    return embedder, collection, wiki, client

embedder, collection, wiki, groq_client = init_kushai()

def search_wiki(query):
    page = wiki.page(query)
    return page.summary[:500] if page.exists() else None

def search_web(query):
    try:
        with DDGS() as d:
            results = list(d.text(query, max_results=3))
        return " ".join([r['body'] for r in results])[:600] if results else None
    except:
        return None

def kushai_respond(user_input, history):
    # RAG search
    q_emb = embedder.encode([user_input]).tolist()
    rag = collection.query(query_embeddings=q_emb, n_results=2)
    knowledge = " ".join(rag['documents'][0])
    
    wiki_r = search_wiki(user_input)
    web_r = search_web(user_input)
    
    context = f"Kushai Knowledge: {knowledge}\n"
    if wiki_r: context += f"Wikipedia: {wiki_r}\n"
    if web_r: context += f"Web: {web_r}\n"
    
    messages = [{"role": "system", "content": f"""You are Kushai, a friendly multilingual AI assistant by Kushai AI Company.
Context: {context}
Rules: Reply in same language as user. Never help with hacking, adult content. Keep responses clear and helpful."""}]
    
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    
    messages.append({"role": "user", "content": user_input})
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 1rem;">
        <div style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800;
                    background:linear-gradient(135deg,#a5b4fc,#c084fc);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            Kushai AI
        </div>
        <div style="color:#4b4b6a; font-size:0.65rem; letter-spacing:2px; text-transform:uppercase;">
            Enterprise Edition
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-title">⚡ Capabilities</div>', unsafe_allow_html=True)
    for cap in ["🌍 Multilingual Support", "📚 Educational RAG", "🔍 Wikipedia + Web", "🎓 Language Training", "🎮 Fun & Entertainment"]:
        st.markdown(f'<div class="sidebar-item">{cap}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-title">📋 Supported Languages</div>', unsafe_allow_html=True)
    for lang in ["🇱🇰 Sinhala", "🇬🇧 English", "🇮🇳 Tamil", "🇯🇵 Japanese"]:
        st.markdown(f'<div class="sidebar-item">{lang}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-title">⚙️ Powered By</div>', unsafe_allow_html=True)
    st.markdown("""
    <div>
        <span class="badge">🦙 Llama 3.3</span>
        <span class="badge">🧠 RAG</span>
        <span class="badge">🔍 ChromaDB</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.rerun()

# Main
st.markdown("""
<div class="kushai-header">
    <div class="kushai-logo">Kushai AI</div>
    <div class="kushai-sub">Your Intelligent Learning Companion</div>
    <div style="margin-top:1rem;">
        <span class="badge">🌍 Multilingual</span>
        <span class="badge">📚 Educational</span>
        <span class="badge">🔍 Wikipedia + Web</span>
        <span class="badge">🤖 RAG Powered</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Chat history
chat_container = st.container()
with chat_container:
    if not st.session_state.history:
        st.markdown("""
        <div style="text-align:center; padding:3rem 0; color:#2e2e42;">
            <div style="font-size:3rem;">🤖</div>
            <div style="font-size:1.1rem; margin-top:1rem; color:#4b4b6a;">
                Hello! I'm Kushai. Ask me anything!
            </div>
            <div style="font-size:0.85rem; margin-top:0.5rem; color:#2e2e42;">
                History · Literature · CS · Languages · Fun
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    for user_msg, bot_msg in st.session_state.history:
        st.markdown(f"""
        <div class="user-msg">
            <div class="msg-label">You</div>
            {user_msg}
        </div>
        <div class="bot-msg">
            <div class="msg-label">Kushai</div>
            {bot_msg}
        </div>
        """, unsafe_allow_html=True)

# Input
st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input("", placeholder="Ask Kushai anything... | ඕනෑ දෙයක් අහන්න... | எதையும் கேளுங்கள்...", label_visibility="collapsed")
with col2:
    send = st.button("Send ✦")

if send and user_input:
    with st.spinner("Kushai is thinking..."):
        reply = kushai_respond(user_input, st.session_state.history)
        st.session_state.history.append((user_input, reply))
        st.rerun()
