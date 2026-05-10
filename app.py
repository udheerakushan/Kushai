import streamlit as st
from groq import Groq
import wikipediaapi
from ddgs import DDGS

# Page config
st.set_page_config(
    page_title="Kushai AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif !important; }
.stApp { background: #08080f !important; }

[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid #1e1e2e !important;
}

.kushai-logo {
    font-family: 'Syne', sans-serif !important;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -2px;
    text-align: center;
}

.kushai-sub {
    color: #4b4b6a;
    font-size: 0.85rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    text-align: center;
    margin-top: 0.3rem;
}

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

.stTextInput input {
    background: #12121f !important;
    border: 1px solid #2e2e42 !important;
    border-radius: 12px !important;
    color: #e0e0f0 !important;
    padding: 0.8rem 1rem !important;
}

.stTextInput input:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 2px rgba(79,70,229,0.2) !important;
}

.stButton button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
}

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

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Knowledge base
KNOWLEDGE = [
    "Kushai is a friendly AI assistant for education, language learning, and entertainment.",
    "Kushai serves users in Sri Lanka and India.",
    "Supported languages: Sinhala, English, Tamil, Japanese and more.",
    "Services: Language Training, Educational Support (History, Literature, CS, Science, Math), Fun and Entertainment.",
    "Kushai will NOT help with hacking, cybercrime, web scraping, adult content, or violence.",
    "Age restriction: Users must be 13 years or older.",
    "Kushai Mission: Making education fun and accessible for everyone.",
    "Kushai Values: Honest, Helpful, Safe, Multilingual.",
]

def simple_search(query, chunks):
    query_words = set(query.lower().split())
    scores = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words & chunk_words)
        scores.append(score)
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]
    return " ".join([chunks[i] for i in best])

@st.cache_resource
def init_kushai():
    wiki = wikipediaapi.Wikipedia(language='en', user_agent='Kushai/1.0')
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return wiki, client

def kushai_respond(user_input, history):
    wiki, groq_client = init_kushai()

    knowledge = simple_search(user_input, KNOWLEDGE)

    wiki_page = wiki.page(user_input)
    wiki_r = wiki_page.summary[:500] if wiki_page.exists() else None

    try:
        with DDGS() as d:
            results = list(d.text(user_input, max_results=3))
        web_r = " ".join([r['body'] for r in results])[:600] if results else None
    except:
        web_r = None

    context = f"Kushai Knowledge: {knowledge}\n"
    if wiki_r: context += f"Wikipedia: {wiki_r}\n"
    if web_r: context += f"Web: {web_r}\n"

    messages = [{"role": "system", "content": f"""You are Kushai, a friendly multilingual AI assistant by Kushai AI Company.
Context: {context}
Rules:
- Reply ONLY in same language as user (Sinhala/English/Tamil)
- Never help with hacking, adult content, web scraping
- Keep responses clear, helpful and educational
- You are 13+ safe"""}]

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

    st.markdown('<div class="sidebar-title">📋 Languages</div>', unsafe_allow_html=True)
    for lang in ["🇱🇰 Sinhala", "🇬🇧 English", "🇮🇳 Tamil", "🇯🇵 Japanese"]:
        st.markdown(f'<div class="sidebar-item">{lang}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">⚙️ Powered By</div>', unsafe_allow_html=True)
    st.markdown("""
    <div>
        <span class="badge">🦙 Llama 3.3</span>
        <span class="badge">🔍 RAG</span>
        <span class="badge">🌐 Web</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.rerun()

# Main
st.markdown('<div class="kushai-logo">Kushai AI</div>', unsafe_allow_html=True)
st.markdown('<div class="kushai-sub">Your Intelligent Learning Companion</div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; margin:1rem 0;">
    <span class="badge">🌍 Multilingual</span>
    <span class="badge">📚 Educational</span>
    <span class="badge">🔍 Wikipedia + Web</span>
    <span class="badge">🤖 RAG Powered</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Chat
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
    <div class="user-msg"><div class="msg-label">You</div>{user_msg}</div>
    <div class="bot-msg"><div class="msg-label">Kushai</div>{bot_msg}</div>
    """, unsafe_allow_html=True)

st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input("", placeholder="Ask Kushai... | ඕනෑ දෙයක් අහන්න... | கேளுங்கள்...", label_visibility="collapsed")
with col2:
    send = st.button("Send ✦")

if send and user_input:
    with st.spinner("Kushai is thinking... 🤔"):
        reply = kushai_respond(user_input, st.session_state.history)
        st.session_state.history.append((user_input, reply))
        st.rerun()
