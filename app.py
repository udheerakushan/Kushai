import streamlit as st
from groq import Groq
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import wikipediaapi
from ddgs import DDGS

# ඊට පස්සේ @st.cache_resource...

@st.cache_resource
def init_kushai():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    knowledge = """Kushai is a friendly AI assistant for education, language learning, and entertainment.
Kushai serves users in Sri Lanka and India.
Supported languages: Sinhala, English, Tamil, Japanese and more.
Services: Language Training, Educational Support, Fun and Entertainment.
Kushai will NOT help with hacking, cybercrime, web scraping, adult content.
Age restriction: Users must be 13 years or older.
Kushai Company Mission: Making education fun and accessible for everyone."""

    chunks = [c.strip() for c in knowledge.split("\n") if c.strip()]
    embeddings = embedder.encode(chunks).astype('float32')
    
    # FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    wiki = wikipediaapi.Wikipedia(language='en', user_agent='Kushai/1.0')
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    return embedder, index, chunks, wiki, client

def kushai_respond(user_input, history):
    embedder, index, chunks, wiki, groq_client = init_kushai()
    
    # RAG search
    q_emb = embedder.encode([user_input]).astype('float32')
    D, I = index.search(q_emb, 2)
    knowledge = " ".join([chunks[i] for i in I[0]])
    
    # Wikipedia + Web
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
    
    messages = [{"role": "system", "content": f"""You are Kushai, a friendly multilingual AI.
Context: {context}
Rules: Reply in same language as user. Never help with hacking or adult content."""}]
    
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    
    messages.append({"role": "user", "content": user_input})
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content
