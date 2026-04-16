import streamlit as st
import google.generativeai as gen
import faiss
import numpy as np
import time
from groq import Groq
from markitdown import MarkItDown
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_KEY = st.secrets.get("GEMINI_KEY")
GROQ_KEY = st.secrets.get("GROQ_KEY")

if not GEMINI_KEY or not GROQ_KEY:
    st.error("API keys not found")
    st.stop()

MAX_CHUNKS = 100

st.set_page_config(page_title="DocTalk", page_icon="📄", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { font-family: 'Inter', sans-serif; box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main,
.main .block-container {
    background: #111 !important;  /* Set background to black */
    color: #fff !important;       /* Set text color to white */
}

[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {
    background: #111 !important;  /* Set background to black */
    border-top: 1px solid #444 !important;  /* Darker border */
    padding: 0.75rem 1rem !important;
}

[data-testid="stSidebar"] {
    background: #222 !important;  /* Set sidebar background to dark gray */
    border-right: 1px solid #444 !important; /* Darker border */
}

[data-testid="stSidebar"] section { padding: 1.5rem 1rem !important; }

[data-testid="stFileUploader"] {
    background: #333 !important;  /* Darker file uploader background */
    border: 1.5px dashed #666 !important;  /* Lighter dashed border */
    border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #aaa !important; }

[data-testid="stChatInput"] {
    background: #333 !important;  /* Dark input box */
    border: 1.5px solid #444 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #fff !important;  /* White text */
    font-size: 0.93rem !important;
}
[data-testid="stChatInput"] textarea:focus { box-shadow: none !important; }
[data-testid="stChatInput"] button {
    background: #444 !important;  /* Dark button background */
    border-radius: 7px !important;
    color: #fff !important;
}

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.35rem 0 !important;
}
[data-testid="stChatMessage"][data-testid*="user"] [data-testid="stChatMessageContent"] {
    background: #333 !important;  /* Dark background for user messages */
    border: 1px solid #444 !important;
    border-radius: 10px !important;
    padding: 0.65rem 0.9rem !important;
}
[data-testid="stChatMessage"][data-testid*="assistant"] [data-testid="stChatMessageContent"] {
    background: #222 !important;  /* Darker background for assistant messages */
}

.stButton > button {
    background: #444 !important;  /* Dark button background */
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    width: 100% !important;
    transition: background 0.15s;
}
.stButton > button:hover { background: #666 !important; }

.stAlert { border-radius: 10px !important; font-size: 0.88rem !important; }

.block-container { padding: 2rem 2rem 0 2rem !important; max-width: 780px !important; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

gen.configure(api_key=GEMINI_KEY)

def embedd(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            res = gen.embed_content(
                model='models/gemini-embedding-001', 
                content=text, 
                task_type='retrieval_query'
            )
            return np.array(res["embedding"], dtype="float32")
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to generate embedding after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt) 

def split_chunks(text, chunk_size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text) and len(chunks) < MAX_CHUNKS:
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    if len(chunks) >= MAX_CHUNKS:
        st.warning(f"Document truncated to {MAX_CHUNKS} chunks due to size limits.")
    
    return chunks

def build_index(chunks):
    embeddings = []
    for c in chunks:
        emb = embedd(c)
        if emb is not None: 
            embeddings.append(emb)
    
    if not embeddings:
        raise ValueError("No embeddings generated. Check the embedd function.")
    
    matrix = np.stack(embeddings)
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index    

def read_pdf(file_obj):
    # reader = PdfReader(file_obj)
    # text = "\n".join(p.extract_text() or "" for p in reader.pages)
    md = MarkItDown()
    result = md.convert(file_obj)
    text = result.text_content
    if not text or not text.strip():
        raise ValueError("No text extracted from the Document.")
    return text

def search(question, index, chunks, top_k=4):
    q_emb = embedd(question).reshape(1, -1)
    _, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0]]

def ask(history, context):
    client = Groq(api_key=GROQ_KEY)
    system = f"""You answer questions about a Document. Use only the context below.
Context: {context}"""
    messages = [{"role": "system", "content": system}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    res = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
    return res.choices[0].message.content

for k, v in {"messages": [], "index": None, "chunks": None, "pdf_name": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar:
    st.markdown("### DocTalk")
    st.caption("Ask questions about any Document.")
    st.divider()
    st.markdown("**Upload a Document**")

    uploaded = st.file_uploader("", type=["pdf", "docx", "doc", "txt", "rtf", "odt"], label_visibility="collapsed")

    if uploaded and uploaded.name != st.session_state.pdf_name:

        if uploaded.size > 10 * 1024 * 1024:
            st.error("File too large. Please upload a file smaller than 10MB.")
        elif uploaded.name != st.session_state.pdf_name:
            with st.spinner("Indexing…"):
                try:
                    text = read_pdf(uploaded)
                    chunks = split_chunks(text) 
                    index = build_index(chunks)
                    
                    st.session_state.update(index=index, chunks=chunks, pdf_name=uploaded.name, messages=[])
                    st.success("Ready!")
                    
                except ValueError as e:
                    # This catches the "No text extracted" error and shows it nicely
                    st.error(f"{e}")
                except Exception as e:
                    # This catches any other unexpected errors
                    st.error(f"An unexpected error occurred: {e}")

    if st.session_state.pdf_name:
        st.caption(f"**{st.session_state.pdf_name}**")
        st.divider()
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

if not st.session_state.index:
    st.markdown("## Ask anything about your Document")
    st.caption("Upload a document in the sidebar to get started.")
else:
    if not st.session_state.messages:
        st.markdown("## Ready — ask away")
        st.caption(f"Loaded: **{st.session_state.pdf_name}**")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Type your question…", disabled=st.session_state.index is None):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            results = search(prompt, st.session_state.index, st.session_state.chunks)
            context = "\n\n".join(results)
            answer  = ask(st.session_state.messages, context)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
