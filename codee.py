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
GEMINI_KEY = os.getenv('GEMINI_KEY')
GROQ_KEY = os.getenv('GROQ_KEY')

if not GEMINI_KEY or not GROQ_KEY:
    st.error("API keys not found")
    st.stop()

MAX_CHUNKS_PER_DOC = 50  # Reduced to avoid rate limits

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

def embedd(text, max_retries=5):
    """Generate embeddings with rate limiting and retry logic"""
    for attempt in range(max_retries):
        try:
            # Add a small delay between requests to avoid hitting rate limits
            if attempt > 0:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                time.sleep(0.1)  # Small delay even on first attempt
            
            res = gen.embed_content(model='models/gemini-embedding-001', content=text, task_type='retrieval_query')
            return np.array(res["embedding"], dtype="float32")
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit error
            if "429" in error_msg or "Quota exceeded" in error_msg:
                # Extract wait time from error message if available
                if "retry in" in error_msg.lower():
                    import re
                    match = re.search(r'retry in (\d+)', error_msg)
                    if match:
                        wait_time = int(match.group(1)) + 1
                    else:
                        wait_time = 60
                else:
                    wait_time = 60
                
                if attempt < max_retries - 1:
                    st.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
            
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to generate embedding after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt) 

def split_chunks(text, chunk_size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text) and len(chunks) < MAX_CHUNKS_PER_DOC:
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    if len(chunks) >= MAX_CHUNKS_PER_DOC:
        st.warning(f"Document truncated to {MAX_CHUNKS_PER_DOC} chunks due to size limits.")
    
    return chunks

def build_index(all_chunks):
    """Build a single FAISS index from all document chunks with progress tracking"""
    embeddings = []
    total = len(all_chunks)
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, c in enumerate(all_chunks):
        status_text.text(f"Processing chunk {idx + 1}/{total}...")
        progress_bar.progress((idx + 1) / total)
        
        emb = embedd(c)
        if emb is not None: 
            embeddings.append(emb)
        
        # Add a small delay every 10 chunks to avoid rate limits
        if (idx + 1) % 10 == 0:
            time.sleep(1)
    
    progress_bar.empty()
    status_text.empty()
    
    if not embeddings:
        raise ValueError("No embeddings generated. Check the embedd function.")
    
    matrix = np.stack(embeddings)
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index    

def read_pdf(file_obj):
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

def ask(history, context, doc_names):
    client = Groq(api_key=GROQ_KEY)
    doc_list = ", ".join(doc_names)
    system = f"""You answer questions about documents. Use only the context below.
Documents loaded: {doc_list}
Context: {context}"""
    messages = [{"role": "system", "content": system}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    res = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
    return res.choices[0].message.content

# Initialize session state
for k, v in {"messages": [], "index": None, "chunks": None, "doc_names": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar:
    st.markdown("### DocTalk")
    st.caption("Ask questions about multiple documents.")
    st.divider()
    st.markdown("**Upload Documents**")

    uploaded_files = st.file_uploader(
        "", 
        type=["pdf", "docx", "doc", "txt", "rtf", "odt"], 
        label_visibility="collapsed",
        accept_multiple_files=True
    )

    if uploaded_files:
        current_names = [f.name for f in uploaded_files]
        
        # Check if the uploaded files have changed
        if current_names != st.session_state.doc_names:
            # Check file sizes
            oversized = [f.name for f in uploaded_files if f.size > 10 * 1024 * 1024]
            if oversized:
                st.error(f"Files too large: {', '.join(oversized)}. Max 10MB per file.")
            else:
                with st.spinner("Indexing documents…"):
                    try:
                        all_chunks = []
                        
                        # Process each document
                        for uploaded_file in uploaded_files:
                            text = read_pdf(uploaded_file)
                            chunks = split_chunks(text)
                            all_chunks.extend(chunks)
                        
                        # Build a single index from all chunks
                        index = build_index(all_chunks)
                        
                        st.session_state.update(
                            index=index, 
                            chunks=all_chunks, 
                            doc_names=current_names, 
                            messages=[]
                        )
                        st.success(f"Indexed {len(uploaded_files)} document(s)!")
                        
                    except ValueError as e:
                        st.error(f"{e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

    if st.session_state.doc_names:
        st.caption("**Loaded documents:**")
        for name in st.session_state.doc_names:
            st.caption(f"• {name}")
        st.divider()
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

if not st.session_state.index:
    st.markdown("## Ask anything about your documents")
    st.caption("Upload one or more documents in the sidebar to get started.")
else:
    if not st.session_state.messages:
        st.markdown("## Ready — ask away")
        st.caption(f"Loaded **{len(st.session_state.doc_names)}** document(s)")

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
            answer = ask(st.session_state.messages, context, st.session_state.doc_names)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
