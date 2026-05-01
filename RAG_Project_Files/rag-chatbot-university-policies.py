# streamlit_faiss_qa_app.py
# Streamlit UI wrapper for building a FAISS-backed document QA index
# based on the user's part-1 and part-2 code.
# Usage: pip install -r requirements.txt (see README) then:
# streamlit run streamlit_faiss_qa_app.py

import streamlit as st
import tempfile, os, re, glob, pickle, io, math, time
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

# 3rd-party libs used by the backend
import pdfplumber, docx
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss

st.set_page_config(page_title="Doc QA (FAISS)", layout="wide")

# ------------------------- Utilities (extracted/adapted) -------------------------

def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\xa0", " ")
    s = re.sub(r"-\n(?=[a-z])", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def _table_to_sentences(table: List[List[str]]) -> List[str]:
    if not table or len(table) < 2:
        return []
    header = [(c.strip() if c else f"col{i}") for i, c in enumerate(table[0])]
    rows = []
    for r in table[1:]:
        r = [(c.strip() if c else "") for c in r]
        if len(r) < len(header):
            r += [""] * (len(header) - len(r))
        rows.append(r)
    try:
        df = pd.DataFrame(rows, columns=header)
        sentences = []
        for _, row in df.iterrows():
            if len(df.columns) >= 2 and str(row.iloc[0]).isdigit() and str(row.iloc[1]).isdigit():
                sentences.append(f"In a {int(row.iloc[0])}-credit course, students are allowed {int(row.iloc[1])} absences.")
                continue
            parts = []
            for col in df.columns:
                val = str(row[col]).strip()
                if val and val.lower() not in ("nan", "none"):
                    parts.append(f"{col}: {val}")
            if parts:
                sentences.append("; ".join(parts) + ".")
        return sentences
    except Exception:
        out = []
        for r in rows:
            parts = [f"{header[i]}: {r[i]}" for i in range(len(header)) if r[i]]
            if parts:
                out.append("; ".join(parts) + ".")
        return out


def extract_pdf_bytes(file_bytes: bytes) -> Tuple[str, List[Dict]]:
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = _clean_text(page.extract_text() or "")
            row_sents = []
            try:
                tables = page.extract_tables()
                for t in tables or []:
                    row_sents.extend(_table_to_sentences(t))
            except Exception:
                pass
            if row_sents:
                txt = (txt + "\n\n" + "\n".join(row_sents)).strip()
            pages.append({"page": i, "text": txt})
    all_text = "\n\n".join([_clean_text(p["text"]) for p in pages if p["text"].strip()])
    return all_text, pages


def extract_docx_bytes(file_bytes: bytes) -> Tuple[str, List[Dict]]:
    doc = docx.Document(io.BytesIO(file_bytes))
    paras = [p.text for p in doc.paragraphs]
    paras = [_clean_text(p) for p in paras if _clean_text(p)]
    text = "\n\n".join(paras)
    return text, [{"page": 1, "text": text}]


def extract_any_bytes(filename: str, file_bytes: bytes) -> Tuple[str, List[Dict]]:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_pdf_bytes(file_bytes)
    elif ext == ".docx":
        return extract_docx_bytes(file_bytes)
    else:
        raise ValueError("Unsupported file type: " + ext)


def word_overlap_chunks(text: str, target_words=200, overlap_words=50) -> List[Tuple[int, int, str]]:
    words = text.split()
    if target_words <= 0:
        raise ValueError("target_words must be > 0")
    step = target_words - overlap_words
    if step <= 0:
        raise ValueError("overlap_words must be < target_words")
    chunks = []
    start = 0
    while start < len(words):
        end = start + target_words
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()
        chunks.append((start, min(end, len(words)), chunk_text))
        start += step
        if start >= len(words):
            break
    return chunks


EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

class Embedder:
    def __init__(self, model_name=EMBED_MODEL_NAME, device=None):
        st.info(f"Loading embedder: {model_name} ...")
        self.model = SentenceTransformer(model_name, device=device)
    def encode(self, texts: List[str], batch_size=32) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size))

class FaissStore:
    def __init__(self, dim:int):
        self.index = faiss.IndexFlatIP(dim)
        self.meta = []
    def add(self, vectors: np.ndarray, metadata: List[Dict]):
        self.index.add(vectors.astype(np.float32))
        self.meta.extend(metadata)
    def search(self, q_vec: np.ndarray, top_k=10):
        scores, idxs = self.index.search(q_vec.astype(np.float32), top_k)
        out = []
        for i in range(top_k):
            idx = idxs[0][i]
            if idx == -1:
                continue
            out.append((float(scores[0][i]), self.meta[idx]))
        return sorted(out, key=lambda x: -x[0])
    def save(self, index_path, meta_path):
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.meta, f)
    def load(self, index_path, meta_path):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)


# small helper to build prompt and answer via flan
_flan_pipe = None

def get_flan_pipe():
    global _flan_pipe
    if _flan_pipe is None:
        _flan_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=120, do_sample=False)
    return _flan_pipe


def build_prompt(query: str, contexts: List[str]) -> str:
    ctx = "\n\n---\n\n".join([f"[Source {i+1}]\n{c}" for i,c in enumerate(contexts)])
    return f"""Answer the question ONLY from the context below.\nIf the answer is not present, say \"I don't know\".\nBe concise (1-2 sentences).\n\nContext:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"""


def generate_answer(query: str, hits: List[Tuple[float, Dict]], min_score=0.0, max_contexts=6) -> str:
    filtered = [h for h in hits if h[0] >= min_score]
    if not filtered:
        return "I don't know."
    contexts = [h[1]["preview"] if len(h[1]["preview"])<1000 else h[1]["preview"][:1000] for h in filtered[:max_contexts]]
    prompt = build_prompt(query, contexts)
    pipe = get_flan_pipe()
    out = pipe(prompt)[0]["generated_text"].strip()
    return re.sub(r"\s+", " ", out)


# ------------------------- Core index-building (streamlit-friendly) -------------------------

@st.cache_data(show_spinner=False)
def collect_files_from_path(path: str) -> List[str]:
    file_list = glob.glob(os.path.join(path, "**/*.[Pp][Dd][Ff]"), recursive=True) + \
                glob.glob(os.path.join(path, "**/*.[Dd][Oo][Cc][Xx]"), recursive=True)
    return file_list


def build_store_from_uploaded(uploaded_files, target_words, overlap_words, batch_size, dedupe=True):
    # returns store, emb
    emb = Embedder(EMBED_MODEL_NAME)
    all_texts = []
    all_meta = []
    total_files = len(uploaded_files)
    for i, uf in enumerate(uploaded_files, start=1):
        st.write(f"[{i}/{total_files}] Processing uploaded file: {uf.name}")
        try:
            data = uf.read()
            text, pages = extract_any_bytes(uf.name, data)
        except Exception as e:
            st.warning(f"Failed to extract {uf.name}: {e}")
            continue
        if not text.strip():
            st.info("Empty text, skipping")
            continue
        chunks = word_overlap_chunks(text, target_words=target_words, overlap_words=overlap_words)
        for cid, (s,e,chunk_text) in enumerate(chunks):
            meta = {
                "filename": uf.name,
                "chunk_id": f"{uf.name}_chunk_{cid}",
                "start_word": s,
                "end_word": e,
                "preview": chunk_text[:2000]
            }
            all_texts.append(chunk_text)
            all_meta.append(meta)
    if len(all_texts) == 0:
        raise ValueError("No chunks were created. Check files.")
    if dedupe:
        seen=set(); uniq_texts=[]; uniq_meta=[]
        for t,m in zip(all_texts, all_meta):
            h = hash(t)
            if h in seen:
                continue
            seen.add(h); uniq_texts.append(t); uniq_meta.append(m)
        all_texts, all_meta = uniq_texts, uniq_meta
    with st.spinner("Embedding chunks (this may take time)..."):
        vecs = emb.encode(all_texts, batch_size=batch_size)
    dim = vecs.shape[1]
    store = FaissStore(dim)
    store.add(vecs, all_meta)
    return store, emb


# ------------------------- Streamlit layout -------------------------

# ------------------------- Streamlit layout -------------------------

st.title("📚 Chatbot for University Policies and Documents")

# --- UI Toggle Button (Hide/Show Settings) ---
if "show_settings" not in st.session_state:
    st.session_state.show_settings = True

toggle_label = "🙈 Hide Settings" if st.session_state.show_settings else "⚙️ Show Settings"
if st.button(toggle_label):
    st.session_state.show_settings = not st.session_state.show_settings
    st.rerun()

# --- Chatbot-only mode (Hide Settings pressed) ---
if not st.session_state.show_settings:
    st.markdown("### 🤖 Chatbot Mode")

    # Minimal chatbot UI
    query = st.text_input("Enter your question:")
    ask_btn = st.button("Ask")

    if ask_btn:
        if "store" not in st.session_state or not st.session_state.store:
            st.error("No index present in session. Please build or load an index first.")
        else:
            with st.spinner("Generating answer..."):
                q_vec = st.session_state.emb.encode([query])
                hits = st.session_state.store.search(q_vec, top_k=6)
                ans = generate_answer(query, hits, min_score=0.0, max_contexts=6)
            st.markdown("### 🧠 Answer (FLAN-T5)")
            st.success(ans)

    st.markdown("---")
    st.caption("Simplified demo view — use 'Show Settings' to access all options.")
    st.stop()

# --- Full Settings Mode (Default) ---
st.markdown("Upload PDF/DOCX files or point to a folder on disk (server). Build an embedding index, then ask questions.")

col1, col2 = st.columns([2,1])

with col1:
    uploaded = st.file_uploader("Upload PDF/DOCX (multiple)", type=["pdf","docx"], accept_multiple_files=True)
    st.write("or specify a server-side folder (useful when running locally)")
    folder_path = st.text_input("Folder path (optional)", value="")

    st.markdown("**Indexing options**")
    target_words = st.number_input("Chunk size (words)", min_value=50, max_value=2000, value=200, step=50)
    overlap = st.number_input("Chunk overlap (words)", min_value=0, max_value=1000, value=50, step=10)
    batch_size = st.number_input("Embedding batch size", min_value=1, max_value=256, value=32, step=1)
    dedupe = st.checkbox("Dedupe chunks (hash)", value=True)
    build_btn = st.button("🔨 Build index")

    load_index = st.file_uploader("Load saved index (.index) and metadata (.pkl)", type=["index","pkl"], accept_multiple_files=True)

with col2:
    st.markdown("**Search options**")
    top_k = st.number_input("Top K results", min_value=1, max_value=50, value=12, step=1)
    min_score = st.slider("Min score filter", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    if "use_flan" not in st.session_state:
        st.session_state.use_flan = False

    st.session_state.use_flan = st.checkbox(
    "Use FLAN-T5 generator for answers (heavy)",
    value=st.session_state.use_flan
    )
    use_flan = st.session_state.use_flan
    query = st.text_input("Enter your question:")
    ask_btn = st.button("Ask")

# --- Session state initialization ---
if "store" not in st.session_state:
    st.session_state.store = None
if "emb" not in st.session_state:
    st.session_state.emb = None
if "meta_preview_samples" not in st.session_state:
    st.session_state.meta_preview_samples = []

# --- Load existing FAISS index ---
if load_index and len(load_index) == 2:
    idx_file = None
    pkl_file = None
    for f in load_index:
        if f.name.endswith('.index'):
            idx_file = f
        if f.name.endswith('.pkl'):
            pkl_file = f
    if idx_file and pkl_file:
        tmp_idx = tempfile.NamedTemporaryFile(delete=False)
        tmp_meta = tempfile.NamedTemporaryFile(delete=False)
        tmp_idx.write(idx_file.read()); tmp_idx.flush()
        tmp_meta.write(pkl_file.read()); tmp_meta.flush()
        try:
            store = FaissStore(1)
            store.load(tmp_idx.name, tmp_meta.name)
            st.session_state.store = store
            st.success("Loaded index and metadata into session.")
        except Exception as e:
            st.error(f"Failed to load index: {e}")

# --- Build index ---
if build_btn:
    try:
        if uploaded:
            store, emb = build_store_from_uploaded(uploaded, target_words=target_words,
                                                   overlap_words=overlap, batch_size=batch_size, dedupe=dedupe)
            st.session_state.store = store
            st.session_state.emb = emb
            st.success("Index built from uploaded files.")

            tmp_idx = tempfile.NamedTemporaryFile(delete=False, suffix='.index')
            tmp_meta = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
            store.save(tmp_idx.name, tmp_meta.name)
            with open(tmp_idx.name, 'rb') as f:
                st.download_button("Download index (.index)", f, file_name="faiss_all.index")
            with open(tmp_meta.name, 'rb') as f:
                st.download_button("Download metadata (.pkl)", f, file_name="meta_all.pkl")

        elif folder_path:
            with st.spinner("Collecting files from folder..."):
                file_list = collect_files_from_path(folder_path)
            if not file_list:
                st.error("No pdf/docx files found in the provided folder path.")
            else:
                emb = Embedder(EMBED_MODEL_NAME)
                all_texts, all_meta = [], []
                total = len(file_list)
                for i, fp in enumerate(file_list, start=1):
                    st.write(f"[{i}/{total}] {fp}")
                    try:
                        with open(fp, 'rb') as f:
                            data = f.read()
                        text, pages = extract_any_bytes(fp, data)
                    except Exception as e:
                        st.warning(f"Failed to extract {fp}: {e}")
                        continue
                    chunks = word_overlap_chunks(text, target_words=target_words, overlap_words=overlap)
                    for cid, (s,e,ch) in enumerate(chunks):
                        meta = {"filename": os.path.basename(fp), "filepath": fp,
                                "chunk_id": f"{os.path.basename(fp)}_chunk_{cid}",
                                "start_word": s, "end_word": e, "preview": ch[:2000]}
                        all_texts.append(ch)
                        all_meta.append(meta)
                if dedupe:
                    seen=set(); ut=[]; um=[]
                    for t,m in zip(all_texts, all_meta):
                        h=hash(t)
                        if h in seen: continue
                        seen.add(h); ut.append(t); um.append(m)
                    all_texts, all_meta = ut, um
                with st.spinner("Embedding chunks (this may take time)..."):
                    vecs = emb.encode(all_texts, batch_size=batch_size)
                dim = vecs.shape[1]
                store = FaissStore(dim)
                store.add(vecs, all_meta)
                st.session_state.store = store
                st.session_state.emb = emb
                st.success("Index built from folder.")
                tmp_idx = tempfile.NamedTemporaryFile(delete=False, suffix='.index')
                tmp_meta = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
                store.save(tmp_idx.name, tmp_meta.name)
                with open(tmp_idx.name, 'rb') as f:
                    st.download_button("Download index (.index)", f, file_name="faiss_all.index")
                with open(tmp_meta.name, 'rb') as f:
                    st.download_button("Download metadata (.pkl)", f, file_name="meta_all.pkl")
        else:
            st.error("Provide either uploaded files or a folder path to build the index.")
    except Exception as e:
        st.error(f"Error while building index: {e}")

# --- Ask Queries ---
if ask_btn:
    if not st.session_state.store or not st.session_state.emb:
        st.error("No index present in session. Build or load an index first.")
    else:
        with st.spinner("Searching..."):
            q_vec = st.session_state.emb.encode([query])
            hits = st.session_state.store.search(q_vec, top_k=top_k)
        st.markdown("### Top matches")
        for i, (sc, meta) in enumerate(hits, start=1):
            if sc < min_score: continue
            st.write(f"{i}. score={sc:.3f} | file={meta.get('filename', meta.get('filepath',''))} | chunk={meta.get('chunk_id')}")
            st.write(meta.get('preview','')[:1000])
            st.write("---")
        if use_flan:
            with st.spinner("Generating answer with FLAN-T5 (this may take a while)..."):
                if st.session_state.use_flan:
                    ans = generate_answer(query, hits, min_score=0.0, max_contexts=6)
                    st.markdown("### 🧠 Answer (FLAN-T5)")
                    st.success(ans)
                else:
                    st.markdown("### 🧠 Answer (Concise from top results)")
                    pieces = [h[1]['preview'][:600] for h in hits if h[0] >= 0.0][:3]
                    st.write(" ".join(pieces)[:1000])
        else:
            st.markdown("### Concise answer from top preview(s)")
            if len(hits) == 0:
                st.write("I don't know.")
            else:
                pieces = [h[1]['preview'][:600] for h in hits if h[0] >= min_score][:3]
                st.write(" ".join(pieces)[:1000])

st.markdown("---")
st.caption("Notes: Building the index (embedding) can be compute-heavy. If you run this on Colab, keep GPU enabled. The FLAN model used for generation is optional and can be slow or memory-heavy.")

