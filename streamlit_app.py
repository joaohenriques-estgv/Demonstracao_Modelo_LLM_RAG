"""
Demo didática de LLM + RAG (Retrieval-Augmented Generation) em Streamlit.

Fluxo mostrado aos alunos:
  1. Definir um contexto base (system prompt)
  2. Carregar ficheiros de contexto (txt, md, pdf, docx)
  3. Os ficheiros são divididos em chunks e indexados (TF-IDF)
  4. A pergunta recupera os chunks mais relevantes
  5. Chunks + pergunta são enviados ao LLM, que responde com base neles

Executar:  streamlit run app.py
"""

import io
import hashlib

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Retrieval Lab", page_icon="🔎", layout="wide")

DEFAULT_SYSTEM = (
    "És um assistente de apoio a alunos. Responde em português de Portugal, "
    "de forma clara e concisa."
)

RAG_INSTRUCTIONS = (
    "\n\nVais receber excertos de documentos fornecidos pelo utilizador, "
    "identificados como [Fonte: ficheiro, chunk N]. Responde usando APENAS "
    "essa informação. Cita as fontes entre parênteses retos. Se a resposta "
    "não estiver nos excertos, diz claramente que não encontraste a "
    "informação nos documentos."
)


# ----------------------------------------------------------------------------
# 1. Leitura de ficheiros
# ----------------------------------------------------------------------------
def read_file(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        from docx import Document

        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    return ""


# ----------------------------------------------------------------------------
# 2. Chunking
# ----------------------------------------------------------------------------
def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Divide o texto em blocos de ~`size` caracteres com sobreposição."""
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # evita cortar a meio de uma palavra
            space = text.rfind(" ", start, end)
            if space > start + size // 2:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


# ----------------------------------------------------------------------------
# 3. Indexação e recuperação (TF-IDF, sem necessidade de API de embeddings)
# ----------------------------------------------------------------------------
def build_index(files, size, overlap):
    records = []  # {"source", "n", "text"}
    for f in files:
        for i, c in enumerate(chunk_text(read_file(f), size, overlap), start=1):
            records.append({"source": f.name, "n": i, "text": c})
    if not records:
        return None
    vec = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", ngram_range=(1, 2))
    matrix = vec.fit_transform([r["text"] for r in records])
    return {"records": records, "vec": vec, "matrix": matrix}


def retrieve(index, query, k):
    q = index["vec"].transform([query])
    scores = cosine_similarity(q, index["matrix"]).ravel()
    top = scores.argsort()[::-1][:k]
    return [(index["records"][i], float(scores[i])) for i in top if scores[i] > 0]


# ----------------------------------------------------------------------------
# 4. Chamada ao LLM
# ----------------------------------------------------------------------------
def call_llm(provider, api_key, model, base_url, system, messages, temperature):
    if provider == "Anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=1500,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    # OpenAI ou qualquer servidor compatível (Ollama, LM Studio, vLLM...)
    from openai import OpenAI

    client = OpenAI(api_key=api_key or "sem-chave", base_url=base_url or None)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system}] + messages,
    )
    return resp.choices[0].message.content


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
st.title("🔎 AI Retrieval Lab")
st.caption("Define o contexto, carrega documentos e vê como o modelo responde com base neles.")

if "history" not in st.session_state:
    st.session_state.history = []  # lista de dicts com role/content/debug

with st.sidebar:
    st.header("⚙️ Modelo")
    provider = st.selectbox("Fornecedor", ["Anthropic", "OpenAI / compatível (Ollama...)"])
    api_key = st.text_input("API key", type="password", help="Não é guardada.")
    if provider == "Anthropic":
        model = st.text_input("Modelo", "claude-sonnet-5-5")
        base_url = ""
    else:
        model = st.text_input("Modelo", "gpt-4o-mini")
        base_url = st.text_input(
            "Base URL (opcional)", "", placeholder="http://localhost:11434/v1 para Ollama"
        )
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.2, 0.1)

    st.header("🧩 Parâmetros de RAG")
    use_rag = st.toggle("Usar RAG", value=True, help="Desliga para comparar com o modelo sem documentos.")
    chunk_size = st.slider("Tamanho do chunk (caracteres)", 200, 2000, 800, 100)
    overlap = st.slider("Sobreposição", 0, 400, 100, 25)
    top_k = st.slider("Nº de chunks recuperados (top-k)", 1, 10, 4)

    if st.button("🗑️ Limpar conversa"):
        st.session_state.history = []
        st.rerun()

col_cfg, col_chat = st.columns([1, 2], gap="large")

with col_cfg:
    st.subheader("1️⃣ Contexto base")
    system_prompt = st.text_area(
        "System prompt", DEFAULT_SYSTEM, height=150,
        help="Define o papel e o comportamento do assistente.",
    )

    st.subheader("2️⃣ Documentos")
    files = st.file_uploader(
        "Ficheiros de contexto", type=["txt", "md", "pdf", "docx"], accept_multiple_files=True
    )

    index = None
    if files:
        key = (
            tuple(hashlib.md5(f.getvalue()).hexdigest() for f in files),
            chunk_size,
            overlap,
        )
        if st.session_state.get("index_key") != key:
            with st.spinner("A indexar documentos..."):
                st.session_state.index = build_index(files, chunk_size, overlap)
                st.session_state.index_key = key
        index = st.session_state.index
        if index:
            st.success(f"{len(files)} ficheiro(s) → {len(index['records'])} chunks indexados")
            with st.expander("Ver chunks"):
                for r in index["records"]:
                    st.markdown(f"**{r['source']} · chunk {r['n']}**")
                    st.text(r["text"][:300] + ("…" if len(r["text"]) > 300 else ""))
        else:
            st.warning("Não foi possível extrair texto dos ficheiros.")
    else:
        st.session_state.pop("index_key", None)
        st.info("Sem ficheiros: o modelo responde só com o contexto base.")

with col_chat:
    st.subheader("3️⃣ Conversa")

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            dbg = msg.get("debug")
            if dbg:
                with st.expander("🔍 Ver o que aconteceu nos bastidores"):
                    if dbg["chunks"]:
                        st.markdown("**Chunks recuperados**")
                        for rec, score in dbg["chunks"]:
                            st.markdown(
                                f"- `{rec['source']}` · chunk {rec['n']} · similaridade **{score:.3f}**"
                            )
                            st.caption(rec["text"][:400])
                    else:
                        st.markdown("_Nenhum chunk usado._")
                    st.markdown("**System prompt enviado**")
                    st.code(dbg["system"], language="text")
                    st.markdown("**Mensagem do utilizador enviada ao modelo**")
                    st.code(dbg["user"], language="text")

    question = st.chat_input("Escreve a tua pergunta...")

    if question:
        if not api_key and provider == "Anthropic":
            st.error("Indica a API key na barra lateral.")
            st.stop()

        with st.chat_message("user"):
            st.markdown(question)

        # --- Retrieval ---
        chunks = []
        if use_rag and index:
            chunks = retrieve(index, question, top_k)

        system = system_prompt
        user_content = question
        if chunks:
            system += RAG_INSTRUCTIONS
            context_block = "\n\n".join(
                f"[Fonte: {r['source']}, chunk {r['n']}]\n{r['text']}" for r, _ in chunks
            )
            user_content = f"Excertos dos documentos:\n\n{context_block}\n\nPergunta: {question}"
        elif use_rag and index:
            user_content = (
                f"{question}\n\n(Nota: não foram encontrados excertos relevantes nos documentos.)"
            )
            system += RAG_INSTRUCTIONS

        # Histórico anterior (só texto limpo) + pergunta atual aumentada
        messages = [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.history
        ] + [{"role": "user", "content": user_content}]

        with st.chat_message("assistant"):
            with st.spinner("A gerar resposta..."):
                try:
                    answer = call_llm(
                        "Anthropic" if provider == "Anthropic" else "OpenAI",
                        api_key, model, base_url, system, messages, temperature,
                    )
                except Exception as e:
                    st.error(f"Erro ao chamar o modelo: {e}")
                    st.stop()
            st.markdown(answer)

        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append(
            {
                "role": "assistant",
                "content": answer,
                "debug": {"chunks": chunks, "system": system, "user": user_content},
            }
        )
        st.rerun()
