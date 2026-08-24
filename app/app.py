import streamlit as st
from pathlib import Path
import sys
import ollama

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db_small_chunks"

# This is the collection that contains your 2368 chunks
COLLECTION_NAME = "research_papers_small_chunks"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Research Paper Answer Bot",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    .answer-box {
        background-color: #f6f8fc;
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #4f8bf9;
        margin-bottom: 20px;
        line-height: 1.7;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    return embeddings


# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

@st.cache_resource
def load_vector_db():

    if not CHROMA_PATH.exists():
        return None, 0

    embeddings = load_embeddings()

    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    )

    document_count = vector_db._collection.count()

    return vector_db, document_count


# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():

    try:

        models = ollama.list()

        return True, models

    except Exception as e:

        return False, str(e)


# ============================================================
# GENERATE STRICT RAG ANSWER
# ============================================================

def generate_answer(question, vector_db):

    # --------------------------------------------------------
    # STEP 1: RETRIEVE TOP 3 MOST RELEVANT DOCUMENTS
    # --------------------------------------------------------

    retrieved_docs = vector_db.similarity_search(
        question,
        k=3
    )

    # --------------------------------------------------------
    # STEP 2: BUILD CONTEXT
    # --------------------------------------------------------

    context = ""

    for i, doc in enumerate(retrieved_docs, start=1):

        paper = doc.metadata.get(
            "paper_title",
            "Unknown Paper"
        )

        page = doc.metadata.get(
            "page_number",
            doc.metadata.get(
                "page",
                "Unknown Page"
            )
        )

        context += f"""
SOURCE {i}
Paper: {paper}
Page: {page}

Content:
{doc.page_content}

--------------------------------------------------
"""

    # --------------------------------------------------------
    # STEP 3: STRICT SOURCE-GROUNDED PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a research paper question-answering assistant.

Answer the user's question using ONLY the information
explicitly supported by the retrieved research paper context.

STRICT RULES:

1. Use ONLY the retrieved sources below.

2. Do NOT use outside knowledge.

3. Do NOT infer technical mechanisms that are not explicitly
   described in the retrieved text.

4. Do NOT combine unrelated information from different papers.

5. Ignore retrieved information that does not directly answer
   the user's question.

6. If the retrieved context provides only a partial answer,
   clearly state that the available research papers provide
   only a partial answer.

7. If the answer is not supported by the retrieved context, say
   exactly:

   "I could not find a supported answer in the retrieved research papers."

8. Give the direct answer first.

 

RETRIEVED RESEARCH PAPER CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    # --------------------------------------------------------
    # STEP 4: GENERATE ANSWER USING OLLAMA
    # --------------------------------------------------------

    response = ollama.generate(
        model="llama3.2:latest",
        prompt=prompt
    )

    return response["response"], retrieved_docs


# ============================================================
# FIND PDF FILES
# ============================================================

def find_pdf_files():

    possible_pdf_folders = [
        PROJECT_ROOT / "data" / "research_papers",
        PROJECT_ROOT / "data" / "papers",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "papers"
    ]

    pdf_files = []

    for folder in possible_pdf_folders:

        if folder.exists():

            found_files = list(folder.glob("*.pdf"))

            if found_files:
                pdf_files = found_files
                break

    return sorted(pdf_files)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    pdf_files = find_pdf_files()

    if pdf_files:

        st.success(
            f"Found {len(pdf_files)} PDF files"
        )

        for pdf in pdf_files:
            st.write(f"📄 {pdf.name}")

    else:

        st.warning("No PDF files found.")

    st.divider()

    st.header("🤖 Model")

    ollama_ok, ollama_info = check_ollama()

    if ollama_ok:

        st.success("Ollama Connected")

        st.caption("Model: llama3.2:latest")
        st.caption("Local AI Model")

    else:

        st.error("Ollama Not Connected")

        st.caption(
            "Make sure Ollama is running."
        )

    st.divider()

    st.header("ℹ️ About")

    st.write(
        "This bot searches research papers stored in "
        "ChromaDB and generates answers using only the "
        "retrieved content."
    )


# ============================================================
# MAIN PAGE
# ============================================================

st.title("📚 Research Paper Answer Bot")

st.write(
    "Ask questions about the AI research papers stored "
    "in the knowledge base."
)

st.caption(
    "The bot searches the research papers and generates "
    "answers based only on the retrieved research paper content."
)

st.divider()


# ============================================================
# LOAD DATABASE
# ============================================================

with st.spinner("Loading knowledge base..."):

    vector_db, document_count = load_vector_db()


# ============================================================
# DATABASE ERROR HANDLING
# ============================================================

if vector_db is None:

    st.error(
        "❌ Vector database could not be loaded."
    )

    st.info(
        f"Expected database location:\n\n"
        f"`{CHROMA_PATH}`"
    )

    st.stop()


if document_count == 0:

    st.error(
        f"❌ The knowledge base contains 0 document chunks."
    )

    st.write(
        "**Expected database location:**"
    )

    st.code(
        str(CHROMA_PATH)
    )

    st.write(
        f"**Collection name:** `{COLLECTION_NAME}`"
    )

    st.stop()


# ============================================================
# DATABASE SUCCESS
# ============================================================

st.success(
    f"✅ Knowledge base loaded successfully — "
    f"{document_count} document chunks found."
)

st.caption(
    f"Collection: {COLLECTION_NAME}"
)


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Ask a question",
    placeholder="Example: What is the Transformer architecture?"
)

ask_button = st.button(
    "🔍 Ask Question",
    use_container_width=True
)


# ============================================================
# ANSWER QUESTION
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif not ollama_ok:

        st.error(
            "Ollama is not connected. "
            "Please start Ollama and try again."
        )

    else:

        try:

            # =================================================
            # GENERATE STRICT RAG ANSWER
            # =================================================

            with st.spinner(
                "Searching research papers and generating answer..."
            ):

                answer, retrieved_docs = generate_answer(
                    question,
                    vector_db
                )

            # =================================================
            # CHECK RETRIEVED DOCUMENTS
            # =================================================

            if not retrieved_docs:

                st.warning(
                    "No relevant information was found "
                    "in the knowledge base."
                )

            else:

                # =================================================
                # DISPLAY ANSWER
                # =================================================

                st.divider()

                st.header("🤖 Answer")

                st.markdown(
                    f"""
                    <div class="answer-box">
                    {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # =================================================
                # DISPLAY SOURCES
                # =================================================

                st.divider()

                st.header("📚 Top-3 Supporting Passages")

                displayed_sources = set()

                source_number = 1

                for doc in retrieved_docs:

                    paper = doc.metadata.get(
                        "paper_title",
                        "Unknown Paper"
                    )

                    page = doc.metadata.get(
                        "page_number",
                        doc.metadata.get(
                            "page",
                            "Unknown Page"
                        )
                    )

                    source_file = doc.metadata.get(
                        "source_file",
                        doc.metadata.get(
                            "source",
                            "Unknown File"
                        )
                    )

                    source_key = (
                        f"{paper}-{page}-{source_file}"
                    )

                    if source_key in displayed_sources:
                        continue

                    displayed_sources.add(source_key)

                    with st.expander(
                        f"Source {source_number} — "
                        f"{paper} (Page {page})"
                    ):

                        st.write(
                            f"**Paper:** {paper}"
                        )

                        st.write(
                            f"**Page:** {page}"
                        )

                        st.write(
                            f"**File:** {source_file}"
                        )

                        st.markdown(
                            "### Relevant Supporting Passage"
                        )

                        st.write(
                            doc.page_content
                        )

                    source_number += 1

        except Exception as e:

            st.error(
                "An error occurred while generating the answer."
            )

            st.exception(e)