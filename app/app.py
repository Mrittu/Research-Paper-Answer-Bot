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

    chroma_path = (
        PROJECT_ROOT
        / "data"
        / "chroma_db_small_chunks"
    )

    # Check whether database folder exists
    if not chroma_path.exists():
        return None, 0

    try:

        embeddings = load_embeddings()

        vector_db = Chroma(
            persist_directory=str(chroma_path),
            embedding_function=embeddings
        )

        # Count documents in ChromaDB
        document_count = vector_db._collection.count()

        return vector_db, document_count

    except Exception:
        return None, 0


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
    # STEP 1: RETRIEVE TOP 3 DOCUMENTS
    # --------------------------------------------------------

    retrieved_docs = vector_db.similarity_search(
        question,
        k=3
    )

    if not retrieved_docs:
        return (
            "I could not find relevant information "
            "in the retrieved research papers.",
            []
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
                "Unknown"
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

3. Do NOT infer information that is not explicitly supported
by the retrieved research paper text.

4. Do NOT combine unrelated information from different papers.

5. Ignore retrieved information that does not directly answer
the user's question.

6. If the retrieved context provides only a partial answer,
clearly state that the available research papers provide
only a partial answer.

7. If the answer is not supported by the retrieved context,
say exactly:

"I could not find a supported answer in the retrieved research papers."

8. Give the direct answer first.

9. At the end of the answer, mention the source number(s)
actually used in this format:

Sources used: SOURCE 1, SOURCE 2

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

    answer = response.get(
        "response",
        "I could not generate an answer."
    )

    return answer, retrieved_docs


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

            found_files = list(
                folder.glob("*.pdf")
            )

            if found_files:

                pdf_files = found_files
                break

    return pdf_files


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

        st.warning(
            "No PDF files found."
        )

    st.divider()


    # ========================================================
    # MODEL STATUS
    # ========================================================

    st.header("🤖 Model")

    ollama_ok, ollama_info = check_ollama()

    if ollama_ok:

        st.success(
            "Ollama Connected"
        )

        st.caption(
            "Model: llama3.2:latest"
        )

        st.caption(
            "Local AI Model"
        )

    else:

        st.error(
            "Ollama Not Connected"
        )

        st.caption(
            "Ollama is required to generate answers."
        )

    st.divider()


    # ========================================================
    # ABOUT
    # ========================================================

    st.header("ℹ️ About")

    st.write(
        "This bot searches AI research papers stored "
        "in ChromaDB and generates answers using only "
        "the retrieved research paper content."
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
# DATABASE PATH
# ============================================================

CHROMA_DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "chroma_db_small_chunks"
)


# ============================================================
# LOAD DATABASE
# ============================================================

with st.spinner(
    "Loading knowledge base..."
):

    vector_db, document_count = load_vector_db()


# ============================================================
# DATABASE ERROR HANDLING
# ============================================================

if vector_db is None:

    st.error(
        "❌ The knowledge base could not be loaded or "
        "contains 0 document chunks."
    )

    st.write(
        "**Expected database location:**"
    )

    st.code(
        str(CHROMA_DB_PATH)
    )

    st.write(
        "**Database folder exists:**",
        CHROMA_DB_PATH.exists()
    )

    if CHROMA_DB_PATH.exists():

        try:

            files = [
                item.name
                for item in CHROMA_DB_PATH.iterdir()
            ]

            st.write(
                "**Files inside database folder:**"
            )

            st.write(files)

        except Exception as e:

            st.write(
                "Could not read database folder:",
                str(e)
            )

    st.stop()


if document_count == 0:

    st.error(
        "❌ The knowledge base contains 0 document chunks."
    )

    st.write(
        "**Expected database location:**"
    )

    st.code(
        str(CHROMA_DB_PATH)
    )

    st.write(
        "**Database folder exists:**",
        CHROMA_DB_PATH.exists()
    )

    if CHROMA_DB_PATH.exists():

        try:

            files = [
                item.name
                for item in CHROMA_DB_PATH.iterdir()
            ]

            st.write(
                "**Files inside database folder:**"
            )

            st.write(files)

        except Exception as e:

            st.write(
                "Could not read database folder:",
                str(e)
            )

    st.stop()


# ============================================================
# DATABASE SUCCESS
# ============================================================

st.success(
    f"✅ Knowledge base loaded successfully — "
    f"{document_count} document chunks found."
)


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Ask a question",
    placeholder=(
        "Example: What is the Transformer architecture?"
    )
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

                # =============================================
                # DISPLAY ANSWER
                # =============================================

                st.divider()

                st.header(
                    "🤖 Answer"
                )

                st.markdown(
                    f"""
                    <div class="answer-box">
                    {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # =============================================
                # DISPLAY SOURCES
                # =============================================

                st.divider()

                st.header(
                    "📚 Top 3 Supporting Passages"
                )

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


                    # Avoid duplicate sources
                    source_key = (
                        f"{paper}-{page}-{source_file}"
                    )

                    if source_key in displayed_sources:

                        continue


                    displayed_sources.add(
                        source_key
                    )


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
                            "### Supporting Passage"
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