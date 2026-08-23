import streamlit as st
from pathlib import Path
import sys
import ollama
import chromadb

# ============================================================
# PROJECT PATH SETUP
# ============================================================

# app.py is inside:
# Research-Paper-Answer-Bot/app/app.py
#
# Therefore parent.parent is the project root:
# Research-Paper-Answer-Bot/

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Research Paper Answer Bot",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# IMPORTS
# ============================================================

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1100px;
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

    chroma_path = (
        project_root
        / "data"
        / "chroma_db_small_chunks"
    )

    # Check database folder
    if not chroma_path.exists():

        return None, 0, None


    # --------------------------------------------------------
    # Connect directly to Chroma database
    # --------------------------------------------------------

    chroma_client = chromadb.PersistentClient(
        path=str(chroma_path)
    )


    # Get all collections stored in this database
    collections = chroma_client.list_collections()


    if not collections:

        return None, 0, None


    # --------------------------------------------------------
    # Find the collection containing the most documents
    # --------------------------------------------------------

    selected_collection = None
    highest_count = 0


    for collection_info in collections:

        collection_name = collection_info.name

        collection = chroma_client.get_collection(
            name=collection_name
        )

        collection_count = collection.count()

        if collection_count > highest_count:

            highest_count = collection_count

            selected_collection = collection_name


    # If no documents exist
    if selected_collection is None:

        return None, 0, None


    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    embeddings = load_embeddings()


    # --------------------------------------------------------
    # Load the correct Chroma collection
    # --------------------------------------------------------

    vector_db = Chroma(
        persist_directory=str(chroma_path),
        embedding_function=embeddings,
        collection_name=selected_collection
    )


    document_count = vector_db._collection.count()


    return (
        vector_db,
        document_count,
        selected_collection
    )


# ============================================================
# CHECK OLLAMA CONNECTION
# ============================================================

def check_ollama():

    try:

        ollama.list()

        return True

    except Exception:

        return False


# ============================================================
# GENERATE STRICT RAG ANSWER
# ============================================================

def generate_answer(question, vector_db):

    # --------------------------------------------------------
    # RETRIEVE RELEVANT DOCUMENTS
    # --------------------------------------------------------

    retrieved_docs = vector_db.similarity_search(
        question,
        k=5
    )


    # If nothing is retrieved
    if not retrieved_docs:

        return (
            "I could not find relevant information "
            "in the available research papers.",
            []
        )


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context_parts = []


    for i, doc in enumerate(
        retrieved_docs,
        start=1
    ):

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


        source_text = f"""
SOURCE {i}
Paper: {paper}
Page: {page}

Content:
{doc.page_content}
"""

        context_parts.append(source_text)


    context = "\n\n" + "\n\n".join(context_parts)


    # --------------------------------------------------------
    # STRICT SOURCE-GROUNDED PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a strict research paper question-answering assistant.

Answer the user's question using ONLY information explicitly
supported by the retrieved research paper context.

STRICT RULES:

1. Use ONLY the retrieved research paper content.

2. Do NOT use outside knowledge.

3. Do NOT invent information.

4. Do NOT assume information that is not explicitly stated.

5. Do NOT combine unrelated information from different papers.

6. Ignore retrieved sources that do not directly support the answer.

7. If the retrieved context provides only a partial answer,
say that the available research papers provide only a partial answer.

8. If the answer is not supported by the retrieved context,
respond exactly:

"I could not find a supported answer in the retrieved research papers."

9. Give a clear and concise answer.

10. At the end, include the source numbers actually used.

Use this format:

ANSWER:
[your answer]

Sources used: SOURCE 1, SOURCE 2


RETRIEVED RESEARCH PAPER CONTEXT:

{context}


USER QUESTION:

{question}


ANSWER:
"""


    # --------------------------------------------------------
    # GENERATE ANSWER USING OLLAMA
    # --------------------------------------------------------

    response = ollama.generate(
        model="llama3.2:latest",
        prompt=prompt
    )


    answer = response["response"]


    return answer, retrieved_docs


# ============================================================
# FIND PDF FILES
# ============================================================

def find_pdf_files():

    possible_folders = [

        project_root
        / "data"
        / "research_papers",

        project_root
        / "research_papers",

        project_root
        / "data"
    ]


    for folder in possible_folders:

        if folder.exists():

            pdf_files = list(
                folder.glob("*.pdf")
            )

            if pdf_files:

                return sorted(pdf_files)


    return []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Knowledge Base")


    # --------------------------------------------------------
    # DISPLAY PDF FILES
    # --------------------------------------------------------

    pdf_files = find_pdf_files()


    if pdf_files:

        st.success(
            f"Found {len(pdf_files)} PDF files"
        )


        for pdf in pdf_files:

            st.write(
                f"📄 {pdf.name}"
            )


    else:

        st.warning(
            "No PDF files found."
        )


    st.divider()


    # --------------------------------------------------------
    # OLLAMA STATUS
    # --------------------------------------------------------

    st.header("🤖 Model")


    ollama_ok = check_ollama()


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
            "Start Ollama and refresh this page."
        )


    st.divider()


    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

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
# LOAD KNOWLEDGE BASE
# ============================================================

with st.spinner(
    "Loading knowledge base..."
):

    (
        vector_db,
        document_count,
        collection_name
    ) = load_vector_db()


# ------------------------------------------------------------
# DATABASE ERROR
# ------------------------------------------------------------

if vector_db is None:

    st.error(
        "❌ The knowledge base could not be loaded "
        "or contains 0 document chunks."
    )


    st.write(
        "Expected database location:"
    )


    st.code(
        str(
            project_root
            / "data"
            / "chroma_db_small_chunks"
        )
    )


    st.stop()


# ------------------------------------------------------------
# EMPTY DATABASE
# ------------------------------------------------------------

if document_count == 0:

    st.error(
        "❌ The knowledge base contains 0 document chunks."
    )

    st.stop()


# ------------------------------------------------------------
# SUCCESS
# ------------------------------------------------------------

st.success(
    f"✅ Knowledge base loaded successfully — "
    f"{document_count} document chunks found."
)


st.caption(
    f"Chroma collection: {collection_name}"
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
# PROCESS QUESTION
# ============================================================

if ask_button:


    # --------------------------------------------------------
    # EMPTY QUESTION
    # --------------------------------------------------------

    if not question.strip():

        st.warning(
            "Please enter a question."
        )


    # --------------------------------------------------------
    # OLLAMA NOT CONNECTED
    # --------------------------------------------------------

    elif not ollama_ok:

        st.error(
            "Ollama is not connected. "
            "Please start Ollama and try again."
        )


    # --------------------------------------------------------
    # PROCESS QUESTION
    # --------------------------------------------------------

    else:

        try:


            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            with st.spinner(
                "Searching research papers and generating answer..."
            ):

                answer, retrieved_docs = generate_answer(
                    question,
                    vector_db
                )


            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            st.divider()


            st.header("🤖 Answer")


            st.markdown(
                f"""
                <div class="answer-box">
                {answer.replace(chr(10), '<br>')}
                </div>
                """,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # DISPLAY RETRIEVED SOURCES
            # ------------------------------------------------

            if retrieved_docs:

                st.divider()


                st.header(
                    "📚 Retrieved Sources"
                )


                for i, doc in enumerate(
                    retrieved_docs,
                    start=1
                ):


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


                    source_file = doc.metadata.get(
                        "source_file",
                        doc.metadata.get(
                            "source",
                            "Unknown File"
                        )
                    )


                    with st.expander(
                        f"Source {i} — "
                        f"{paper} "
                        f"(Page {page})"
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
                            "### Relevant Content"
                        )


                        st.write(
                            doc.page_content
                        )


        # ----------------------------------------------------
        # ERROR HANDLING
        # ----------------------------------------------------

        except Exception as e:

            st.error(
                "An error occurred while generating the answer."
            )


            st.exception(e)