from pathlib import Path
import sys

# ============================================================
# PROJECT PATH SETUP
# ============================================================

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# ============================================================
# IMPORTS
# ============================================================

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ============================================================
# LOAD CHROMADB
# ============================================================

chroma_path = project_root / "data" / "chroma_db"

if not chroma_path.exists():
    print("ERROR: ChromaDB database not found.")
    print("Expected location:", chroma_path)
    sys.exit()

print("\nLoading ChromaDB...")

vector_db = Chroma(
    persist_directory=str(chroma_path),
    embedding_function=embedding_model,
    collection_name="research_papers"
)

document_count = vector_db._collection.count()

print("ChromaDB loaded successfully.")
print("Total document chunks:", document_count)


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

test_questions = [

    "What is the Transformer architecture?",

    "What is BERT?",

    "What is Chain of Thought prompting?",

    "What is Retrieval Augmented Generation?",

    "What is LoRA and how does it work?"

     "What is instruction tuning?",

    "What is self-attention?",

    "How does GPT generate text?",

    "What are the advantages of the Transformer model?",

    "How does LoRA reduce the number of parameters required for fine-tuning?"
]



# ============================================================
# RUN RETRIEVAL EVALUATION
# ============================================================

print("\n")
print("=" * 80)
print("RESEARCH PAPER ANSWER BOT - RETRIEVAL EVALUATION")
print("=" * 80)


for question_number, question in enumerate(
    test_questions,
    start=1
):

    print("\n")
    print("-" * 80)

    print(f"QUESTION {question_number}")
    print("-" * 80)

    print("Question:")
    print(question)

    print("\nRetrieving top 5 relevant document chunks...")

    try:

        results = vector_db.similarity_search_with_score(
            question,
            k=5
        )

        if not results:

            print("\nNo relevant documents were retrieved.")
            continue


        print(f"\nRetrieved {len(results)} document chunks.")

        print("\nRETRIEVED SOURCES")
        print("-" * 80)


        for source_number, (document, score) in enumerate(
            results,
            start=1
        ):

            paper = document.metadata.get(
                "paper_title",
                "Unknown Paper"
            )

            page = document.metadata.get(
                "page",
                "Unknown Page"
            )

            source_file = document.metadata.get(
                "source_file",
                "Unknown File"
            )


            print(f"\nSource {source_number}")

            print("Paper:", paper)
            print("Page:", page)
            print("File:", source_file)
            print("Distance Score:", round(float(score), 4))

            content_preview = document.page_content[:500]

            print("\nContent Preview:")
            print(content_preview)

            if len(document.page_content) > 500:
                print("...")


    except Exception as error:

        print("\nERROR during evaluation:")
        print(error)


print("\n")
print("=" * 80)
print("EVALUATION COMPLETED")
print("=" * 80)