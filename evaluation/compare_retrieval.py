from pathlib import Path
import sys
import time
import pandas as pd

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db_small_chunks"

COLLECTION_NAME = "research_papers_small_chunks"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

print("Embedding model loaded successfully.")


# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

print("\nLoading ChromaDB...")

vector_db = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings
)

document_count = vector_db._collection.count()

print(f"Documents in database: {document_count}")


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

questions = [
    "What is BERT?",
    "What is the main contribution of the Transformer architecture?",
    "What is LoRA?",
    "What is Retrieval-Augmented Generation?",
    "What is Chain of Thought prompting?"
]


# ============================================================
# COMPARE RETRIEVAL STRATEGIES
# ============================================================

results = []

for question in questions:

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    # ========================================================
    # 1. DENSE SIMILARITY SEARCH
    # ========================================================

    start_time = time.time()

    dense_docs = vector_db.similarity_search(
        question,
        k=3
    )

    dense_time = time.time() - start_time

    dense_sources = []

    for doc in dense_docs:

        paper = doc.metadata.get(
            "paper_title",
            "Unknown"
        )

        page = doc.metadata.get(
            "page_number",
            doc.metadata.get("page", "Unknown")
        )

        dense_sources.append(
            f"{paper} (Page {page})"
        )

    print("\nDENSE SIMILARITY SEARCH")

    for i, source in enumerate(dense_sources, start=1):
        print(f"{i}. {source}")

    print(
        f"Retrieval Time: "
        f"{dense_time:.4f} seconds"
    )


    # ========================================================
    # 2. MMR RETRIEVAL
    # ========================================================

    start_time = time.time()

    mmr_docs = vector_db.max_marginal_relevance_search(
        question,
        k=3,
        fetch_k=10,
        lambda_mult=0.5
    )

    mmr_time = time.time() - start_time

    mmr_sources = []

    for doc in mmr_docs:

        paper = doc.metadata.get(
            "paper_title",
            "Unknown"
        )

        page = doc.metadata.get(
            "page_number",
            doc.metadata.get("page", "Unknown")
        )

        mmr_sources.append(
            f"{paper} (Page {page})"
        )

    print("\nMMR RETRIEVAL")

    for i, source in enumerate(mmr_sources, start=1):
        print(f"{i}. {source}")

    print(
        f"Retrieval Time: "
        f"{mmr_time:.4f} seconds"
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    results.append(
        {
            "question": question,
            "dense_sources": " | ".join(dense_sources),
            "dense_retrieval_time_seconds": round(
                dense_time,
                4
            ),
            "mmr_sources": " | ".join(mmr_sources),
            "mmr_retrieval_time_seconds": round(
                mmr_time,
                4
            )
        }
    )


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

output_folder = PROJECT_ROOT / "evaluation" / "evaluation"

output_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE CSV
# ============================================================

output_file = (
    output_folder /
    "retrieval_strategy_comparison.csv"
)

df = pd.DataFrame(results)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RETRIEVAL STRATEGY COMPARISON COMPLETED")
print("=" * 70)

print("\nAverage Retrieval Time")

print(
    f"Dense Similarity Search: "
    f"{df['dense_retrieval_time_seconds'].mean():.4f} seconds"
)

print(
    f"MMR Retrieval: "
    f"{df['mmr_retrieval_time_seconds'].mean():.4f} seconds"
)

print(f"\nResults saved to:\n{output_file}")