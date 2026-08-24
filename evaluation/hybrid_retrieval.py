from pathlib import Path
import time
import pandas as pd

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
# LOAD CHROMADB
# ============================================================

print("\nLoading ChromaDB...")

vector_db = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings
)

print(f"Documents in database: {vector_db._collection.count()}")


# ============================================================
# LOAD DOCUMENTS FOR BM25
# ============================================================

print("\nLoading documents for BM25...")

chroma_data = vector_db.get(
    include=["documents", "metadatas"]
)

documents = chroma_data["documents"]
metadatas = chroma_data["metadatas"]

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

print(f"BM25 index created with {len(documents)} chunks.")


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize_scores(scores):

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        return [1.0 for _ in scores]

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]


# ============================================================
# TRUE HYBRID RETRIEVAL
# ============================================================

def hybrid_search(question, k=3):

    # --------------------------------------------------------
    # 1. DENSE VECTOR RETRIEVAL
    # --------------------------------------------------------

    dense_results = vector_db.similarity_search_with_score(
        question,
        k=20
    )

    # Chroma returns distance, so lower = better.
    dense_documents = [
        doc.page_content
        for doc, score in dense_results
    ]

    dense_metadatas = [
        doc.metadata
        for doc, score in dense_results
    ]

    dense_distances = [
        score
        for doc, score in dense_results
    ]

    # Convert distance to similarity
    dense_similarities = [
        1 / (1 + distance)
        for distance in dense_distances
    ]

    dense_scores = normalize_scores(
        dense_similarities
    )

    dense_score_map = {}

    for document, metadata, score in zip(
        dense_documents,
        dense_metadatas,
        dense_scores
    ):
        dense_score_map[document] = {
            "metadata": metadata,
            "score": score
        }


    # --------------------------------------------------------
    # 2. BM25 KEYWORD RETRIEVAL
    # --------------------------------------------------------

    tokenized_query = question.lower().split()

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    normalized_bm25_scores = normalize_scores(
        bm25_scores
    )

    # --------------------------------------------------------
    # 3. COMBINE SCORES
    # --------------------------------------------------------

    combined_results = []

    dense_weight = 0.6
    bm25_weight = 0.4

    for index, document in enumerate(documents):

        dense_score = 0.0

        if document in dense_score_map:
            dense_score = dense_score_map[
                document
            ]["score"]

        bm25_score = normalized_bm25_scores[
            index
        ]

        hybrid_score = (
            dense_weight * dense_score
            + bm25_weight * bm25_score
        )

        metadata = metadatas[index]

        source_type = []

        if dense_score > 0:
            source_type.append("Dense")

        if bm25_score > 0:
            source_type.append("BM25")

        combined_results.append(
            {
                "document": document,
                "metadata": metadata,
                "dense_score": dense_score,
                "bm25_score": bm25_score,
                "hybrid_score": hybrid_score,
                "source": " + ".join(source_type)
            }
        )


    # --------------------------------------------------------
    # 4. SORT BY HYBRID SCORE
    # --------------------------------------------------------

    combined_results = sorted(
        combined_results,
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return combined_results[:k]


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
# RUN HYBRID RETRIEVAL EVALUATION
# ============================================================

results = []

for question in questions:

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    start_time = time.time()

    retrieved_results = hybrid_search(
        question,
        k=3
    )

    retrieval_time = time.time() - start_time

    sources = []

    for i, result in enumerate(
        retrieved_results,
        start=1
    ):

        metadata = result["metadata"]

        paper = metadata.get(
            "paper_title",
            "Unknown"
        )

        page = metadata.get(
            "page_number",
            metadata.get("page", "Unknown")
        )

        source_text = (
            f"{paper} (Page {page}) "
            f"[{result['source']}] "
            f"Hybrid Score: "
            f"{result['hybrid_score']:.4f}"
        )

        sources.append(source_text)

        print(
            f"{i}. {source_text}"
        )

    print(
        f"\nRetrieval Time: "
        f"{retrieval_time:.4f} seconds"
    )

    results.append({
        "question": question,
        "hybrid_sources": " | ".join(sources),
        "retrieval_time_seconds": round(
            retrieval_time,
            4
        )
    })


# ============================================================
# SAVE RESULTS
# ============================================================

output_folder = (
    PROJECT_ROOT /
    "evaluation" /
    "evaluation"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    output_folder /
    "hybrid_retrieval_evaluation.csv"
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
print("HYBRID RETRIEVAL STRETCH GOAL COMPLETED")
print("=" * 70)

print(
    f"\nAverage Retrieval Time: "
    f"{df['retrieval_time_seconds'].mean():.4f} seconds"
)

print(
    f"\nResults saved to:\n{output_file}"
)