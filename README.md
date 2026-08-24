# 📚 Research Paper Answer Bot

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about a collection of AI research papers.

The system retrieves relevant content from the research papers using multiple retrieval strategies and generates source-grounded answers using a local Large Language Model.

---

## 🚀 Project Overview

Research papers contain valuable technical information, but finding specific answers across multiple papers can be time-consuming.

The Research Paper Answer Bot solves this problem by implementing a Retrieval-Augmented Generation (RAG) pipeline.

The system:

1. Loads research paper PDFs.
2. Extracts and processes document content.
3. Splits documents into smaller chunks.
4. Converts chunks into vector embeddings.
5. Stores embeddings in ChromaDB.
6. Retrieves relevant chunks based on a user's question.
7. Sends the retrieved context to a local Large Language Model.
8. Generates an answer grounded in the retrieved research papers.
9. Displays supporting passages along with the answer, including paper title and page information.

---

## 🎯 Problem Statement

Finding information across multiple research papers requires manually reading and searching through large documents.

Traditional language models may generate answers using general knowledge, which can result in hallucinated or unsupported information.

This project implements a Retrieval-Augmented Generation (RAG) based question-answering system that retrieves relevant information from a research paper knowledge base before generating an answer.

---

## ✨ Features

- 📄 Question answering over multiple research paper PDFs
- 🔍 Dense semantic similarity search
- 🧩 MMR retrieval
- 🔀 Hybrid retrieval using Dense Search + BM25
- 🧠 Retrieval-Augmented Generation (RAG)
- 📦 Vector database using ChromaDB
- 🤗 Hugging Face embedding models
- 🦙 Local LLM generation using Ollama
- 📚 Top supporting passage display
- 📄 Paper title and page information for retrieved passages
- 🔒 Strict source-grounded answer generation
- 🚫 Reduced hallucination using retrieved context
- 🧪 Embedding model comparison
- 🧩 Chunking strategy comparison
- 📊 Retrieval strategy evaluation
- ⭐ Hybrid retrieval stretch goal
- 🖥️ Interactive web interface using Streamlit

---

# 🔄 RAG Workflow

The Research Paper Answer Bot follows a Retrieval-Augmented Generation (RAG) workflow:

```text
Research Paper PDFs
        ↓
Document Loading
        ↓
Text Chunking
        ↓
Embedding Generation
        ↓
ChromaDB Vector Database
        ↓
User Question
        ↓
Retrieval Strategy
(Dense / MMR / Hybrid)
        ↓
Retrieve Relevant Document Chunks
        ↓
Send Question + Retrieved Context to Llama 3.2
        ↓
Generate Source-Grounded Answer
        ↓
Display Answer + Supporting Passages

Workflow Explanation

Document Loading:
Research paper PDFs are loaded from the knowledge base.

Text Chunking:
The extracted text is divided into smaller chunks for efficient retrieval.

Embedding Generation:
Each document chunk is converted into vector embeddings using an embedding model.

Vector Storage:
The embeddings are stored in ChromaDB.

Question Input:
The user enters a question through the Streamlit application.

Retrieval:
The system retrieves relevant document chunks using the selected retrieval strategy.

Context Retrieval:
The most relevant chunks are selected as context for answer generation.

Answer Generation:
The user's question and retrieved context are sent to Llama 3.2 through Ollama.

Source-Grounded Response:
The model generates an answer based on the retrieved research paper content.

Supporting Passage Display:
The application displays the answer together with supporting passages, including paper title and page number.

🏗️ RAG Architecture
                ┌───────────────────────┐
                │ Research Paper PDFs   │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Document Loading      │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Text Chunking         │
                │ Chunk Size: 500       │
                │ Overlap: 100          │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ BGE Embeddings        │
                │ bge-small-en-v1.5     │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ ChromaDB              │
                │ 2368 Document Chunks  │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ User Question         │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Retrieval             │
                │ Dense / MMR / Hybrid  │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Relevant Context      │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Ollama + Llama 3.2    │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Source-Grounded       │
                │ Answer + Citations    │
                └───────────────────────┘
🛠️ Technologies Used
Technology	Purpose
Python	Core programming language
Streamlit	Web application interface
LangChain	Document processing and RAG components
ChromaDB	Vector database for storing document embeddings
Hugging Face	Embedding model integration
BAAI/bge-small-en-v1.5	Final embedding model
Ollama	Local LLM runtime
Llama 3.2	Answer generation
PyPDF	PDF document processing
Rank-BM25	Keyword-based retrieval
Jupyter Notebook	Data processing and experimentation
🧪 Experiments
1. Embedding Model Comparison

Two embedding models were tested for document retrieval:

all-MiniLM-L6-v2
BAAI/bge-small-en-v1.5

The models were compared based on the relevance of the retrieved research paper chunks.

The final application uses:

BAAI/bge-small-en-v1.5
2. Chunking Strategy Comparison

Different chunking configurations were tested:

Configuration	Chunk Size	Chunk Overlap
Initial Configuration	1000	200
Final Configuration	500	100

The smaller chunk configuration produced more focused document chunks for retrieval.

The final ChromaDB knowledge base contains:

2368 document chunks
3. Strict Source-Grounded Answer Generation

The prompting strategy was designed to ensure that answers are generated only from the retrieved research paper content.

The model is instructed to:

Use only the retrieved context.
Avoid outside knowledge.
Avoid unsupported information.
Ignore irrelevant retrieved sources.
Clearly indicate when a supported answer cannot be found.

For unsupported questions, the system returns:

I could not find a supported answer in the retrieved research papers.

This approach prioritizes answer reliability and source grounding.

4. Retrieval Strategy Comparison

The project implements and compares multiple retrieval strategies.

Dense Similarity Search

Dense retrieval uses semantic embeddings to retrieve document chunks that are conceptually similar to the user's question.

Maximal Marginal Relevance (MMR)

MMR retrieval balances:

Relevance to the user's question
Diversity among retrieved document chunks

This helps reduce repetitive or highly similar results.

The retrieval strategies were evaluated using questions related to:

BERT
Transformer architecture
LoRA
Retrieval-Augmented Generation
Chain-of-Thought prompting
5. Hybrid Retrieval - Stretch Goal ⭐

As a stretch goal, hybrid retrieval was implemented.

The hybrid retrieval approach combines:

Dense vector retrieval using ChromaDB
Keyword-based retrieval using BM25

The results are combined and ranked using a hybrid scoring approach.

Hybrid retrieval provides:

Semantic understanding from vector embeddings
Exact keyword matching from BM25
Improved retrieval coverage for technical research terminology
⚙️ Final Configuration
Component	Final Configuration
Embedding Model	BAAI/bge-small-en-v1.5
Chunk Size	500
Chunk Overlap	100
Vector Database	ChromaDB
Vector Database Path	data/chroma_db_small_chunks
Retrieval Strategies Tested	Dense, MMR, Hybrid
Final Knowledge Base	2368 document chunks
Retrieved Supporting Passages	Top 3
LLM Runtime	Ollama
Language Model	llama3.2:latest
Web Framework	Streamlit
Answer Strategy	Strict Source-Grounded RAG
📚 Research Papers Used

The knowledge base contains research papers related to artificial intelligence and large language models.

Topics include:

Transformer architecture
BERT
GPT-3
Retrieval-Augmented Generation
Chain-of-Thought Prompting
LoRA
Instruction-following language models

Examples of research papers in the knowledge base include:

Attention Is All You Need
BERT
Language Models are Few-Shot Learners
Retrieval-Augmented Generation
Chain-of-Thought Prompting
LoRA
InstructGPT
📊 Evaluation

The RAG system was evaluated using multiple questions related to the research papers.

Example evaluation questions include:

What is BERT?
What is the main contribution of the Transformer architecture?
What is LoRA?
What is Retrieval-Augmented Generation?
What is Chain of Thought prompting?

The evaluation focused on:

Retrieval relevance
Correct paper retrieval
Retrieval diversity
Retrieval execution time
Answer relevance
Source grounding
Unsupported-question handling
📈 Experimental Results
Dense vs MMR Retrieval Comparison
Retrieval Strategy	Average Retrieval Time
Dense Similarity Search	0.0519 seconds
MMR Retrieval	0.0267 seconds
Observation

MMR retrieval achieved a lower average retrieval time in this experiment while also providing more diverse retrieved results.

Hybrid Retrieval Results
Retrieval Strategy	Average Retrieval Time
Hybrid Retrieval	0.0498 seconds
Observation

Hybrid retrieval successfully combined Dense Search and BM25 keyword retrieval.

The hybrid results included:

Dense-only matches
BM25-only matches
Dense + BM25 combined matches

This demonstrated the successful implementation of the hybrid retrieval stretch goal.

Observed Retrieval Quality

Relevant research papers were successfully retrieved for major concepts:

Question Topic	Retrieved Research Paper
BERT	BERT
Transformer	Attention Is All You Need
LoRA	LoRA
RAG	Retrieval-Augmented Generation
Chain of Thought	Chain-of-Thought Prompting
🖥️ Application Demonstration

The Streamlit application allows users to:

View the research paper knowledge base.
Enter a question about the research papers.
Retrieve relevant research paper chunks.
Generate a source-grounded answer.
View the top supporting passages.
See the paper title and page information for the retrieved passages.
Application Home Screen

The Streamlit interface displays:

Research Paper Answer Bot
Research paper knowledge base
Loaded documents
Question input area

Add your application home screen screenshot here.

Generated Answer

After the user enters a research question, the system:

Retrieves relevant context.
Sends the question and context to Llama 3.2.
Generates a source-grounded answer.

Add your generated answer screenshot here.

Supporting Passages

The application displays the top supporting passages alongside the answer.

Each supporting passage includes:

Paper title
Page number
Relevant retrieved content

Add your supporting passages screenshot here.

🚫 Unsupported Question Handling

The system was also tested with questions unrelated to the research paper knowledge base.

Example:

Who won the FIFA World Cup in 2022?

Because the system uses strict source-grounded generation, it should avoid answering questions when the retrieved research papers do not contain sufficient supporting information.

Expected response:

I could not find a supported answer in the retrieved research papers.
📁 Project Structure
Research-Paper-Answer-Bot/
│
├── app/
│   └── app.py
│
├── data/
│   ├── research_papers/
│   │   ├── attention_is_all_you_need.pdf
│   │   ├── bert.pdf
│   │   ├── rag.pdf
│   │   ├── gpt3.pdf
│   │   ├── lora.pdf
│   │   └── ...
│   │
│   ├── chroma_db/
│   ├── chroma_db_bge/
│   ├── chroma_db_minilm_comparison/
│   └── chroma_db_small_chunks/
│
├── evaluation/
│   ├── evaluate_rag.py
│   ├── compare_retrieval.py
│   ├── hybrid_retrieval.py
│   └── evaluation/
│       ├── retrieval_strategy_comparison.csv
│       └── hybrid_retrieval_evaluation.csv
│
├── experiments/
│
├── models/
│
├── notebooks/
│   └── research_paper_rag.ipynb
│
├── src/
│
├── .gitignore
├── README.md
└── requirements.txt
💻 Installation
1. Clone the Repository
git clone https://github.com/Mrittu/Research-Paper-Answer-Bot.git

Move into the project directory:

cd Research-Paper-Answer-Bot
2. Create a Virtual Environment

For Windows:

python -m venv .venv

Activate the environment:

.\.venv\Scripts\Activate.ps1
3. Install Dependencies
pip install -r requirements.txt
🦙 Ollama Setup

Install Ollama and ensure that the Ollama service is running.

Pull the required language model:

ollama pull llama3.2

Verify the installed models:

ollama list

The application uses:

llama3.2:latest
▶️ Running the Application

From the project root, activate the virtual environment:

.\.venv\Scripts\Activate.ps1

Then run:

streamlit run app/app.py

The application will open in your browser, typically at:

http://localhost:8501
🧪 Running the Evaluation Scripts
Dense vs MMR Retrieval Comparison

Run:

python .\evaluation\compare_retrieval.py

This evaluates:

Dense Similarity Search
MMR Retrieval

The results are saved to:

evaluation/evaluation/retrieval_strategy_comparison.csv
Hybrid Retrieval Evaluation

Run:

python .\evaluation\hybrid_retrieval.py

This evaluates the hybrid retrieval approach using:

Dense retrieval
BM25 retrieval
Dense + BM25 combined retrieval

The results are saved to:

evaluation/evaluation/hybrid_retrieval_evaluation.csv
🔍 Example Questions

Try asking:

What is BERT?
What is the main contribution of the Transformer architecture?
What is LoRA?
What is Retrieval-Augmented Generation?
What is Chain of Thought prompting?
⚠️ Limitations

The current system has several limitations:

Retrieval quality depends on the embedding model and chunking strategy.
Relevant information may not always appear in the top retrieved chunks.
The generated answer may be incomplete when the retrieved context is limited.
Strict grounding can reduce hallucination but may increase the number of unanswered questions.
Answer quality depends on the research papers available in the knowledge base.
Similar research papers may produce overlapping or partially relevant retrieval results.
Evaluation is currently based primarily on representative questions and retrieval observations.
🔮 Future Improvements

Possible future enhancements include:

Reranking retrieved document chunks
Query rewriting
Metadata filtering by research paper
Retrieval confidence thresholds
Conversational memory
Support for additional document formats
Improved automated quantitative RAG evaluation
Cloud deployment
Multi-user support
Additional embedding and retrieval model comparisons
📦 Submission Checklist
Code
 Notebook runs end-to-end without errors
 At least 2 embedding models compared
 Vector database created and documents indexed
 At least 2 retrieval strategies implemented and compared
 Complete RAG chain built with LLM integration
 Supporting passages displayed with answers
 Paper title and page information included
 At least 1 stretch goal implemented
Presentation
 10–15 slide presentation prepared
 RAG architecture diagram included
 Embedding comparison documented
 Retrieval comparison documented
 Sample Q&A outputs included
Demo
 Streamlit application runs successfully
 Live research-paper questions can be demonstrated
 Supporting passages are displayed alongside answers
✅ Conclusion

The Research Paper Answer Bot successfully demonstrates an end-to-end Retrieval-Augmented Generation system for answering questions from a collection of research papers.

The complete pipeline includes:

PDF Research Papers
        ↓
Document Loading
        ↓
Text Chunking
        ↓
Embedding Generation
        ↓
ChromaDB Vector Storage
        ↓
Dense / MMR / Hybrid Retrieval
        ↓
Relevant Context Retrieval
        ↓
Strict RAG Prompt
        ↓
Ollama + Llama 3.2
        ↓
Source-Grounded Answer
        ↓
Top Supporting Passages

The project includes experiments with:

Multiple embedding models
Different chunking configurations
Dense similarity retrieval
MMR retrieval
Hybrid Dense + BM25 retrieval
Strict source-grounded prompting

The final system uses:

BAAI/bge-small-en-v1.5 embeddings
ChromaDB vector storage
Dense, MMR, and Hybrid retrieval experiments
BM25 for keyword-based hybrid retrieval
Llama 3.2 through Ollama
Streamlit for the interactive user interface

The project demonstrates how Retrieval-Augmented Generation can improve the reliability of AI-generated answers by grounding responses in retrieved research paper content and displaying supporting evidence.
