# 📚 Research Paper Answer Bot

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about a collection of AI research papers. The system retrieves relevant content from the research papers using vector similarity search and generates source-grounded answers using a local Large Language Model.

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
9. Displays the retrieved sources along with the answer.

---

## 🎯 Problem Statement

Finding information across multiple research papers requires manually reading and searching through large documents.

Traditional language models may generate answers using general knowledge, which can result in hallucinated or unsupported information.

This project implements a Retrieval-Augmented Generation (RAG) based question-answering system that retrieves relevant information from a research paper knowledge base before generating an answer.

---

## ✨ Features

- 📄 Question answering over multiple research paper PDFs
- 🔍 Semantic similarity search
- 🧠 Retrieval-Augmented Generation (RAG)
- 📦 Vector database using ChromaDB
- 🤗 Hugging Face embedding models
- 🦙 Local LLM generation using Ollama
- 📚 Retrieved source display
- 🔒 Strict source-grounded answer generation
- 🚫 Reduced hallucination using retrieved context
- 🧪 Embedding model comparison
- 🧩 Chunking strategy comparison
- 📊 Retrieval and answer evaluation
- 🖥️ Interactive web interface using Streamlit

---

## 🔄 RAG Workflow

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
Similarity Search
        ↓
Retrieve Relevant Document Chunks
        ↓
Send Question + Retrieved Context to Llama 3.2
        ↓
Generate Source-Grounded Answer
        ↓
Display Answer + Retrieved Sources

Workflow Explanation
Document Loading: Research paper PDFs are loaded from the knowledge base.
Text Chunking: The extracted text is divided into smaller chunks for efficient retrieval.
Embedding Generation: Each document chunk is converted into vector embeddings using an embedding model.
Vector Storage: The embeddings are stored in ChromaDB.
Question Input: The user enters a question through the Streamlit application.
Similarity Search: The system searches ChromaDB to find the most relevant document chunks.
Context Retrieval: The top relevant chunks are selected as context.
Answer Generation: The user's question and retrieved context are sent to Llama 3.2 through Ollama.
Source-Grounded Response: The model generates an answer based only on the retrieved research paper content.
Source Display: The application displays the generated answer along with the retrieved paper sources and relevant content.

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

3. Strict Source-Grounded Answer Generation

The prompting strategy was improved to ensure that answers are generated only from the retrieved research paper content.

The model is instructed to:

Use only the retrieved context.
Avoid outside knowledge.
Avoid unsupported information.
Ignore irrelevant retrieved sources.
Clearly indicate when a supported answer cannot be found.

For unsupported questions, the system returns:

I could not find a supported answer in the retrieved research papers.

This approach prioritizes answer reliability and source grounding.

⚙️ Final Configuration
Component	Final Configuration
Embedding Model	BAAI/bge-small-en-v1.5
Chunk Size	500
Chunk Overlap	100
Vector Database	ChromaDB
Vector Database Path	data/chroma_db_small_chunks
Retrieval Method	Similarity Search
Retrieved Documents	Top 5
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

What is the Transformer architecture?
What is BERT?
What is Retrieval Augmented Generation?
What is LoRA and how does it work?
What is Chain of Thought prompting?

The evaluation focused on:

Retrieval relevance
Correct paper retrieval
Answer relevance
Source grounding
Unsupported-question handling

The evaluation script is available in:

evaluation/evaluate_rag.py

Run the evaluation from the project root:

python .\evaluation\evaluate_rag.py

Or, when using the virtual environment:

.\.venv\Scripts\python.exe .\evaluation\evaluate_rag.py
🚫 Unsupported Question Handling

The system was also tested with questions that are unrelated to the research paper knowledge base.

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
│   └── evaluate_rag.py
│
├── notebooks/
│   └── research_paper_rag.ipynb
│
├── requirements.txt
│
└── README.md
💻 Installation
1. Clone the Repository
git clone <your-repository-url>

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

Navigate to the application directory:

cd app

Run the Streamlit application:

streamlit run app.py

The application will open in your browser.

🔍 Example Questions

Try asking:

What is the Transformer architecture?
What is BERT?
What is Retrieval Augmented Generation?
What is LoRA and how does it work?
What is Chain of Thought prompting?
⚠️ Limitations

The current system has several limitations:

Retrieval quality depends on the embedding model and chunking strategy.
Relevant information may not always appear in the top retrieved chunks.
The generated answer may be incomplete when the retrieved context is limited.
Strict grounding can reduce hallucination but may increase the number of unanswered questions.
Answer quality depends on the research papers available in the knowledge base.
Similar research papers may produce overlapping or partially relevant retrieval results.
🔮 Future Improvements

Possible future enhancements include:

Hybrid search using keyword and vector retrieval
Reranking retrieved document chunks
Query rewriting
Metadata filtering by research paper
Retrieval confidence thresholds
Conversational memory
Support for additional document formats
Direct PDF page citations
Automated quantitative RAG evaluation
Cloud deployment
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
Similarity Search
        ↓
Relevant Context Retrieval
        ↓
Strict RAG Prompt
        ↓
Ollama Llama 3.2
        ↓
Source-Grounded Answer

The project also includes experiments with different embedding models, chunking strategies, retrieval configurations, and prompting approaches.

The final system uses BGE embeddings, ChromaDB, similarity search, Llama 3.2 through Ollama, and a strict source-grounded prompting strategy to improve answer reliability and reduce unsupported responses.
