Insurance AI Assistant – Powered by RAG

A Retrieval-Augmented Generation (RAG) system designed to help users accurately answer queries related to insurance policy documents. The system combines LLM capabilities with document-grounded retrieval to generate reliable, context-aware responses.

📌 Problem Statement

Insurance policy documents are long, complex, and often difficult for users to interpret. Customers frequently ask questions about:

Eligibility

Coverage conditions

Exclusions

Claim rules

Benefits and terminology

The challenge is to build a system that can:

“Answer user queries based on the actual policy documents using Retrieval-Augmented Generation (RAG).”

This ensures responses are grounded in factual, document-based evidence while maintaining the natural language fluency of Large Language Models.

🧠 System Design Overview

The RAG architecture follows the flow shown in the System Design – LLM diagram and consists of these components:

1. Data Ingestion

Policy PDFs and DOCX files are loaded using SimpleDirectoryReader.

Text is extracted and converted into document objects.

2. Vector Indexing

Documents are chunked and embedded using LlamaIndex.

These embeddings are stored in a VectorStoreIndex for efficient semantic search.

3. Retrieval

Upon receiving a query, the system retrieves the most relevant nodes.

Reranking is performed using Cohere Rerank.

similarity_top_k ensures only the best matches are used.

4. LLM Generation

Retrieved context is passed into OpenAI GPT-3.5-Turbo.

Custom prompt templates guide the LLM:

textQATemplate for initial responses

refinedTemplate for refinement with additional context

5. Post-Processing

Responses are refined if needed.

A caching mechanism is applied to improve performance and reduce API usage.

Data Flow
User Query → Retriever → Relevant Nodes → LLM (GPT-3.5-Turbo) → Final Answer

🛠️ Technologies Used

Below are the key libraries installed via pip and their roles:

🔹 LlamaIndex

llama-index, llama-index-llms-openai

Core framework for building RAG systems

Handles ingestion, indexing, retrieval, and integration with LLMs

🔹 OpenAI

openai

Provides access to GPT-3.5-Turbo for generating human-like responses

🔹 PyPDF

pypdf

Extracts text from PDF policy documents

🔹 docx2txt

docx2txt

Extracts text from DOCX files

🔹 Cohere Rerank

llama-index-postprocessor-cohere-rerank

Improves retrieval accuracy by reranking nodes

🔹 diskcache

diskcache

Lightweight on-disk caching system to store query responses

⚙️ Setup & Installation
1. Install Required Libraries
pip install llama-index
pip install llama-index-llms-openai
pip install openai
pip install pypdf
pip install docx2txt
pip install llama-index-postprocessor-cohere-rerank
pip install diskcache

2. Mount Google Drive (for Colab setups)
from google.colab import drive
drive.mount('/content/drive')

3. Initialize OpenAI API Key
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

📂 Data Ingestion

Policy documents must be placed in:

/content/drive/MyDrive/hdfc-insurance-policy/policy-docs


Supported formats include:

PDF

DOCX

Load the documents:

from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    "/content/drive/MyDrive/hdfc-insurance-policy/policy-docs"
).load_data()

📌 Query Engine & Prompt Engineering
Vector Index Construction
from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)
queryEngine = index.as_query_engine(similarity_top_k=5)

LLM Configuration
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")

Custom Prompt Templates
textQATemplate
Context information is provided below
---------------------
{contextStr}
---------------------
Using both the context and your own knowledge, answer the question: {queryStr}
If the context isn’t helpful, respond using your own understanding.

refinedTemplate
The original question is: {queryStr}
Existing answer: {existingAnswer}

Additional context:
------------
{contextMsg}
------------

Using context and your own knowledge, update or repeat the answer.

🔥 Caching Mechanism

To improve speed and reduce redundant API calls, diskcache is used:

import diskcache as dc
cache = dc.Cache("./gpt_cache")

queryResponse Function
def queryResponse(userInput):
    cacheResponse = cache.get(userInput)

    if cacheResponse is None:
        response = queryEngine.query(userInput)
        fileName = response.source_nodes[0].node.metadata['file_name']

        enhancedResponse = (
            response.response +
            "\nReference Document: " + fileName +
            "\nSimilarity Score: " + str(response.source_nodes[1].score)
        )

        cache.set(userInput, enhancedResponse)
        return enhancedResponse

    return cacheResponse

💬 Usage Example
Run a Query
queryResponse("What are Accidental Death Benefits?")

Example Output
Accidental Death Benefit refers to...
Reference Document: policy123.pdf
Similarity Score: 0.89

🧪 Validation & Feedback Loop

The validatePipeline function allows controlled testing of the RAG system.

Function Purpose

Runs multiple predefined questions

Displays the system response

Asks user whether the answer is Good or Bad

Stores results in a DataFrame

Feedback Storage

The resulting DataFrame includes:

Question	Response	Good or Bad

This helps evaluate accuracy and continuously improve prompts, chunking, and indexing.

🏁 Conclusion / Key Findings

This project demonstrates the successful implementation of a RAG-based Insurance AI Assistant using LlamaIndex and OpenAI.

Key achievements:

Built an end-to-end retrieval-augmented QA system

Achieved grounded, accurate responses based on policy documents

Implemented caching to optimize performance

Used Cohere Rerank to improve retrieval precision

Added a feedback loop for iterative refinement

Key Learnings:

Prompt engineering significantly impacts quality

Reranking greatly improves retrieval accuracy

Caching is crucial for real-world deployment

LlamaIndex simplifies RAG pipeline creation

The system is scalable and can be extended for:

Customer-facing chatbots

Insurance support automation

Policy comparison tools

Compliance assurance systems

📎 Repository Structure (Suggested)
📁 Insurance-AI-RAG
 ├── README.md
 ├── /policy-docs
 ├── main.ipynb
 ├── gpt_cache/
 └── requirements.txt