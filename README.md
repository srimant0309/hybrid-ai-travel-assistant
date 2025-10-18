# Hybrid Travel AI Assistant

A retrieval-augmented conversational assistant for travel queries. It combines Pinecone vector search(a semantic vectorDB), Neo4j graph database, and Google Gemini generative models, with embeddings from Hugging Face multilingual-e5-large, presented via a simple Gradio User Interface.

## Additional Features
- Retrieval-Augmented Generation (RAG) using Pinecone + Neo4j
- Conversational memory (stores context of previous user query/prompt)
- Embedding cache for repeated prompts
- Async parallel Neo4j lookups after Pinecone retrieval
- Clean local Gradio UI
- Structured logging of inputs, retrieval results, and outputs

## Architecture Overview
1) User prompt → embed (cache if available)
2) Pinecone top-K retrieval
3) Neo4j queries for context expansion
4) Query context + prompt → Gemini (gemini-pro) generation
5) Stream result to Gradio UI

Key files:
- `gradio_app.py` — UI entrypoint
- `hybrid_chat.py` — chat/RAG pipeline
- `config.py` — API keys and DB settings
- `pinecone_upload.py` — Pinecone index creation/upload
- `load_to_neo4j.py` — dataset insertion to Neo4j
- `visualize_graph.py` — optional graph view helper
- `improvements.md` — detailed list of changes and features

## Setup
1) Create/activate a virtual environment (or use `task_venv/`).
2) Install dependencies from `requirements.txt`.
    ```pip install -r requirements.txt```
3) Configure `config.py` with:
   - Gemini API key
   - Pinecone API key, index, environment/region
   - Neo4j URI, username, password, database
   - Embedding model id (multilingual-e5-large)/embedding size

## Data Insertion
- Neo4j: Run `load_to_neo4j.py` to load entities/relations (from `vietnam_travel_dataset.json`). 
- Pinecone: Run `pinecone_upload.py` to create and populate the index based on `config.py`.

## Run the App
Run the Gradio UI:

```bash
python gradio_app.py
```

Then open the printed local URL in your browser to interact with the travel assistant.

## Logging
- Logs are written to `logging/logs.txt`.
- Includes prompt text, Pinecone and Neo4j results, and the final LLM response.

## More Details
See `improvements.md` for a comprehensive list of fixes, updates, and feature implementations.
