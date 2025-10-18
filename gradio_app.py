"""
Gradio UI for Hybrid Travel Assistant

This app wraps the existing pipeline (embeddings + Pinecone + Neo4j + Gemini)
from `my_hc_new_features.py` and provides a simple web UI with conversational
memory of the last assistant response.
"""
from __future__ import annotations

import os
import atexit
from typing import Optional, Tuple

import gradio as gr

import config
from hybrid_chat import (
    EMBED_MODEL,
    CHAT_MODEL,
    TOP_K,
    CACHE_FILE,
    LOG_FILE,
    INDEX_NAME,
    load_embedding_model,
    initialize_pinecone,
    initialize_neo4j,
    initialize_chat_model,
    pinecone_query,
    fetch_graph_context,
    build_prompt,
    call_chat,
    log_turn,
)
import shelve


# Initialize services once at startup
embedding_model = load_embedding_model(EMBED_MODEL)
pinecone_index = initialize_pinecone(
    api_key=config.PINECONE_API_KEY,
    index_name=INDEX_NAME,
    dimension=config.PINECONE_VECTOR_DIM,
    region=getattr(config, "PINECONE_ENV", "us-east1-gcp"),
)
neo4j_driver = initialize_neo4j(
    uri=config.NEO4J_URI, user=config.NEO4J_USER, password=config.NEO4J_PASSWORD
)
chat_model = initialize_chat_model(
    api_key=getattr(config, "GOOGLE_API_KEY", None), model_name=CHAT_MODEL
)


# Ensure directories for cache and logging exist
os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
if LOG_FILE:
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

# Open the embedding cache. To avoid concurrency issues, we'll set queue(concurrency_count=1)
cache = shelve.open(CACHE_FILE)


def _cleanup():
    try:
        cache.close()
    except Exception:
        pass
    try:
        neo4j_driver.close()
    except Exception:
        pass


atexit.register(_cleanup)


def chat_fn(user_query: str, last_assistant_response: Optional[str]) -> Tuple[str, Optional[str]]:
    if not user_query or not user_query.strip():
        return "Please enter a question.", last_assistant_response

    # Pinecone search
    matches = pinecone_query(user_query, embedding_model, pinecone_index, cache, top_k=TOP_K)
    match_ids = [m["id"] for m in matches]

    # Graph context
    graph_facts = fetch_graph_context(match_ids, neo4j_driver)

    # Prompt and model call
    prompt = build_prompt(
        user_query,
        matches,
        graph_facts,
        prior_assistant_response=last_assistant_response,
    )
    answer = call_chat(prompt, chat_model)

    # Log the turn
    log_turn(
        LOG_FILE,
        {
            "user_query": user_query,
            "top_k": TOP_K,
            "pinecone_matches": matches,
            "match_ids": match_ids,
            "graph_facts": graph_facts,
            "prompt": prompt,
            "response": answer,
            "prior_assistant_response": last_assistant_response,
            "models": {
                "embedding_model": EMBED_MODEL,
                "chat_model": CHAT_MODEL,
            },
            "ui": "gradio",
        },
    )

    # Return answer and update the session's last response
    return answer, answer


def clear_fn() -> Tuple[str, Optional[str], str]:
    return "", None, ""


with gr.Blocks(title="Hybrid Travel Assistant") as demo:
    gr.Markdown("# Hybrid Travel Assistant\nAsk travel questions with semantic search + graph context.")

    last_response_state = gr.State(value=None)  # Holds last assistant response

    with gr.Row():
        user_in = gr.Textbox(label="Ask a travel question", lines=3, placeholder="e.g., What are must-see places in Hanoi for 2 days?")
    with gr.Row():
        ask_btn = gr.Button("Ask")
        clear_btn = gr.Button("Clear")

    answer_out = gr.Markdown(label="Assistant Answer")

    # Bind events
    ask_btn.click(chat_fn, inputs=[user_in, last_response_state], outputs=[answer_out, last_response_state])
    user_in.submit(chat_fn, inputs=[user_in, last_response_state], outputs=[answer_out, last_response_state])

    # Clear resets textbox, state, and output
    clear_btn.click(clear_fn, inputs=None, outputs=[user_in, last_response_state, answer_out])

# Enable queuing (use default concurrency for this Gradio version)
demo.queue()

if __name__ == "__main__":
    # Launch the app on localhost. Adjust server_port if needed.
    demo.launch()
