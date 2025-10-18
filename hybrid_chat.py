# hybrid_chat.py
import json
from datetime import datetime
import shelve 
from typing import List
import asyncio
from time import perf_counter
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from google.api_core.exceptions import NotFound
import config

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "intfloat/multilingual-e5-large"
CHAT_MODEL = "gemini-flash-latest"
TOP_K = 20
CACHE_FILE = "cache/embedding_cache.shelf" 
LOG_FILE = "logging/logs.txt"

INDEX_NAME = config.PINECONE_INDEX_NAME
_GOOGLE_API_KEY = getattr(config, "GOOGLE_API_KEY", None)

def load_embedding_model(model_name: str):
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Embedding model loaded.")
    return model

def initialize_pinecone(api_key: str, index_name: str, dimension: int, region: str):
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        print(f"Creating managed index: {index_name}")
        pc.create_index(name=index_name, dimension=dimension, metric="cosine", spec=ServerlessSpec(cloud="gcp", region=region))
    print(f"Connecting to Pinecone index: {index_name}...")
    return pc.Index(index_name)

def initialize_neo4j(uri: str, user: str, password: str):
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print("Neo4j connection successful.")
    return driver

def initialize_chat_model(api_key: str, model_name: str):
    print(f"Initializing chat model: {model_name}...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    print("Chat model initialized.")
    return model

# -----------------------------
# Helper functions
# -----------------------------
def embed_text(text: str, model, cache) -> List[float]:
    """Get embedding for a text string, using a cache to avoid re-computation."""
    # Check if the embedding is already in the cache
    if text in cache:
        print("DEBUG: Embedding cache hit!")
        return cache[text]
    
    print("DEBUG: Embedding cache miss. Computing new embedding.")
    # E5 models require 'query: ' prefix for queries
    emb = model.encode([f"query: {text}"], convert_to_tensor=False)
    embedding_list = emb[0].tolist() if hasattr(emb[0], 'tolist') else list(emb[0])
    
    # Store the new embedding in the cache before returning
    cache[text] = embedding_list
    return embedding_list

def pinecone_query(query_text: str, embedding_model, index, cache, top_k=TOP_K):
    """Query Pinecone index using embedding."""
    vec = embed_text(query_text, embedding_model, cache) # Pass cache to embed_text
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    print(f"DEBUG: Found {len(res['matches'])} matches in Pinecone.")
    return res["matches"]

def fetch_graph_context(node_ids: List[str], driver):
    facts = []
    cypher_default = (
        "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
        "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, m.name AS name, m.type AS type, m.description AS description "
        "LIMIT 3"
    )
    cypher_node_type = "MATCH (n:Entity {id:$nid}) RETURN n.type AS type"
    cypher_by_kind = (
        "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
        "WHERE ( "
        "  (m.type IS NOT NULL AND toLower(m.type) IN $kind_aliases) OR "
        "  (m.category IS NOT NULL AND toLower(m.category) IN $kind_aliases) OR "
        "  any(lbl IN labels(m) WHERE toLower(lbl) IN $kind_label_aliases) "
        ") "
        "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, m.name AS name, m.type AS type, m.description AS description "
        "LIMIT 5"
    )
    with driver.session() as session:
        for nid in node_ids:
            # Determine if the node is a city by property or by id pattern
            n_type_rec = session.run(cypher_node_type, nid=nid).single()
            n_type = n_type_rec["type"] if n_type_rec else None
            is_city = (n_type == "city") or str(nid).lower().startswith("city_")

            if is_city:
                for kind in ("hotel", "attraction", "activity"):
                    aliases = [kind, f"{kind}s"]
                    recs = session.run(
                        cypher_by_kind,
                        nid=nid,
                        kind_aliases=aliases,
                        kind_label_aliases=aliases,
                    )
                    for r in recs:
                        facts.append({
                            "source": nid,
                            "rel": r["rel"],
                            "target_id": r["id"],
                            "target_name": r["name"],
                            "target_desc": (r["description"] or "")[:400],
                            "labels": r["labels"],
                        })
            else:
                recs = session.run(cypher_default, nid=nid)
                for r in recs:
                    facts.append({
                        "source": nid,
                        "rel": r["rel"],
                        "target_id": r["id"],
                        "target_name": r["name"],
                        "target_desc": (r["description"] or "")[:400],
                        "labels": r["labels"],
                    })
    print(f"DEBUG: Fetched {len(facts)} graph facts from Neo4j.")
    return facts

async def fetch_graph_context_async(node_ids: List[str], driver):
    """Fetch neighboring nodes concurrently for each node id using asyncio thread offloading."""
    cypher_default = (
        "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
        "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, m.name AS name, m.type AS type, m.description AS description "
        "LIMIT 3"
    )
    cypher_node_type = "MATCH (n:Entity {id:$nid}) RETURN n.type AS type"
    cypher_by_kind = (
        "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
        "WHERE ( "
        "  (m.type IS NOT NULL AND toLower(m.type) IN $kind_aliases) OR "
        "  (m.category IS NOT NULL AND toLower(m.category) IN $kind_aliases) OR "
        "  any(lbl IN labels(m) WHERE toLower(lbl) IN $kind_label_aliases) "
        ") "
        "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, m.name AS name, m.type AS type, m.description AS description "
        "LIMIT 5"
    )

    def _fetch_one(nid: str):
        results: List[dict] = []
        with driver.session() as session:
            n_type_rec = session.run(cypher_node_type, nid=nid).single()
            n_type = n_type_rec["type"] if n_type_rec else None
            is_city = (n_type == "city") or str(nid).lower().startswith("city_")

            if is_city:
                for kind in ("hotel", "attraction", "activity"):
                    aliases = [kind, f"{kind}s"]
                    recs = session.run(
                        cypher_by_kind,
                        nid=nid,
                        kind_aliases=aliases,
                        kind_label_aliases=aliases,
                    )
                    for r in recs:
                        results.append({
                            "source": nid,
                            "rel": r["rel"],
                            "target_id": r["id"],
                            "target_name": r["name"],
                            "target_desc": (r["description"] or "")[:400],
                            "labels": r["labels"],
                        })
            else:
                recs = session.run(cypher_default, nid=nid)
                for r in recs:
                    results.append({
                        "source": nid,
                        "rel": r["rel"],
                        "target_id": r["id"],
                        "target_name": r["name"],
                        "target_desc": (r["description"] or "")[:400],
                        "labels": r["labels"],
                    })
        return results

    tasks = [asyncio.to_thread(_fetch_one, nid) for nid in node_ids]
    groups = await asyncio.gather(*tasks, return_exceptions=False)
    facts: List[dict] = []
    for g in groups:
        facts.extend(g)
    print(f"DEBUG: Fetched {len(facts)} graph facts from Neo4j (async).")
    return facts

def build_prompt(user_query, pinecone_matches, graph_facts, prior_assistant_response: str | None = None):
    system = ("You are a helpful travel assistant. Use the provided semantic search results and graph facts to answer the user's query concisely. For any response suggest cities, hotels, attractions, activities. Cite node ids when referencing specific places or attractions or activities or hotels.")
    vec_context = []
    for m in pinecone_matches:
        meta = m["metadata"]
        score = m.get("score", 0)
        snippet = f"- id: {m['id']}, name: {meta.get('name','')}, type: {meta.get('type','')}, tags: {meta.get('tags', [])}, score: {score:.2f}"
        #snippet = f"- id: {m['id']}, score: {score:.2f}"
        if meta.get("city"):
            snippet += f", city: {meta.get('city')}"
        vec_context.append(snippet)
    graph_context = [f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}" for f in graph_facts]
    prior_block = ""
    if prior_assistant_response:
        prior_block = f"\n\nPRIOR HISTORY OF LAST ASSISTANT RESPONSE:\n{prior_assistant_response.strip()}\n"
    prompt = (
        f"SYSTEM: {system}\n\n"
        f"USER QUERY: {user_query}"
        f"{prior_block}\n\n"
        "CONTEXT FROM SEMANTIC SEARCH:\n" + "\n".join(vec_context) +
        "\n\nCONTEXT FROM GRAPH DATABASE:\n" + "\n".join(graph_context) +
        "\n\nRESPONSE: Based on the provided context, answer the user's question. If helpful, suggest 2-3 concrete itinerary steps or tips and mention node ids for references."
    )
    return prompt

def call_chat(prompt, gemini_model):
    try:
        resp = gemini_model.generate_content(prompt)
        return resp.text
    except NotFound as nf:
        return f"Error: Model '{CHAT_MODEL}' not found or unavailable. {nf.message}"
    except Exception as e:
        return f"An error occurred while calling the Gemini API: {e}"

def log_turn(log_path: str, entry: dict):
    """Append a structured log entry to logs.txt."""
    try:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **entry,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("-----\n")
            f.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
            f.write("\n")
    except Exception as e:
        # Best-effort logging; don't crash the app
        print(f"[LOGGING ERROR] {e}")
# -----------------------------
# Interactive chat loop
# -----------------------------
# --- Update function signature to accept the cache ---
def interactive_chat(embedding_model, pinecone_index, neo4j_driver, chat_model, cache):
    """Main interactive chat loop."""
    print("\nHybrid travel assistant. Type 'exit' to quit.")
    last_assistant_response: str | None = None
    while True:
        query = input("\nEnter your travel question: ").strip()
        if not query or query.lower() in ("exit","quit"):
            break

        matches = pinecone_query(query, embedding_model, pinecone_index, cache, top_k=TOP_K) # Pass cache
        match_ids = [m["id"] for m in matches]
        # Fetch graph facts concurrently for each match id
        try:
            #_t0 = perf_counter()
            #graph_facts = asyncio.run(fetch_graph_context_async(match_ids, neo4j_driver))
            #_t1 = perf_counter()
            #print(f"TIMING: fetch_graph_context_async took {(_t1 - _t0)*1000:.1f} ms for {len(graph_facts)} facts.")
            graph_facts = fetch_graph_context(match_ids, neo4j_driver)
        except RuntimeError:
            graph_facts = fetch_graph_context(match_ids, neo4j_driver)
        prompt = build_prompt(query, matches, graph_facts, prior_assistant_response=last_assistant_response)
        answer = call_chat(prompt, chat_model)
        
        # Append a structured log entry for this turn
        log_turn(
            LOG_FILE,
            {
                "user_query": query,
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
            },
        )

        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n======================\n")

        # Update conversational memory with the latest assistant response
        last_assistant_response = answer

# -----------------------------
# Main execution block
# -----------------------------
if __name__ == "__main__":
    neo4j_driver = None
    try:
        # --- CHANGE 2: Open the cache file using a 'with' block ---
        with shelve.open(CACHE_FILE) as cache:
            # Initialize all clients and models
            embedding_model = load_embedding_model(EMBED_MODEL)
            pinecone_index = initialize_pinecone(
                api_key=config.PINECONE_API_KEY, index_name=INDEX_NAME,
                dimension=config.PINECONE_VECTOR_DIM, region=getattr(config, "PINECONE_ENV", "us-east1-gcp")
            )
            neo4j_driver = initialize_neo4j(
                uri=config.NEO4J_URI, user=config.NEO4J_USER, password=config.NEO4J_PASSWORD
            )
            chat_model = initialize_chat_model(
                api_key=_GOOGLE_API_KEY, model_name=CHAT_MODEL
            )
            
            # Start the chat loop, passing the initialized objects AND the cache
            interactive_chat(embedding_model, pinecone_index, neo4j_driver, chat_model, cache)

    except Exception as e:
        print(f"\nAn error occurred during setup or execution: {e}")

    finally:
        if neo4j_driver:
            try:
                neo4j_driver.close()
                print("\nNeo4j connection closed.")
            except Exception as e:
                print(f"Error closing Neo4j driver: {e}")