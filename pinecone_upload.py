# pinecone_upload.py
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import config

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32

INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = config.PINECONE_VECTOR_DIM  # 1024 for multilingual-e5-large

# -----------------------------
# Helper functions
# -----------------------------

def load_embedding_model():
    """Loads the SentenceTransformer model from Hugging Face."""
    print("Loading multilingual-e5-large model...")
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    print("Model loaded.")
    return model

def initialize_pinecone_index(pinecone_client, index_name, vector_dim):
    """Checks for an existing Pinecone index and creates one if it doesn't exist."""
    existing_indexes = pinecone_client.list_indexes().names()
    if index_name not in existing_indexes:
        print(f"Creating managed index: {index_name}")
        pinecone_client.create_index(
            name=index_name,
            dimension=vector_dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="gcp",
                region="us-east1-gcp"
            )
        )
    else:
        print(f"Index {index_name} already exists.")
    
    # Connect to the index
    return pinecone_client.Index(index_name)

def get_embeddings(texts, model):
    """Generate embeddings using a local SentenceTransformer model."""
    # The E5 model requires a prefix for documents being indexed
    texts_with_prefix = [f"passage: {text}" for text in texts]
    
    # The model.encode() method returns numpy arrays; convert them to lists
    embeddings = model.encode(texts_with_prefix, convert_to_tensor=False)
    return [embedding.tolist() for embedding in embeddings]

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    # Initialize clients and models
    embedding_model = load_embedding_model()
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = initialize_pinecone_index(pc, INDEX_NAME, VECTOR_DIM)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        text_parts = []
        if node.get("semantic_text"):
            text_parts.append(node.get("semantic_text"))
        if node.get("description"):
            text_parts.append(node.get("description"))
        
        semantic_text = ". ".join(text_parts)[:1000]

        if not semantic_text.strip():
            continue
        
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        # If the 'best_time_to_visit' key exists in the node, add it to the meta dictionary
        if "best_time_to_visit" in node:
            meta["best_time_to_visit"] = node["best_time_to_visit"]

        items.append((node["id"], semantic_text, meta))

    print(f"Preparing to upsert {len(items)} items to Pinecone...")

    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = get_embeddings(texts, model=embedding_model)

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        index.upsert(vectors)

    print("All items uploaded successfully.")

# -----------------------------
if __name__ == "__main__":
    main()
