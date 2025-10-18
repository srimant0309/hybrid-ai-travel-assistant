# config_example.py — copy to config.py and fill with real values.
NEO4J_URI = "neo4j+s://[db_id].databases.neo4j.io" 
NEO4J_USER = "neo4j" #neo4j
NEO4J_PASSWORD = "YourPasswordHere"

GOOGLE_API_KEY = "YourGoogleAPIKeyHere"

PINECONE_API_KEY = "YourPineconeAPIKeyHere"
PINECONE_ENV = "us-east1-gcp"   # example
PINECONE_INDEX_NAME = "your-index-name"
PINECONE_VECTOR_DIM = 1024      # adjust to embedding model used (text-embedding-3-large ~ 3072? check your model); we assume 1536 for common OpenAI models — change if needed.
EMBED_MODEL = "intfloat/multilingual-e5-large"
