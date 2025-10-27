# config_example.py — copy to config.py and fill with real values.
NEO4J_URI = "neo4j+s://b8e20b60.databases.neo4j.io" 
NEO4J_USER = "neo4j" #neo4j
NEO4J_PASSWORD = "U4XX6cIxXzfB5Cm0ZJpStXWz95w-brQVy1nAGFYNb7c"

GOOGLE_API_KEY = "AIzaSyB3Qqaahm1l310PthzZQlJnG9gVhZbaGac"

PINECONE_API_KEY = "pcsk_2EK5Av_MXSzRukCcHr3krTFQtNNtfJJe8UQjGDTRnou1Yf4ampEG4A66RvQKaEQbwahcXa"
PINECONE_ENV = "us-east1-gcp"   # example
PINECONE_INDEX_NAME = "vietnam-travel-e5-improved"
PINECONE_VECTOR_DIM = 1024      # adjust to embedding model used (text-embedding-3-large ~ 3072? check your model); we assume 1536 for common OpenAI models — change if needed.
EMBED_MODEL = "intfloat/multilingual-e5-large"
