from langchain_community.embeddings import JinaEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

embedding = JinaEmbeddings(
    model_name="jina-embeddings-v2-base-en",
    jina_api_key=os.getenv("JINA_API_KEY")
)

vector = embedding.embed_query("What is artificial intelligence?")

print(len(vector))
print(vector[:10])