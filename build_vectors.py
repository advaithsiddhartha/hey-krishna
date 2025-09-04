import faiss
import json
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Gita JSON (english only)
with open("gita_verses.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Embed all verses with Gemini
texts = [d["english"] for d in data]
embeddings = []
for text in texts:
    emb = genai.embed_content(model="models/embedding-001", content=text)["embedding"]
    embeddings.append(emb)

embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save
faiss.write_index(index, "gita_indexes.faiss")
print(f"FAISS index built with dim={dim}, total={len(embeddings)}")
