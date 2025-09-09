from flask import Flask, render_template, request, jsonify
import faiss
import json
import numpy as np
import google.generativeai as genai
import random
import itertools
import os
from dotenv import load_dotenv
import gc
import tracemalloc
import psutil

# -------------------
# Setup Flask + dotenv
# -------------------
app = Flask(__name__)
load_dotenv()

# -------------------
# Memory logging functions
# -------------------
def log_memory(stage=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[MEMORY] {stage} | RSS: {mem_info.rss / (1024*1024):.2f} MB, VMS: {mem_info.vms / (1024*1024):.2f} MB")

tracemalloc.start()

# -------------------
# Load API Keys
# -------------------
api_keys_str = os.getenv("API_KEYS", "")
API_KEYS = [k.strip() for k in api_keys_str.split(",") if k.strip()]
if not API_KEYS:
    raise ValueError("No API keys found. Please set API_KEYS in your environment.")
key_cycle = itertools.cycle(API_KEYS)

def get_next_model(model_name: str):
    api_key = next(key_cycle)
    genai.configure(api_key="AIzaSyDEEsQ2Mu0SnH9CQW6ZRlJggZ_jlASGas4")
    return genai.GenerativeModel(model_name)

# -------------------
# Load FAISS index and data
# -------------------
with open("gita_verses.json", "r", encoding="utf-8") as f:
    data = json.load(f)

faiss_path = os.path.join(os.path.dirname(__file__), "gita_indexes.faiss")
index = faiss.read_index(faiss_path)
print("Loaded FAISS index ✅")
log_memory("After loading FAISS index")

def find_relevant_verses(query, k=3):
    log_memory("Before embedding query")
    embedding = genai.embed_content(model="models/embedding-001", content=query)["embedding"]
    query_embedding = np.array([embedding], dtype="float32")
    log_memory("Before FAISS search")
    distances, indices = index.search(query_embedding, k)
    gc.collect()
    log_memory("After FAISS search")
    return [data[i] for i in indices[0]]

# -------------------
# Routes
# -------------------
@app.route("/")
def home():
    random_num = random.randint(1, 15)
    image_file = f"images/{random_num}.jpg"
    return render_template("index.html", image_file=image_file)

@app.route("/ask-page")
def ask_page():
    random_num = random.randint(1, 15)
    image_file = f"images/{random_num}.jpg"
    return render_template("ask.html", image_file=image_file)

@app.route("/tech")
def tech():
    return render_template("tech.html")

@app.route("/ask", methods=["POST"])
def ask():
    req = request.json
    name = req.get("name", "Friend")
    age = req.get("age", "unknown")
    query = req.get("query", "")
    language = req.get("language", "English")
    mode = req.get("mode", "normal")

    log_memory("Before finding relevant verses")
    results = find_relevant_verses(query)

    prompt = f"""
    You are Lord Krishna, giving guidance to {name}, who is {age} years old.

    {name}'s Question:
    {query}
    
    Your task:
    1. Provide a compassionate, mentor-like answer appropriate for a {age}-year-old.

    """

    llm_model = "gemini-2.5-pro" if mode == "deep" else "gemini-2.5-flash"
    llm = get_next_model(llm_model)

    log_memory("Before generating LLM content")
    response = llm.generate_content(prompt)
    krishna_response = response.text

    if krishna_response.startswith("```html"):
        krishna_response = krishna_response[7:]
    if krishna_response.endswith("```"):
        krishna_response = krishna_response[:-3]

    gc.collect()
    log_memory("After generating LLM content")

    return jsonify({
        "response": krishna_response,
        "verses": results
    })

# -------------------
# Run App
# -------------------
if __name__ == "__main__":
    app.run(debug=True, port=7970)
