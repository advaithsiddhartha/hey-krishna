import os
import gc
import json
import uuid
import time
import faiss
import random
import psutil
import traceback
import tracemalloc
import numpy as np
import google.generativeai as genai

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env")

genai.configure(api_key=API_KEY)

# =========================================================
# FLASK SETUP
# =========================================================
app = Flask(__name__)

# =========================================================
# START MEMORY TRACKING
# =========================================================
tracemalloc.start()

# =========================================================
# DEBUG HELPERS
# =========================================================

def print_divider():
    print("\n" + "=" * 80 + "\n")

def log_memory(stage=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    rss = mem_info.rss / (1024 * 1024)
    vms = mem_info.vms / (1024 * 1024)

    current, peak = tracemalloc.get_traced_memory()

    print(f"""
[MEMORY] {stage}

RSS Memory : {rss:.2f} MB
VMS Memory : {vms:.2f} MB

Python Current Allocated : {current / 1024 / 1024:.2f} MB
Python Peak Allocated    : {peak / 1024 / 1024:.2f} MB
""")

def log_time(stage, start_time):
    elapsed = time.time() - start_time
    print(f"[TIME] {stage}: {elapsed:.2f} sec")

def log_exception(e):
    print_divider()
    print("❌ EXCEPTION OCCURRED")
    print(type(e).__name__)
    print(str(e))
    print("\nFULL TRACEBACK:\n")
    traceback.print_exc()
    print_divider()

# =========================================================
# MODEL HELPER
# =========================================================

def get_model(model_name: str):
    print(f"📦 Loading Gemini Model: {model_name}")
    return genai.GenerativeModel(model_name)

# =========================================================
# LOAD DATA + FAISS
# =========================================================

print_divider()
print("📖 Loading Bhagavad Gita JSON...")

with open("gita_verses.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} verses")

print("\n📦 Loading FAISS index...")

faiss_path = os.path.join(os.path.dirname(__file__), "gita_indexes.faiss")

index = faiss.read_index(faiss_path)

print("✅ FAISS Index Loaded")
print(f"📏 Vector Dimension: {index.d}")
print(f"📚 Total Indexed Vectors: {index.ntotal}")

log_memory("After FAISS Load")

print_divider()

# =========================================================
# VECTOR SEARCH
# =========================================================

def find_relevant_verses(query, k=3):

    try:
        print_divider()
        print("🔍 STARTING VECTOR SEARCH")
        print(f"📝 Query: {query}")

        embedding_start = time.time()

        log_memory("Before Embedding")

        response = genai.embed_content(
            model="models/embedding-001",
            content=query
        )

        embedding = response["embedding"]

        log_time("Embedding Generation", embedding_start)

        query_embedding = np.array([embedding], dtype="float32")

        print(f"✅ Embedding Shape: {query_embedding.shape}")

        search_start = time.time()

        log_memory("Before FAISS Search")

        distances, indices = index.search(query_embedding, k)

        log_time("FAISS Search", search_start)

        print("\n📊 SEARCH RESULTS")

        for rank, idx in enumerate(indices[0]):
            print(f"""
Result #{rank + 1}

Index: {idx}
Distance: {distances[0][rank]}

Verse:
{data[idx]["english"][:200]}
""")

        gc.collect()

        log_memory("After FAISS Search")

        print_divider()

        return [data[i] for i in indices[0]]

    except Exception as e:
        log_exception(e)
        raise

# =========================================================
# ROUTES
# =========================================================

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

# =========================================================
# MAIN ASK ROUTE
# =========================================================

@app.route("/ask", methods=["POST"])
def ask():

    request_id = str(uuid.uuid4())[:8]

    try:

        print_divider()
        print(f"🚀 NEW REQUEST [{request_id}]")

        total_start = time.time()

        req = request.json

        print("\n📨 RAW REQUEST JSON:")
        print(json.dumps(req, indent=2))

        name = req.get("name", "Friend")
        age = req.get("age", "unknown")
        query = req.get("query", "")
        language = req.get("language", "English")
        mode = req.get("mode", "normal")

        print(f"""
👤 User Details

Name     : {name}
Age      : {age}
Language : {language}
Mode     : {mode}

Question:
{query}
""")

        # =====================================================
        # FIND VERSES
        # =====================================================

        retrieval_start = time.time()

        results = find_relevant_verses(query)

        log_time("Total Retrieval Pipeline", retrieval_start)

        # =====================================================
        # BUILD PROMPT
        # =====================================================

        prompt = f"""
You are Lord Krishna, giving guidance to {name}, who is {age} years old.

{name}'s Question:
{query}

Your task:
1. Provide a compassionate, mentor-like answer appropriate for a {age}-year-old.
2. Use all the three verses given.
3. Whenever quoting verses:
   - Wrap Sanskrit verse in <b> tags
   - Wrap translation in <b> tags
   - Explain each verse
4. Format properly in HTML
5. Maintain spacing
6. Keep language simple
7. No markdown asterisks

Knowledge Base:
{results}

IMPORTANT:
- Respond fully in {language}
- Include Sanskrit + translation
- Detailed but meaningful
- End with divine Krishna closing statement
"""

        print_divider()
        print("🧠 PROMPT GENERATED")

        print(f"\n📏 Prompt Length: {len(prompt)} characters")

        preview = prompt[:1500]

        print(f"\n📄 PROMPT PREVIEW:\n\n{preview}")

        print_divider()

        # =====================================================
        # MODEL SELECTION
        # =====================================================

        llm_model = (
            "gemini-2.5-pro"
            if mode == "deep"
            else "gemini-2.5-flash"
        )

        llm = get_model(llm_model)

        # =====================================================
        # GENERATE RESPONSE
        # =====================================================

        generation_start = time.time()

        print(f"⚡ Sending request to Gemini: {llm_model}")

        response = llm.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 1200,
                "temperature": 0.8
            }
        )

        log_time("LLM Generation", generation_start)

        print("✅ Gemini Response Received")

        # =====================================================
        # RAW RESPONSE DEBUG
        # =====================================================

        print_divider()
        print("📦 RAW GEMINI RESPONSE OBJECT")
        print(response)

        krishna_response = response.text

        print(f"\n📏 Response Length: {len(krishna_response)} chars")

        print("\n📄 RESPONSE PREVIEW:\n")
        print(krishna_response[:2000])

        # =====================================================
        # CLEAN RESPONSE
        # =====================================================

        if krishna_response.startswith("<html>"):
            krishna_response = krishna_response[6:]

        if krishna_response.endswith("</html>"):
            krishna_response = krishna_response[:-7]

        gc.collect()

        log_memory("After Response Generation")

        total_elapsed = time.time() - total_start

        print_divider()

        print(f"""
✅ REQUEST COMPLETE [{request_id}]

⏱ Total Time: {total_elapsed:.2f} sec
📦 Model Used: {llm_model}
📚 Retrieved Verses: {len(results)}
""")

        print_divider()

        return jsonify({
            "success": True,
            "response": krishna_response,
            "verses": results,
            "request_id": request_id,
            "model_used": llm_model
        })

    except Exception as e:

        log_exception(e)

        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    print_divider()

    print("""
🌸 KRISHNA AI SERVER STARTING
""")

    log_memory("Initial Startup")

    app.run(
        debug=True,
        host="0.0.0.0",
        port=7970
    )
