from flask import Flask, render_template, request, jsonify
import faiss
import json
import numpy as np
import google.generativeai as genai
import random
import os
import gc
import tracemalloc
import psutil
import traceback
import time
from dotenv import load_dotenv

# =========================================================
# LOAD ENV VARIABLES
# =========================================================
load_dotenv()

# =========================================================
# SETUP FLASK
# =========================================================
app = Flask(__name__)

# =========================================================
# MEMORY TRACKING
# =========================================================
tracemalloc.start()

def log_memory(stage=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    print(f"""
================ MEMORY LOG ================

STAGE : {stage}

RSS MEMORY : {mem_info.rss / (1024*1024):.2f} MB
VMS MEMORY : {mem_info.vms / (1024*1024):.2f} MB

============================================
""")

# =========================================================
# LOAD GEMINI API KEY
# =========================================================
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("""
❌ GEMINI_API_KEY NOT FOUND

Create a .env file and add:

GEMINI_API_KEY=your_api_key_here
""")

print("✅ Gemini API Key Loaded")

# Configure Gemini
genai.configure(api_key=API_KEY)

# =========================================================
# GEMINI MODEL HELPER
# =========================================================
def get_model(model_name: str):
    print(f"\n📦 Loading Gemini Model: {model_name}")
    return genai.GenerativeModel(model_name)

# =========================================================
# LOAD GITA DATA
# =========================================================
print("\n📖 Loading Gita JSON...")

with open("gita_verses.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} verses")

# =========================================================
# LOAD FAISS INDEX
# =========================================================
print("\n📦 Loading FAISS Index...")

faiss_path = os.path.join(os.path.dirname(__file__), "gita_indexes-new.faiss")

index = faiss.read_index(faiss_path)

print("✅ FAISS Index Loaded")
print(f"📏 Vector Dimension: {index.d}")
print(f"📚 Total Indexed Vectors: {index.ntotal}")

log_memory("After FAISS Load")

# =========================================================
# VECTOR SEARCH
# =========================================================
def find_relevant_verses(query, k=3):

    try:

        print("\n" + "="*80)
        print("🔍 VECTOR SEARCH START")
        print("="*80)

        print(f"\n📝 USER QUERY:\n{query}")

        log_memory("Before Embedding")

        embed_start = time.time()

        embedding_response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query
        )

        embedding = embedding_response["embedding"]

        embed_time = time.time() - embed_start

        print(f"\n⏱ Embedding Time: {embed_time:.2f} sec")

        query_embedding = np.array([embedding], dtype="float32")

        print(f"✅ Embedding Shape: {query_embedding.shape}")

        log_memory("Before FAISS Search")

        faiss_start = time.time()

        distances, indices = index.search(query_embedding, k)

        faiss_time = time.time() - faiss_start

        print(f"\n⏱ FAISS Search Time: {faiss_time:.4f} sec")

        print("\n📊 SEARCH RESULTS")

        for i, idx in enumerate(indices[0]):

            print(f"""
---------------- RESULT {i+1} ----------------

Index      : {idx}
Distance   : {distances[0][i]}

Verse Preview:
{data[idx]["english"][:250]}

----------------------------------------------
""")

        gc.collect()

        log_memory("After FAISS Search")

        return [data[i] for i in indices[0]]

    except Exception as e:

        print("\n❌ VECTOR SEARCH ERROR")
        traceback.print_exc()

        raise e

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

    try:

        print("\n" + "="*100)
        print("🚀 NEW REQUEST RECEIVED")
        print("="*100)

        total_start = time.time()

        req = request.json

        print("\n📨 RAW REQUEST JSON")
        print(json.dumps(req, indent=2))

        # -------------------------------------------------
        # INPUTS
        # -------------------------------------------------

        name = req.get("name", "Friend")
        age = req.get("age", "unknown")
        query = req.get("query", "")
        language = req.get("language", "English")
        mode = req.get("mode", "normal")

        print(f"""
================ USER DETAILS ================

NAME      : {name}
AGE       : {age}
LANGUAGE  : {language}
MODE      : {mode}

QUESTION:
{query}

==============================================
""")

        # -------------------------------------------------
        # RETRIEVE VERSES
        # -------------------------------------------------

        print("\n🔍 Finding Relevant Verses...")

        retrieval_start = time.time()

        results = find_relevant_verses(query)

        retrieval_time = time.time() - retrieval_start

        print(f"\n✅ Retrieval Completed in {retrieval_time:.2f} sec")

        print(f"\n📚 Total Retrieved Verses: {len(results)}")

        # -------------------------------------------------
        # BUILD PROMPT
        # -------------------------------------------------

        prompt = f"""
You are Lord Krishna, giving guidance to {name}, who is {age} years old.

{name}'s Question:
{query}

Your task:
1. Provide a compassionate, mentor-like answer appropriate for a {age}-year-old.
2. Use all the three verses given and whenever quoting verses:
   - Wrap Sanskrit verse in <b> tags.
   - Wrap translation in <b> tags.
   - Provide explanation after each verse.
3. Format answer in HTML.
4. Maintain spacing between slokas.
5. Use simple language.

Knowledge base:
{results}

IMPORTANT:
- Respond fully in {language}
- Output HTML only
- Include Sanskrit + translation
- No markdown asterisks
- End with divine Krishna closing statement
"""

        print("\n" + "="*80)
        print("🧠 PROMPT GENERATED")
        print("="*80)

        print(f"\n📏 PROMPT LENGTH: {len(prompt)} characters")

        print("\n📄 PROMPT PREVIEW:\n")
        print(prompt[:3000])

        # -------------------------------------------------
        # MODEL SELECTION
        # -------------------------------------------------

        llm_model = (
            "gemini-2.5-pro"
            if mode == "deep"
            else "gemini-2.5-flash"
        )

        print(f"\n⚡ USING MODEL: {llm_model}")

        llm = get_model(llm_model)

        # -------------------------------------------------
        # GEMINI API CALL
        # -------------------------------------------------

        print("\n📡 SENDING REQUEST TO GEMINI...")

        generation_start = time.time()

        response = llm.generate_content(
            prompt,
            generation_config={
                "temperature": 0.8,
                "max_output_tokens": 1200
            }
        )

        generation_time = time.time() - generation_start

        print(f"\n⏱ Gemini Generation Time: {generation_time:.2f} sec")

        print("\n✅ Gemini Response Received")

        # -------------------------------------------------
        # RAW GEMINI RESPONSE
        # -------------------------------------------------

        print("\n" + "="*100)
        print("📦 RAW GEMINI RESPONSE OBJECT")
        print("="*100)

        print(response)

        # -------------------------------------------------
        # TOKEN USAGE
        # -------------------------------------------------

        try:

            print("\n📊 TOKEN USAGE")

            print(response.usage_metadata)

        except Exception as token_error:

            print("\n⚠️ Could not fetch token usage")
            print(token_error)

        # -------------------------------------------------
        # CANDIDATES
        # -------------------------------------------------

        try:

            print("\n🧠 RESPONSE CANDIDATES")

            for i, candidate in enumerate(response.candidates):

                print(f"\n----------- Candidate {i+1} -----------")
                print(candidate)

        except Exception as candidate_error:

            print("\n⚠️ Could not fetch candidates")
            print(candidate_error)

        # -------------------------------------------------
        # SAFETY RATINGS
        # -------------------------------------------------

        try:

            print("\n🛡 SAFETY RATINGS")

            for candidate in response.candidates:

                print(candidate.safety_ratings)

        except Exception as safety_error:

            print("\n⚠️ Could not fetch safety ratings")
            print(safety_error)

        # -------------------------------------------------
        # FINISH REASON
        # -------------------------------------------------

        try:

            print("\n🏁 FINISH REASON")

            for candidate in response.candidates:

                print(candidate.finish_reason)

        except Exception as finish_error:

            print("\n⚠️ Could not fetch finish reason")
            print(finish_error)

        # -------------------------------------------------
        # RESPONSE TEXT
        # -------------------------------------------------

        krishna_response = response.text

        print("\n" + "="*80)
        print("📄 RESPONSE PREVIEW")
        print("="*80)

        print(krishna_response[:4000])

        # -------------------------------------------------
        # CLEAN HTML TAGS
        # -------------------------------------------------

        if krishna_response.startswith("<html>"):
            krishna_response = krishna_response[6:]

        if krishna_response.endswith("</html>"):
            krishna_response = krishna_response[:-7]

        gc.collect()

        log_memory("After Gemini Generation")

        total_time = time.time() - total_start

        print(f"""

✅ REQUEST SUCCESSFUL

⏱ TOTAL REQUEST TIME : {total_time:.2f} sec
📦 MODEL USED         : {llm_model}
📚 VERSES USED        : {len(results)}

""")

        return jsonify({
            "success": True,
            "response": krishna_response,
            "verses": results,
            "model_used": llm_model
        })

    # =====================================================
    # FULL GEMINI ERROR DEBUGGING
    # =====================================================

    except Exception as e:

        print("\n" + "="*100)
        print("❌ GEMINI ERROR DETECTED")
        print("="*100)

        print(f"\n❌ ERROR TYPE:\n{type(e).__name__}")

        print(f"\n❌ ERROR MESSAGE:\n{str(e)}")

        print("\n❌ FULL TRACEBACK:\n")

        traceback.print_exc()

        print("\n" + "="*100)

        return jsonify({
            "success": False,
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

# =========================================================
# RUN APP
# =========================================================
if __name__ == "__main__":

    print("\n🌸 Krishna AI Server Starting...\n")

    app.run(
        debug=True,
        host="0.0.0.0",
        port=7970
    )
