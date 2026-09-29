# 📂 Exhaustive Code Breakdown & Architecture Guide

This document is a comprehensive, deep-dive explanation of the **GitHub Codebase RAG** application. It answers *how* it works overall, *why* certain technologies were chosen, and provides an **exhaustive, line-by-line breakdown** of the entire codebase so you can understand the exact inner workings of every function, API call, and UI component.

---

## 🏛️ Overall Architecture & How it Works in Detail

At its core, this application is a **Retrieval-Augmented Generation (RAG)** pipeline wrapped in a desktop graphical user interface (GUI). 

Instead of expecting the AI model to magically know your private GitHub codebase (which it wasn't trained on), this application acts as a bridge. Here is the exact data flow:

1. **Ingestion (Fetching & Chunking):** 
   When you paste a GitHub URL, the app connects to the public GitHub API. It uses recursive functions to walk through the folders, skipping compiled files, images, and `.git` histories. It downloads the raw text of every code file. Because an entire repository is hundreds of thousands of words, it uses a **Chunker** to slice the code into small, 300-token blocks.
2. **Embedding (Math Magic):**
   AI models don't read text; they read numbers. The **Embedder** takes each 300-token block of code and passes it through an Embedding Model (either a local `MiniLM` model or the `Jina API`). This model converts the code's semantic meaning into an array of floats (e.g., `[0.012, -0.453...]`).
3. **Vector Database (ChromaDB):**
   These number arrays (embeddings) are saved permanently to your hard drive inside the `chroma_db/` folder. This acts as our ultra-fast search engine.
4. **Retrieval & Chatting:**
   When you type a question (e.g., *"How does the auth work?"*), the app takes your question, embeds it into numbers, and asks ChromaDB: *"Find the 15 code chunks whose numbers mathematically match this question's numbers."* 
5. **Generation (The Local LLM):**
   The app takes those top 15 code chunks, glues them together into a massive string of text, and secretly injects them into the Qwen LLM prompt, effectively saying: *"Here is the relevant code. Now, answer the user's question based ONLY on this code."*

---

## 💻 Line-by-Line Code Breakdown (`app.py`)

Below is the complete source code for `app.py`, broken down into functional blocks with **exhaustive, line-by-line comments** explaining what the code is doing and why.

### Block 1: Imports and Bootstrapping

```python
# ── Bootstrap: path setup & grpc mock (grpc DLL is blocked on this machine) ───
import os, sys, re                     # standard libraries: os for paths, sys for system configs, re for Regex

# We need to ensure Python runs in the exact directory where the script is located.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) # Get the absolute path to the folder containing app.py
os.chdir(_PROJECT_DIR)                                    # Change Python's working directory to that folder

# Set up environment variables so HuggingFace models download locally, not to the C: drive root.
os.environ.setdefault("HF_HOME",            os.path.join(_PROJECT_DIR, "models"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_PROJECT_DIR, "models", "hub"))
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python") # Force pure-python protobuf
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False") # Disable ChromaDB telemetry

# Windows Defender Application Control blocks unsigned DLLs (like grpc). ChromaDB uses grpc.
# We must create a fake module (MagicMock) to trick Python into thinking grpc loaded successfully.
from unittest.mock import MagicMock as _MM
for _mod in [
    "grpc", "grpc._channel", "grpc._compression", "grpc._cython",
    "grpc._cython.cygrpc", "grpc._interceptor", "grpc._utilities",
    "grpc.experimental", "grpc.aio",
    "opentelemetry.exporter.otlp.proto.grpc",
    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
    "opentelemetry.exporter.otlp.proto.grpc._log_exporter",
    "opentelemetry.exporter.otlp.proto.grpc.exporter",
]:
    sys.modules.setdefault(_mod, _MM()) # Inject the fake object into sys.modules

# ── Standard imports ──────────────────────────────────────────────────────────
import threading, tkinter as tk        # threading for background tasks (so the UI doesn't freeze), tk for standard UI elements
import customtkinter as ctk            # modern UI wrapper over tkinter
from typing import List, Dict          # Type hinting to keep code predictable (List of Dictionaries)
```

### Block 2: Silent AES Encryption

This is a custom security block that encrypts the `.env` file automatically so your API tokens aren't visible in plain text.

```python
def _handle_env_encryption():
    import os, base64                                  # base64 is needed to format the key for Fernet
    from cryptography.fernet import Fernet             # Fernet is a standard AES encryption wrapper
    from cryptography.hazmat.primitives import hashes  # Used for hashing the password
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC # Key Derivation Function
    
    env_file = ".env"      # The raw text file
    enc_file = ".env.enc"  # The encrypted file
    
    if not os.path.exists(env_file) and not os.path.exists(enc_file):
        return # If neither exists, skip encryption

    # Hardcoded master key so the encryption happens silently without user input
    secret = b"rag_app_silent_encryption_key_2026"
    # PBKDF2 hashes the secret 100,000 times with a salt to generate a cryptographically secure 256-bit AES key
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=b'rag_salt_123', iterations=100000)
    key = base64.urlsafe_b64encode(kdf.derive(secret))
    f = Fernet(key) # Initialize the AES encryptor with our derived key
    
    # If the plain text .env file exists, we need to encrypt it
    if os.path.exists(env_file):
        with open(env_file, "rb") as f_in:      # Open .env in binary read mode
            data = f_in.read()                  # Read raw bytes
        with open(enc_file, "wb") as f_out:     # Open .env.enc in binary write mode
            f_out.write(f.encrypt(data))        # Write the encrypted payload to disk
            
        os.remove(env_file)                     # 🚨 Delete the raw .env file for security
        
        # We still need the keys right now, so we decode them into memory
        for line in data.decode().splitlines(): # Split the decoded data line by line
            if "=" in line and not line.strip().startswith("#"): # Ignore comments
                k, v = line.split("=", 1)       # Split KEY=VALUE
                os.environ[k.strip()] = v.strip() # Save to os.environ so the app can use it
                
    # If only the encrypted file exists (normal runtime behavior)
    elif os.path.exists(enc_file):
        with open(enc_file, "rb") as f_in:      # Read encrypted file
            data = f.decrypt(f_in.read())       # Decrypt it into memory using our key
        for line in data.decode().splitlines(): # Parse into os.environ
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

_handle_env_encryption() # Call the function immediately as the app starts
ctk.set_appearance_mode("dark")    # Force the CustomTkinter UI to Dark Mode
ctk.set_default_color_theme("blue") # Set the accent color to Blue
```

### Block 3: Constants & Global State

```python
# We only want to analyze code files. We skip compiled binaries (.exe, .dll) and images.
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rs", ".cpp", ".c",
    ".cs", ".rb", ".php", ".swift", ".kt",
    ".md", ".txt", ".yaml", ".yml", ".env.example"
}

# We explicitly ignore junk directories so we don't waste API tokens and database space
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "dist",
    "build", ".next", "vendor", "venv", ".venv",
    "coverage", ".pytest_cache", "eggs", "target",
    "site", "docs", "assets", "public", "static"
}

# Read variables from os.environ (which were just populated by our encryption function)
CHROMA_PATH    = os.getenv("CHROMA_PATH",   "./chroma_db")           # Where to save the database
MODEL_PATH     = os.getenv("MODEL_PATH",    "./Qwen3.5-4B.q4_k_m.gguf") # Path to local LLM
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN",  "")                      # GitHub Auth token
JINA_API_KEY   = os.getenv("JINA_API_KEY",  "")                      # Jina embedding key

MAX_CHUNK_TOKENS = 300 # Limit each code slice to 300 tokens (approx 1200 characters)
OVERLAP_LINES    = 5   # Keep 5 lines of overlap between chunks so functions aren't split in half

_embed_model = None # Lazy-loaded singleton for the MiniLM model (saves RAM on startup)
_llm         = None # Lazy-loaded singleton for the Qwen model
```

### Block 4: Fetcher (GitHub Integration)

```python
def fetch_repo_files(url: str) -> List[Dict]:
    from github import Github, Auth # PyGithub library
    # Example: "https://github.com/owner/repo" -> splits by "github.com/" -> gets "owner/repo" -> splits by "/"
    parts = url.rstrip("/").split("github.com/")[-1].split("/")
    owner, repo = parts[0], parts[1] # Extract 'owner' and 'repo' names
    
    # If we have a token, use it (GitHub rate limits unauthenticated requests to 60/hr)
    token = GITHUB_TOKEN or None
    g = Github(auth=Auth.Token(token)) if token else Github()
    repository = g.get_repo(f"{owner}/{repo}") # Connect to the repo
    
    files = [] # Empty list to store our code
    def _walk(path=""): # Recursive function to scan directories
        try:
            contents = repository.get_contents(path) # Fetch files in current directory
        except Exception:
            return # Skip if access denied or folder doesn't exist
            
        for item in contents:
            if item.type == "dir": # If item is a folder...
                if item.name not in SKIP_DIRS: # And it's not a junk folder...
                    _walk(item.path) # RECURSION: call this function again to go deeper
            else: # If item is a file...
                ext = os.path.splitext(item.name)[1].lower() # Get extension (e.g. '.py')
                if ext in ALLOWED_EXTENSIONS:
                    try:
                        # Download the actual file bytes and decode to a string
                        text = item.decoded_content.decode("utf-8", errors="ignore")
                        lang = ext.lstrip(".") # Remove the dot (e.g., 'py')
                        # Save it as a dictionary
                        files.append({"path": item.path, "text": text, "language": lang})
                    except Exception:
                        pass # Ignore encoding errors
    _walk() # Start the walk at the root folder
    return files
```

### Block 5: Chunker

```python
def _count_tokens(text: str) -> int:
    return len(text) // 4 # Rough approximation: 1 token is about 4 characters in English code

def chunk_files(files: List[Dict]) -> List[Dict]:
    all_chunks = []
    for file in files:
        lines = file["text"].splitlines() # Split the massive file into a list of lines
        chunks, start, buf = [], 0, []
        for i, line in enumerate(lines):
            buf.append(line) # Add current line to the buffer
            # Check if the buffer is too big
            if _count_tokens("\n".join(buf)) >= MAX_CHUNK_TOKENS:
                chunks.append({
                    "text": "\n".join(buf), # Join buffer back into a string
                    "path": file["path"],
                    "language": file["language"],
                    "start_line": start,
                    "end_line": i,
                    "chunk_id": f"{file['path']}::{start}-{i}", # Unique ID for ChromaDB
                })
                start = max(0, i - OVERLAP_LINES) # Backtrack 5 lines for context overlap
                buf   = lines[start:i + 1] # Start the new buffer with the overlap lines
        
        if buf: # If there are leftover lines at the end of the file, save them too
            chunks.append({
                "text": "\n".join(buf),
                "path": file["path"],
                "language": file["language"],
                "start_line": start,
                "end_line": len(lines),
                "chunk_id": f"{file['path']}::{start}-{len(lines)}",
            })
        all_chunks.extend(chunks) # Add file chunks to master list
        print(f"[Chunker] {file['path']} -> {len(chunks)} chunks") # Log to terminal
    return all_chunks
```

### Block 6: Embedder (Translating Code to Math)

```python
def _embed_via_jina(texts: List[str]) -> List[List[float]]:
    import requests as req, time
    # Jina is an API highly optimized for coding languages. It supports up to 8k context embeddings.
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
    result = []
    batch     = 8   # Send 8 chunks at a time to not overload the API
    max_chars = 800 # Hard limit string sizes to prevent 413 Payload Too Large errors
    
    for i in range(0, len(texts), batch):
        chunk = [t[:max_chars] for t in texts[i:i+batch]] # Slice out the batch
        resp = req.post( # Make HTTP POST request to Jina
            "https://api.jina.ai/v1/embeddings",
            headers=headers,
            json={"model": "jina-embeddings-v2-base-code", "input": chunk},
            timeout=60,
        )
        if not resp.ok: # If API fails, crash the thread so the user knows
            raise RuntimeError(f"Jina API error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()["data"]
        data.sort(key=lambda x: x["index"]) # Ensure results match the input order
        result.extend([d["embedding"] for d in data]) # Extract the float arrays
        if i + batch < len(texts):
            time.sleep(1) # Sleep 1 second to avoid HTTP 429 Too Many Requests
    return result

def _get_local_embed_model():
    global _embed_model
    if _embed_model is None: # Only load the model if it hasn't been loaded yet (Singleton)
        from sentence_transformers import SentenceTransformer
        # MiniLM-L6 is a tiny 22MB embedding model that runs fast on CPU
        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _embed_model

def embed_texts(texts: List[str]) -> List[List[float]]:
    if JINA_API_KEY: # Prefer Jina API if key exists, otherwise run local model
        return _embed_via_jina(texts)
    return _get_local_embed_model().encode(
        texts, batch_size=4, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

def embed_query(query: str) -> List[float]:
    return embed_texts([query])[0] # Wrapper to embed a single string
```

### Block 7: VectorStore (ChromaDB)

```python
def _chroma_client():
    import chromadb
    return chromadb.PersistentClient(path=CHROMA_PATH) # Connects to the local sqlite-based vector db

def get_or_create_collection(repo_name: str):
    # Chroma requires collection names to have no slashes and be lowercase
    safe = repo_name.replace("/", "_").replace("-", "_").lower()
    # hnsw:space cosine tells the DB to use cosine similarity (angles of vectors) for searching
    return _chroma_client().get_or_create_collection(name=safe, metadata={"hnsw:space": "cosine"})

def store_chunks(chunks: List[Dict], repo_name: str):
    col     = get_or_create_collection(repo_name)
    texts   = [c["text"]     for c in chunks] # Extract all raw text
    ids     = [c["chunk_id"] for c in chunks] # Extract all unique IDs
    # Extract metadata so the LLM knows which file it's looking at
    metas   = [{"path": c["path"], "language": c["language"],
                "start_line": c["start_line"], "end_line": c["end_line"]} for c in chunks]
    vectors = embed_texts(texts) # Run the embedder function!
    
    batch   = 100
    for i in range(0, len(chunks), batch): # Insert into Chroma in batches of 100
        col.add(
            documents=texts[i:i+batch],
            embeddings=vectors[i:i+batch],
            ids=ids[i:i+batch],
            metadatas=metas[i:i+batch],
        )

def search_chunks(query: str, repo_name: str, n: int = 15) -> List[Dict]:
    col = get_or_create_collection(repo_name)
    vec = embed_query(query) # Convert the user's question into math
    # The database calculates the mathematical distance between the question vector and all code vectors
    res = col.query(query_embeddings=[vec], n_results=n) # Retrieve the top 15 closest matches
    results = []
    # Zip together the text and the metadata from the results
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        results.append({"text": doc, "path": meta.get("path", ""), "language": meta.get("language", "")})
    return results

def _clean_orphan_folders():
    import shutil
    try:
        c = _chroma_client()
        active_ids = {str(col.id) for col in c.list_collections()}
        if not os.path.exists(CHROMA_PATH): return
        for item in os.listdir(CHROMA_PATH):
            item_path = os.path.join(CHROMA_PATH, item)
            if os.path.isdir(item_path) and item not in active_ids:
                shutil.rmtree(item_path, ignore_errors=True) # Automatically sweeps and deletes any orphan UUID directories
    except Exception: pass

def list_collections() -> List[str]:
    try:
        _clean_orphan_folders() # Automatic background cleanup sweep
        return [c.name for c in _chroma_client().list_collections()]
    except Exception: return []

def delete_collection(repo_name: str):
    import os, shutil
    safe = repo_name.replace("/", "_").replace("-", "_").lower()
    try:
        c = _chroma_client()
        col_id = None
        try: col_id = str(c.get_collection(safe).id)
        except Exception: pass
        c.delete_collection(safe)
        if col_id:
            col_dir = os.path.join(CHROMA_PATH, col_id)
            if os.path.exists(col_dir):
                shutil.rmtree(col_dir, ignore_errors=True)
        manifest_file = _manifest_path(repo_name)
        if os.path.exists(manifest_file):
            try: os.remove(manifest_file)
            except Exception: pass
        _clean_orphan_folders() # Guarantees zero leftover ghost folders
        print(f"[VectorStore] Force-deleted all files for '{safe}'")
    except Exception as e:
        _clean_orphan_folders()
        raise RuntimeError(f"Could not delete '{safe}': {e}")
```

### Block 8: File Manifest Generation

Because RAG search is bad at answering *"list all files"*, we create a hardcoded JSON index of the files.

```python
def _manifest_path(repo_name: str) -> str:
    safe = repo_name.replace("/", "_").replace("-", "_").lower()
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return os.path.join(CHROMA_PATH, f"{safe}_manifest.json")

def save_manifest(files: List[Dict], repo_name: str):
    import json
    with open(_manifest_path(repo_name), "w", encoding="utf-8") as f:
        # Save a lightweight list of just paths and languages
        json.dump([{"path": fi["path"], "language": fi["language"]} for fi in files], f)

def get_manifest(repo_name: str) -> List[Dict]:
    import json
    try:
        with open(_manifest_path(repo_name), encoding="utf-8") as f:
            return json.load(f)
    except Exception: return []

def format_file_tree(manifest: List[Dict]) -> str:
    from collections import defaultdict
    tree = defaultdict(list)
    for item in manifest:
        parts  = item["path"].split("/")
        folder = parts[0] if len(parts) > 1 else "(root)"
        name   = "/".join(parts[1:]) if len(parts) > 1 else parts[0]
        tree[folder].append((name, item["language"]))

    lines = [f"📁 **{len(manifest)} files indexed:**\n"]
    for folder in sorted(tree):
        lines.append(f"\n**{folder}/**")
        for name, lang in sorted(tree[folder]):
            lines.append(f"  • `{name}` _{lang}_")
    return "\n".join(lines) # Returns a beautiful Markdown-formatted directory tree
```

### Block 9: Intent Detection & LLM Prompting

```python
def detect_intent(question: str) -> str:
    q = question.lower().strip()
    # Detects conversational greetings ("hello", "hi", "hey")
    if re.search(r'^(hi|hello|hey|greetings|good\s*(morning|afternoon|evening)|yo|sup|hiya|howdy)\b', q):
        return "greeting"
    # Detects if user is asking about the bot itself (e.g. "what can you do")
    if re.search(r'\b(who are (you|u)|what are (you|u)|what.{0,15}can (you|u) do|help( me)?)\b', q):
        return "self_help"
    # Detects if user wants to see the files
    if re.search(r'\b(list|show|all|what).{0,15}(files?|structure|tree|paths?|directory)\b', q):
        return "list_files"
    # Detects a broad question like "what is this repo"
    if re.search(r'\b(what is|what does|about|overview|purpose|explain|summarize|describe).{0,20}(this|repo|project|codebase)\b', q):
        return "overview"
    # Detects deep dive questions
    if re.search(r'\b(how (does|do|to|can)|explain how|walk.?me.?through)\b', q):
        return "explain"
    return "rag" # Default behavior

def search_readme_first(repo_name: str, n: int = 8) -> List[Dict]:
    # Custom search function designed to target READMEs and architecture files
    col = get_or_create_collection(repo_name)
    overview_query = "project overview introduction purpose what this does getting started installation features tech stack"
    vec = embed_query(overview_query) # Embed this fake "perfect" overview query
    res = col.query(query_embeddings=[vec], n_results=n)
    results = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        results.append({"text": doc, "path": meta.get("path", ""), "language": meta.get("language", "")})
    # Sort the results so files named 'readme' are forcefully pushed to the top of the context
    results.sort(key=lambda r: (0 if "readme" in r["path"].lower() else 1 if "/" not in r["path"] else 2))
    return results

# The main instructions given to the LLM to govern its behavior
_SYSTEM_PROMPT = """You are an expert software engineer helping users understand a GitHub repository.
You receive retrieved code snippets and file contents as context.

Rules:
- Synthesize a comprehensive answer using the provided context.
- If the exact answer isn't fully contained in the snippets, use your expert programming knowledge to deduce and explain how the retrieved components likely fit together.
- Be direct — lead with the answer, then elaborate in detail.
- For explanations: walk through the logic step-by-step, referencing specific classes, functions, and file paths.
- Use markdown formatting: ```lang blocks, bullet points, and **bold** for key terms.
- Be helpful and detailed. Do not simply give up if the context is fragmented; explain everything you can based on what IS provided."""

def _build_prompt(question: str, context_chunks: List[Dict], intent: str = "rag") -> str:
    context_parts = []
    current_len = 0
    for c in context_chunks: # Assemble all retrieved chunks into a mega-string
        text = c["text"].strip()
        if len(text) > 2500:
            text = text[:2500] + "\n...[TRUNCATED]..."
        part = f"**File:** `{c['path']}`\n```{c['language']}\n{text}\n```"
        
        # FAILSAFE: Hard cap the context at ~14,000 characters to prevent token limit crashes
        if current_len + len(part) > 14000:
            break
            
        context_parts.append(part)
        current_len += len(part)
        
    context = "\n\n---\n\n".join(context_parts)

    # Override the user's task instruction based on their intent to force better LLM outputs
    if intent == "overview":
        task = (
            "Give a clear, structured overview of this repository:\n"
            "1. **What it does** — one sentence purpose.\n"
            "2. **Tech stack** — languages, frameworks, key libraries.\n"
            "3. **Architecture** — main components and how they connect.\n"
            "4. **Entry points** — where execution starts or key files to read first.\n"
            "Base your answer entirely on the code context above."
        )
    elif intent == "explain":
        task = (
            f"Explain in detail: {question}\n"
            "Walk through the logic step by step. "
            "Reference specific files and functions from the context above."
        )
    elif intent == "self_help":
        task = (
            "The user is asking about YOU and your capabilities. "
            "Explain that you are an AI Codebase Assistant (RAG). "
            "You can search through their indexed GitHub repositories, explain logic, summarize architecture, and list files. "
            "Respond naturally and conversationally as an AI assistant. Do not mention code snippets since there are none provided."
        )
    elif intent == "greeting":
        task = (
            f"The user greeted you with: '{question}'. "
            "Reply with a warm, friendly, concise greeting. "
            "Ask how you can assist them with the codebase today."
        )
    else:
        task = question # Pass raw question

    context_block = f"### Repository Code Context\n\n{context}\n\n" if context else ""

    # Format the prompt using ChatML tags (<|im_start|>) which is what Qwen requires
    return (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{context_block}"
        f"### Task\n{task}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def _get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama # Llama.cpp runs C/C++ optimized machine learning on the CPU
        print(f"[LLM] Loading {MODEL_PATH}…")
        _llm = Llama(
            model_path   = MODEL_PATH, # The GGUF file
            n_ctx        = 8192,       # Context window (memory size). 8192 is enough to read ~15 code chunks
            n_gpu_layers = int(os.getenv("GPU_LAYERS", "0")), # Offloads math to GPU if requested
            n_threads    = os.cpu_count() or 4, # Max out CPU cores for speed
            verbose      = False,
        )
    return _llm

def ask(question: str, context_chunks: List[Dict], repo_name: str = None) -> str:
    intent = detect_intent(question)

    if intent == "self_help":
        context_chunks = [] # Purge context to save tokens and prevent the AI from hallucinating codebase info

    if intent == "list_files":
        if repo_name:
            manifest = get_manifest(repo_name)
            if manifest: return format_file_tree(manifest) # Instant generation
        paths = sorted({c["path"] for c in context_chunks})
        return "**Files found in context**:\n" + "\n".join(f"  • `{p}`" for p in paths)

    if intent == "overview" and repo_name:
        context_chunks = search_readme_first(repo_name) # Swap standard search for Readme Search

    prompt = _build_prompt(question, context_chunks, intent)
    temp = 0.3 if intent in ("overview", "explain") else 0.15 # Higher temp = more creative/expansive answers

    if os.path.isfile(MODEL_PATH): # Ensure the model actually exists on disk
        llm = _get_llm()
        out = llm(
            prompt, max_tokens=1024, temperature=temp, top_p=0.9, top_k=40,
            repeat_penalty=1.1, stop=["<|im_end|>", "<|im_start|>"], echo=False,
        )
        text = out["choices"][0]["text"].strip()
        # Some Qwen models output reasoning in <think> tags. We strip them out for clean UI.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

    return "❌ No local LLM available. Please ensure the GGUF model file is placed in the folder."
```

### Block 10: The CustomTkinter User Interface (GUI)

```python
class RAGApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("GitHub Codebase RAG") # Window title
        self.geometry("1100x700")         # Window resolution
        self.resizable(True, True)
        self._current_repo = None         # Tracks which database is active
        self._build_ui()                  # Calls the UI creation method
        self.after(200, self._init_app)   # Defers startup loading by 200ms so the UI renders instantly first

    def _build_ui(self):
        # Configure a 2-column layout (Sidebar + Main chat area)
        self.grid_columnconfigure(1, weight=1) # Main column takes up all expandable space
        self.grid_rowconfigure(0, weight=1)

        # ---- SIDEBAR ----
        sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew") # sticky="nsew" makes it stretch North/South/East/West
        sidebar.grid_rowconfigure(4, weight=1)

        # Title labels
        ctk.CTkLabel(sidebar, text="GitHub RAG", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, padx=16, pady=(20, 4))
        ctk.CTkLabel(sidebar, text="Local · Free · Private", font=ctk.CTkFont(size=11), text_color="gray").grid(row=1, column=0, padx=16, pady=(0, 16))

        # Input box for GitHub URLs
        ctk.CTkLabel(sidebar, text="INDEX A REPO", font=ctk.CTkFont(size=10, weight="bold"), text_color="gray").grid(row=2, column=0, padx=16, sticky="w")
        self.url_entry = ctk.CTkEntry(sidebar, placeholder_text="https://github.com/owner/repo", width=188)
        self.url_entry.grid(row=3, column=0, padx=16, pady=4)
        
        # Ingestion trigger button
        self.ingest_btn = ctk.CTkButton(sidebar, text="🚀 Ingest Repo", command=self._ingest, width=188)
        self.ingest_btn.grid(row=4, column=0, padx=16, pady=(0, 16), sticky="n")

        # Repository Listbox (Standard Tkinter Listbox because CustomTkinter lacks a good native listbox)
        ctk.CTkLabel(sidebar, text="INDEXED REPOS", font=ctk.CTkFont(size=10, weight="bold"), text_color="gray").grid(row=5, column=0, padx=16, pady=(8, 4), sticky="w")
        self.repo_list = tk.Listbox(sidebar, bg="#1a1a2e", fg="#e0e0e0", selectbackground="#3b82f6",
                                     relief="flat", borderwidth=0, font=("Segoe UI", 11), activestyle="none")
        self.repo_list.grid(row=6, column=0, padx=10, pady=4, sticky="nsew")
        sidebar.grid_rowconfigure(6, weight=1) # Let the listbox expand vertically
        self.repo_list.bind("<<ListboxSelect>>", self._on_repo_select) # Trigger event when clicked

        # Sidebar utility buttons (Refresh, Clear, Delete)
        btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_frame.grid(row=7, column=0, padx=10, pady=8)
        ctk.CTkButton(btn_frame, text="↻ Refresh",    width=82, height=28, command=self._refresh_repos).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="✖ Clear Chat", width=82, height=28, command=self._clear_chat).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="❌ Delete",     width=82, height=28, fg_color="#7f1d1d", hover_color="#991b1b", command=self._delete_repo).pack(side="left", padx=2)

        # ---- MAIN AREA ----
        main = ctk.CTkFrame(self, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew", padx=0)
        main.grid_rowconfigure(1, weight=1) # Chat box expands vertically
        main.grid_columnconfigure(0, weight=1) # Chat box expands horizontally

        # Top Status Bar
        self.status_bar = ctk.CTkLabel(main, text="⏳ Initialising…", font=ctk.CTkFont(size=12), fg_color="#1e1e2e", corner_radius=6, anchor="w")
        self.status_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 0))

        # Large Read-Only Textbox for Chat Output
        self.chat_box = ctk.CTkTextbox(main, wrap="word", font=ctk.CTkFont(size=13), state="disabled")
        self.chat_box.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=12, pady=8)

        # Bottom Input Area
        self.query_entry = ctk.CTkEntry(main, placeholder_text="Ask about architecture, functions, logic…", font=ctk.CTkFont(size=13))
        self.query_entry.grid(row=2, column=0, sticky="ew", padx=(12, 4), pady=(0, 12))
        self.query_entry.bind("<Return>", lambda e: self._send()) # Allows pressing 'Enter' to send
        ctk.CTkButton(main, text="Send →", width=90, command=self._send).grid(row=2, column=1, padx=(0, 12), pady=(0, 12))

    # --- UI LOGIC METHODS ---

    def _init_app(self):
        # We start initialization on a background thread so the GUI doesn't freeze!
        def _run():
            try:
                list_collections() # Forces ChromaDB to wake up and connect to sqlite
                self._refresh_repos() # Populate the sidebar
                embed_backend = "Jina API" if JINA_API_KEY else "Local MiniLM"
                self._set_status(f"✅ Ready  |  LLM: Local GGUF  |  Embeddings: {embed_backend}", "green")
            except Exception as e:
                self._set_status(f"❌ Init error: {e}", "red")
        threading.Thread(target=_run, daemon=True).start()

    def _set_status(self, msg: str, color="white"):
        # UI updates MUST happen on the main thread. self.after(0) forces Tkinter to execute this safely.
        colors = {"green": "#22c55e", "red": "#ef4444", "orange": "#f97316", "white": "#e0e0e0"}
        self.after(0, lambda: self.status_bar.configure(text=f"  {msg}", text_color=colors.get(color, color)))

    def _append_chat(self, text: str):
        # Temporarily unlocks the chatbox, injects text, scrolls to bottom, then locks it again.
        def _do():
            self.chat_box.configure(state="normal")
            self.chat_box.insert("end", text)
            self.chat_box.see("end")
            self.chat_box.configure(state="disabled")
        self.after(0, _do)

    def _clear_chat(self):
        self.chat_box.configure(state="normal")
        self.chat_box.delete("1.0", "end")
        self.chat_box.configure(state="disabled")

    def _refresh_repos(self):
        repos = list_collections()
        self.after(0, lambda: (
            self.repo_list.delete(0, "end"),
            [self.repo_list.insert("end", f"  {r}") for r in repos]
        ))

    def _on_repo_select(self, _event=None):
        sel = self.repo_list.curselection() # Check what item was clicked
        if sel:
            self._current_repo = self.repo_list.get(sel[0]).strip()
            self._set_status(f"📂 Selected: {self._current_repo}", "white")
            self._append_chat(f"\n📂 Switched to: {self._current_repo}\n")

    def _delete_repo(self):
        if not self._current_repo:
            self._set_status("⚠ Select a repo to delete first", "orange")
            return
        try:
            delete_collection(self._current_repo)
            self._set_status(f"🗑 Deleted '{self._current_repo}'", "green")
            self._append_chat(f"\n🗑 Deleted repository database for: {self._current_repo}\n")
            self._current_repo = None
            self._refresh_repos()
        except Exception as e:
            self._set_status(f"❌ Delete error: {e}", "red")

    def _ingest(self):
        url = self.url_entry.get().strip()
        if not url:
            self._set_status("Paste a GitHub URL first", "orange"); return
        self.ingest_btn.configure(state="disabled") # Disable button to prevent double-clicks
        self._set_status("📥 Fetching repo…", "orange")
        # Run ingestion in a background thread because fetching and embedding takes 10+ seconds
        threading.Thread(target=self._ingest_worker, args=(url,), daemon=True).start()

    def _ingest_worker(self, url: str):
        try:
            self._set_status("📥 Fetching files…", "orange")
            files = fetch_repo_files(url)
            self._set_status(f"✂️  Chunking {len(files)} files…", "orange")
            chunks = chunk_files(files)
            repo_name = url.rstrip("/").split("github.com/")[-1]
            save_manifest(files, repo_name)
            self._set_status(f"🔢 Embedding {len(chunks)} chunks…", "orange")
            store_chunks(chunks, repo_name)
            self._current_repo = repo_name.replace("/", "_").replace("-", "_").lower()
            self._refresh_repos()
            self._set_status(f"✅ Indexed '{repo_name}' — {len(chunks)} chunks", "green")
            self._append_chat(f"\n✅ Repo indexed: {repo_name} ({len(chunks)} chunks)\n")
        except Exception as e:
            self._set_status(f"❌ Ingest error: {e}", "red")
            self._append_chat(f"\n✗ Error during ingestion: {e}\n")
        finally: # Ensure the button is re-enabled even if an error occurs
            self.after(0, lambda: self.ingest_btn.configure(state="normal"))

    def _send(self):
        question = self.query_entry.get().strip()
        if not question: return
        if not self._current_repo:
            self._append_chat("\n⚠ Select or ingest a repository first.\n"); return
        self.query_entry.delete(0, "end") # Clear input box
        self._append_chat(f"\n🧑 {question}\n")
        self._set_status("🤖 Thinking…", "orange")
        # Run LLM generation in background thread because it takes 5-20 seconds to answer
        threading.Thread(target=self._send_worker, args=(question,), daemon=True).start()

    def _send_worker(self, question: str):
        try:
            chunks = search_chunks(question, self._current_repo) # Query vector DB
            answer = ask(question, chunks, repo_name=self._current_repo) # Pass to LLM
            self._append_chat(f"\n🤖 {answer}\n")
            self._set_status("✅ Ready", "green")
        except Exception as e:
            self._append_chat(f"\n❌ Error: {e}\n")
            self._set_status(f"❌ {e}", "red")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        app = RAGApp()   # Initialize the application class
        app.mainloop()   # Start the infinite Tkinter event listening loop
    except Exception as e: # Catch fatal crashes (like Out of Memory)
        import traceback
        print("\n" + "="*50)
        print("CRITICAL ERROR: The application crashed.")
        traceback.print_exc() # Print full stack trace to the console
        print("="*50 + "\n")
```
