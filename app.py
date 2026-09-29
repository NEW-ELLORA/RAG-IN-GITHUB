"""
GitHub Codebase RAG — Desktop GUI
Single-file app: fetcher · chunker · embedder · vectorstore · LLM · UI
Run: python app.py  (or double-click run.bat)
"""
# ── Bootstrap: path setup & grpc mock (grpc DLL is blocked on this machine) ───
import os, sys, re

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_PROJECT_DIR)                          # ensure relative paths work
os.environ.setdefault("HF_HOME",            os.path.join(_PROJECT_DIR, "models"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_PROJECT_DIR, "models", "hub"))
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# ── Auto-detect CUDA Toolkit and inject into PATH so llama.dll can find cudart ─
_cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if os.path.isdir(_cuda_base):
    for _ver in sorted(os.listdir(_cuda_base), reverse=True):  # pick newest
        _cuda_bin = os.path.join(_cuda_base, _ver, "bin")
        if os.path.isdir(_cuda_bin) and _cuda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _cuda_bin + ";" + os.environ.get("PATH", "")
            print(f"[CUDA] Injected {_cuda_bin} into PATH")
            break

# chromadb imports opentelemetry which tries to load grpc — mock it out
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
    sys.modules.setdefault(_mod, _MM())

# ── Standard imports ──────────────────────────────────────────────────────────
import threading, tkinter as tk
import customtkinter as ctk
from typing import List, Dict

def _handle_env_encryption():
    import os, base64
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    
    env_file = ".env"
    enc_file = ".env.enc"
    
    if not os.path.exists(env_file) and not os.path.exists(enc_file):
        return

    # Use a hardcoded key so encryption/decryption happens silently
    secret = b"rag_app_silent_encryption_key_2026"
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=b'rag_salt_123', iterations=100000)
    key = base64.urlsafe_b64encode(kdf.derive(secret))
    f = Fernet(key)
    
    if os.path.exists(env_file):
        with open(env_file, "rb") as f_in:
            data = f_in.read()
        with open(enc_file, "wb") as f_out:
            f_out.write(f.encrypt(data))
            
        os.remove(env_file)
        
        for line in data.decode().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
                
    elif os.path.exists(enc_file):
        with open(enc_file, "rb") as f_in:
            data = f.decrypt(f_in.read())
        for line in data.decode().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

_handle_env_encryption()
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Constants ─────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rs", ".cpp", ".c",
    ".cs", ".rb", ".php", ".swift", ".kt",
    ".md", ".txt", ".yaml", ".yml", ".env.example"
}
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "dist",
    "build", ".next", "vendor", "venv", ".venv",
    "coverage", ".pytest_cache", "eggs", "target",
    "site", "docs", "assets", "public", "static"
}

CHROMA_PATH    = os.getenv("CHROMA_PATH",   "./chroma_db")
_env_model     = os.getenv("MODEL_PATH", "")
_default_model = os.path.join(_PROJECT_DIR, "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
MODEL_PATH     = _env_model if (_env_model and os.path.isfile(_env_model)) else _default_model
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN",  "")
JINA_API_KEY   = os.getenv("JINA_API_KEY",  "")
MAX_CHUNK_TOKENS = 300
OVERLAP_LINES    = 5

# ── Lazy singletons ───────────────────────────────────────────────────────────
_embed_model = None
_llm         = None

# ══════════════════════════════════════════════════════════════════════════════
# FETCHER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_repo_files(url: str) -> List[Dict]:
    from github import Github, Auth
    parts = url.rstrip("/").split("github.com/")[-1].split("/")
    owner, repo = parts[0], parts[1]
    token = GITHUB_TOKEN or None
    g = Github(auth=Auth.Token(token)) if token else Github()
    repository = g.get_repo(f"{owner}/{repo}")

    files = []
    def _walk(path=""):
        try:
            contents = repository.get_contents(path)
        except Exception:
            return
        for item in contents:
            if item.type == "dir":
                if item.name not in SKIP_DIRS:
                    _walk(item.path)
            else:
                ext = os.path.splitext(item.name)[1].lower()
                if ext in ALLOWED_EXTENSIONS:
                    try:
                        text = item.decoded_content.decode("utf-8", errors="ignore")
                        lang = ext.lstrip(".")
                        files.append({"path": item.path, "text": text, "language": lang})
                    except Exception:
                        pass
    _walk()
    return files

# ══════════════════════════════════════════════════════════════════════════════
# CHUNKER
# ══════════════════════════════════════════════════════════════════════════════

def _count_tokens(text: str) -> int:
    return len(text) // 4

def chunk_files(files: List[Dict]) -> List[Dict]:
    all_chunks = []
    for file in files:
        lines = file["text"].splitlines()
        chunks, start, buf = [], 0, []
        for i, line in enumerate(lines):
            buf.append(line)
            if _count_tokens("\n".join(buf)) >= MAX_CHUNK_TOKENS:
                chunks.append({
                    "text": "\n".join(buf),
                    "path": file["path"],
                    "language": file["language"],
                    "start_line": start,
                    "end_line": i,
                    "chunk_id": f"{file['path']}::{start}-{i}",
                })
                start = max(0, i - OVERLAP_LINES)
                buf   = lines[start:i + 1]
        if buf:
            chunks.append({
                "text": "\n".join(buf),
                "path": file["path"],
                "language": file["language"],
                "start_line": start,
                "end_line": len(lines),
                "chunk_id": f"{file['path']}::{start}-{len(lines)}",
            })
        all_chunks.extend(chunks)
        print(f"[Chunker] {file['path']} -> {len(chunks)} chunks")
    print(f"\n[Chunker] Total chunks: {len(all_chunks)}")
    return all_chunks

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDER
# ══════════════════════════════════════════════════════════════════════════════

def _embed_via_jina(texts: List[str]) -> List[List[float]]:
    import requests as req, time
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
    result = []
    batch     = 8
    max_chars = 800
    for i in range(0, len(texts), batch):
        chunk = [t[:max_chars] for t in texts[i:i+batch]]
        resp = req.post(
            "https://api.jina.ai/v1/embeddings",
            headers=headers,
            json={"model": "jina-embeddings-v2-base-code", "input": chunk},
            timeout=60,
        )
        if not resp.ok:
            raise RuntimeError(f"Jina API error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()["data"]
        data.sort(key=lambda x: x["index"])
        result.extend([d["embedding"] for d in data])
        print(f"[Embedder] {min(i+batch, len(texts))}/{len(texts)} via Jina API")
        if i + batch < len(texts):
            time.sleep(1)
    return result

def _get_local_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print("[Embedder] Loading all-MiniLM-L6-v2 (local fallback)...")
        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        print("[Embedder] Model loaded.")
    return _embed_model

def embed_texts(texts: List[str]) -> List[List[float]]:
    if JINA_API_KEY:
        return _embed_via_jina(texts)
    return _get_local_embed_model().encode(
        texts, batch_size=4, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

def embed_query(query: str) -> List[float]:
    return embed_texts([query])[0]

# ══════════════════════════════════════════════════════════════════════════════
# VECTORSTORE
# ══════════════════════════════════════════════════════════════════════════════

def _chroma_client():
    import chromadb
    return chromadb.PersistentClient(path=CHROMA_PATH)

def get_or_create_collection(repo_name: str):
    safe = repo_name.replace("/", "_").replace("-", "_").lower()
    return _chroma_client().get_or_create_collection(name=safe, metadata={"hnsw:space": "cosine"})

def store_chunks(chunks: List[Dict], repo_name: str):
    col     = get_or_create_collection(repo_name)
    texts   = [c["text"]     for c in chunks]
    ids     = [c["chunk_id"] for c in chunks]
    metas   = [{"path": c["path"], "language": c["language"],
                "start_line": c["start_line"], "end_line": c["end_line"]} for c in chunks]
    vectors = embed_texts(texts)
    batch   = 100
    for i in range(0, len(chunks), batch):
        col.add(
            documents=texts[i:i+batch],
            embeddings=vectors[i:i+batch],
            ids=ids[i:i+batch],
            metadatas=metas[i:i+batch],
        )
    print(f"[VectorStore] Stored {len(chunks)} chunks for '{repo_name}'")

def search_chunks(query: str, repo_name: str, n: int = 15) -> List[Dict]:
    col = get_or_create_collection(repo_name)
    vec = embed_query(query)
    res = col.query(query_embeddings=[vec], n_results=n)
    results = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        results.append({"text": doc, "path": meta.get("path", ""), "language": meta.get("language", "")})
    return results

def _clean_orphan_folders():
    import shutil
    try:
        c = _chroma_client()
        active_ids = {str(col.id) for col in c.list_collections()}
        if not os.path.exists(CHROMA_PATH):
            return
        for item in os.listdir(CHROMA_PATH):
            item_path = os.path.join(CHROMA_PATH, item)
            if os.path.isdir(item_path) and item not in active_ids:
                shutil.rmtree(item_path, ignore_errors=True)
    except Exception:
        pass

def list_collections() -> List[str]:
    try:
        _clean_orphan_folders()
        return [c.name for c in _chroma_client().list_collections()]
    except Exception:
        return []

def delete_collection(repo_name: str):
    import os, shutil
    safe = repo_name.replace("/", "_").replace("-", "_").lower()
    try:
        c = _chroma_client()
        col_id = None
        try:
            col_id = str(c.get_collection(safe).id)
        except Exception:
            pass
        
        c.delete_collection(safe)
        
        if col_id:
            col_dir = os.path.join(CHROMA_PATH, col_id)
            if os.path.exists(col_dir):
                shutil.rmtree(col_dir, ignore_errors=True)
            
        manifest_file = _manifest_path(repo_name)
        if os.path.exists(manifest_file):
            try: os.remove(manifest_file)
            except Exception: pass

        _clean_orphan_folders()
        print(f"[VectorStore] Force-deleted all files for '{safe}'")
    except Exception as e:
        _clean_orphan_folders()
        raise RuntimeError(f"Could not delete '{safe}': {e}")

# ══════════════════════════════════════════════════════════════════════════════
# FILE MANIFEST  
# ══════════════════════════════════════════════════════════════════════════════

def _manifest_path(repo_name: str) -> str:
    safe = repo_name.replace("/", "_").replace("-", "_").lower()
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return os.path.join(CHROMA_PATH, f"{safe}_manifest.json")

def save_manifest(files: List[Dict], repo_name: str):
    import json
    with open(_manifest_path(repo_name), "w", encoding="utf-8") as f:
        json.dump([{"path": fi["path"], "language": fi["language"]} for fi in files], f)

def get_manifest(repo_name: str) -> List[Dict]:
    import json
    try:
        with open(_manifest_path(repo_name), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

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
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# INTENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_intent(question: str) -> str:
    q = question.lower().strip()
    if re.search(r'^(hi|hello|hey|greetings|good\s*(morning|afternoon|evening)|yo|sup|hiya|howdy)\b', q):
        return "greeting"
    if re.search(r'\b(who are (you|u)|what are (you|u)|what.{0,15}can (you|u) do|help( me)?)\b', q):
        return "self_help"
    if re.search(r'\b(list|show|all|what).{0,15}(files?|structure|tree|paths?|directory)\b', q):
        return "list_files"
    if re.search(r'\b(what is|what does|about|overview|purpose|explain|summarize|describe).{0,20}(this|repo|project|codebase)\b', q):
        return "overview"
    if re.search(r'\b(how (does|do|to|can)|explain how|walk.?me.?through)\b', q):
        return "explain"
    return "rag"

def search_readme_first(repo_name: str, n: int = 8) -> List[Dict]:
    col = get_or_create_collection(repo_name)
    overview_query = "project overview introduction purpose what this does getting started installation features tech stack"
    vec = embed_query(overview_query)
    res = col.query(query_embeddings=[vec], n_results=n)
    results = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        results.append({"text": doc, "path": meta.get("path", ""), "language": meta.get("language", "")})
    results.sort(key=lambda r: (0 if "readme" in r["path"].lower() else 1 if "/" not in r["path"] else 2))
    return results

# ══════════════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════════════

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
    for c in context_chunks:
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
        task = question

    context_block = f"### Repository Code Context\n\n{context}\n\n" if context else ""

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
        from llama_cpp import Llama
        print(f"[LLM] Loading {MODEL_PATH}…")
        _llm = Llama(
            model_path   = MODEL_PATH,
            n_ctx        = 16384,
            n_gpu_layers = 0, # CPU execution (13th Gen i7 multi-core optimized)
            n_threads    = os.cpu_count() or 4,
            verbose      = False,
        )
        print("[LLM] Model ready.")
    return _llm

def ask(question: str, context_chunks: List[Dict], repo_name: str = None) -> str:
    intent = detect_intent(question)

    if intent in ("self_help", "greeting"):
        context_chunks = []

    if intent == "list_files":
        if repo_name:
            manifest = get_manifest(repo_name)
            if manifest:
                return format_file_tree(manifest)
        paths = sorted({c["path"] for c in context_chunks})
        return "**Files found in context** (full manifest not available):\n" + "\n".join(f"  • `{p}`" for p in paths)

    if intent == "overview" and repo_name:
        context_chunks = search_readme_first(repo_name)

    prompt = _build_prompt(question, context_chunks, intent)
    temp = 0.3 if intent in ("overview", "explain") else 0.15

    if os.path.isfile(MODEL_PATH):
        llm = _get_llm()
        out = llm(
            prompt, max_tokens=1024, temperature=temp, top_p=0.9, top_k=40,
            repeat_penalty=1.1, stop=["<|im_end|>", "<|im_start|>"], echo=False,
        )
        text = out["choices"][0]["text"].strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

    return "❌ No local LLM available. Please ensure the GGUF model file is placed in the folder."

# ══════════════════════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════════════════════

class RAGApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("GitHub Codebase RAG")
        self.geometry("1100x700")
        self.resizable(True, True)
        self._current_repo = None
        self._build_ui()
        self.after(200, self._init_app)

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(sidebar, text="GitHub RAG", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, padx=16, pady=(20, 4))
        ctk.CTkLabel(sidebar, text="Local · Free · Private", font=ctk.CTkFont(size=11), text_color="gray").grid(row=1, column=0, padx=16, pady=(0, 16))

        ctk.CTkLabel(sidebar, text="INDEX A REPO", font=ctk.CTkFont(size=10, weight="bold"), text_color="gray").grid(row=2, column=0, padx=16, sticky="w")
        self.url_entry = ctk.CTkEntry(sidebar, placeholder_text="https://github.com/owner/repo", width=188)
        self.url_entry.grid(row=3, column=0, padx=16, pady=4)
        self.ingest_btn = ctk.CTkButton(sidebar, text="🚀 Ingest Repo", command=self._ingest, width=188)
        self.ingest_btn.grid(row=4, column=0, padx=16, pady=(0, 16), sticky="n")

        ctk.CTkLabel(sidebar, text="INDEXED REPOS", font=ctk.CTkFont(size=10, weight="bold"), text_color="gray").grid(row=5, column=0, padx=16, pady=(8, 4), sticky="w")
        self.repo_list = tk.Listbox(sidebar, bg="#1a1a2e", fg="#e0e0e0", selectbackground="#3b82f6",
                                     relief="flat", borderwidth=0, font=("Segoe UI", 11), activestyle="none")
        self.repo_list.grid(row=6, column=0, padx=10, pady=4, sticky="nsew")
        sidebar.grid_rowconfigure(6, weight=1)
        self.repo_list.bind("<<ListboxSelect>>", self._on_repo_select)

        btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_frame.grid(row=7, column=0, padx=10, pady=8)
        ctk.CTkButton(btn_frame, text="↻ Refresh",    width=82, height=28, command=self._refresh_repos).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="✖ Clear Chat", width=82, height=28, command=self._clear_chat).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="❌ Delete",     width=82, height=28,
                      fg_color="#7f1d1d", hover_color="#991b1b",
                      command=self._delete_repo).pack(side="left", padx=2)

        main = ctk.CTkFrame(self, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew", padx=0)
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        self.status_bar = ctk.CTkLabel(main, text="⏳ Initialising…", font=ctk.CTkFont(size=12),
                                        fg_color="#1e1e2e", corner_radius=6, anchor="w")
        self.status_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 0))

        self.chat_box = ctk.CTkTextbox(main, wrap="word", font=ctk.CTkFont(size=13), state="disabled")
        self.chat_box.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=12, pady=8)

        self.query_entry = ctk.CTkEntry(main, placeholder_text="Ask about architecture, functions, logic…", font=ctk.CTkFont(size=13))
        self.query_entry.grid(row=2, column=0, sticky="ew", padx=(12, 4), pady=(0, 12))
        self.query_entry.bind("<Return>", lambda e: self._send())
        ctk.CTkButton(main, text="Send →", width=90, command=self._send).grid(row=2, column=1, padx=(0, 12), pady=(0, 12))

    def _init_app(self):
        def _run():
            try:
                list_collections() 
                self._refresh_repos()
                embed_backend = "Jina API"   if JINA_API_KEY   else "Local MiniLM"
                self._set_status(f"✅ Ready  |  LLM: Local GGUF  |  Embeddings: {embed_backend}", "green")
            except Exception as e:
                self._set_status(f"❌ Init error: {e}", "red")
        threading.Thread(target=_run, daemon=True).start()

    def _set_status(self, msg: str, color="white"):
        colors = {"green": "#22c55e", "red": "#ef4444", "orange": "#f97316", "white": "#e0e0e0"}
        self.after(0, lambda: self.status_bar.configure(text=f"  {msg}", text_color=colors.get(color, color)))

    def _append_chat(self, text: str):
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
        sel = self.repo_list.curselection()
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
        self.ingest_btn.configure(state="disabled")
        self._set_status("📥 Fetching repo…", "orange")
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
        finally:
            self.after(0, lambda: self.ingest_btn.configure(state="normal"))

    def _send(self):
        question = self.query_entry.get().strip()
        if not question: return
        if not self._current_repo:
            self._append_chat("\n⚠ Select or ingest a repository first.\n"); return
        self.query_entry.delete(0, "end")
        self._append_chat(f"\n🧑 {question}\n")
        self._set_status("🤖 Thinking…", "orange")
        threading.Thread(target=self._send_worker, args=(question,), daemon=True).start()

    def _send_worker(self, question: str):
        try:
            chunks = search_chunks(question, self._current_repo)
            answer = ask(question, chunks, repo_name=self._current_repo)
            self._append_chat(f"\n🤖 {answer}\n")
            self._set_status("✅ Ready", "green")
        except Exception as e:
            self._append_chat(f"\n❌ Error: {e}\n")
            self._set_status(f"❌ {e}", "red")

# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        app = RAGApp()
        app.mainloop()
    except Exception as e:
        import traceback
        print("\n" + "="*50)
        print("CRITICAL ERROR: The application crashed.")
        traceback.print_exc()
        print("="*50 + "\n")
